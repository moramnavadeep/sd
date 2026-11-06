"""Train Model B (saree region segmentation).

  python training/train_segmentation.py --config training/configs/segmentation.yaml
  python training/train_segmentation.py --config ... --epochs 2 --limit 200   # smoke test

Model selection uses the VALIDATION split only. The test split is touched only by evaluate.py.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import ROOT, load_config, seed_everything, device, load_split, split_summary, new_run_dir, save_json, git_rev
from datasets import SegDataset
from metrics import confusion, per_class_from_confusion, macro
from saree_ai.constants import SEG_CLASSES, SEG_IGNORE, SAREE_CLASS_IDS
from saree_ai.models import build_segmentation


def evaluate_seg(model, loader, dev):
    model.eval()
    cm = np.zeros((len(SEG_CLASSES),) * 2, np.int64)
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(dev)).argmax(1).cpu().numpy()
            cm += confusion(y.numpy(), pred, len(SEG_CLASSES), ignore=SEG_IGNORE)
    pc = per_class_from_confusion(cm, SEG_CLASSES)
    saree_names = [SEG_CLASSES[i] for i in SAREE_CLASS_IDS]
    return {"per_class": pc, "miou": macro(pc, "iou"),
            "miou_saree": float(np.mean([pc[n]["iou"] for n in saree_names if pc[n]["iou"] is not None])) if any(pc[n]["iou"] is not None for n in saree_names) else None,
            "macro_f1": macro(pc, "f1"), "pixel_accuracy": float(np.trace(cm) / max(cm.sum(), 1)), "confusion": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="cap samples per split (smoke tests)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tc = cfg["train"]
    if a.epochs: tc["epochs"] = a.epochs
    seed_everything(cfg["seed"])
    dev = device(tc.get("device", "auto"))
    splits = load_split(cfg["data"]["dirs"], cfg["data"]["split"])
    if a.limit:
        splits = {k: v[: a.limit] for k, v in splits.items()}
    summary = split_summary(splits)
    print("split summary:", summary)
    hw = cfg["input_hw"]
    tr = SegDataset(splits["train"], hw, augment=True, seed=cfg["seed"])
    va = SegDataset(splits["val"], hw, augment=False)
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("need non-empty train and val splits with masks")
    nw = tc.get("num_workers", 0)
    tl = DataLoader(tr, tc["batch_size"], shuffle=True, num_workers=nw, drop_last=True, persistent_workers=nw > 0)
    vl = DataLoader(va, tc["batch_size"], shuffle=False, num_workers=nw)

    model = build_segmentation(cfg["model"]["arch"], **cfg["model"]).to(dev)
    # class weights ~ 1/sqrt(freq) from a sample of training masks
    counts = np.ones(len(SEG_CLASSES))
    for i in range(0, len(tr), max(1, len(tr) // 200)):
        m = tr[i][1].numpy(); counts += np.bincount(m[m != SEG_IGNORE].ravel(), minlength=len(SEG_CLASSES))
    w = 1 / np.sqrt(counts / counts.sum()); w = w / w.mean()
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev), ignore_index=SEG_IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tc["lr"], total_steps=tc["epochs"] * len(tl))
    scaler = torch.amp.GradScaler(enabled=tc.get("amp", True) and dev.type == "cuda")

    run = new_run_dir("segmentation")
    best, history = -1.0, []
    for ep in range(tc["epochs"]):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=scaler.is_enabled()):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        val = evaluate_seg(model, vl, dev)
        score = val["miou_saree"] or 0.0
        history.append({"epoch": ep, "loss": tot / len(tl), "val_miou": val["miou"], "val_miou_saree": score, "sec": time.time() - t0})
        print(history[-1])
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "cfg": cfg}, run / "best.pt")
    save_json(run / "summary.json", {"task": "segmentation", "config": cfg, "split_summary": summary, "history": history,
                                     "best_val_miou_saree": best, "git": git_rev(), "device": str(dev)})
    print("run dir:", run.relative_to(ROOT))


if __name__ == "__main__":
    main()

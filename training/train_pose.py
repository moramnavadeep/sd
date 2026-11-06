"""Train / fine-tune our own pose model (PoseHeatmapNet) - the path to replacing the baseline.

Data options (set in configs/pose_own.yaml -> data.dirs):
  * our own consented saree recordings with landmark annotations (manifest.jsonl)
  * COCO-Keypoints converted with data/convert_coco.py (annotations CC BY 4.0; images keep
    their individual Flickr licences - check before redistribution)
  * pseudo-labels: run data/pseudo_label_pose.py over our OWN consented videos with the
    baseline, then human-review a sample (distillation; record this in the model card)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import ROOT, load_config, seed_everything, device, load_split, split_summary, new_run_dir, save_json, git_rev
from datasets import PoseDataset
from metrics import pck
from saree_ai.models import PoseHeatmapNet


def decode(hm: torch.Tensor) -> np.ndarray:
    n, k, h, w = hm.shape
    idx = hm.view(n, k, -1).argmax(-1)
    return torch.stack([(idx % w).float() * 4 + 2, (idx // w).float() * 4 + 2], -1).cpu().numpy()


def evaluate_pose(model, loader, dev):
    model.eval(); P, G, V, N = [], [], [], []
    with torch.inference_mode():
        for x, hm, wt, kk in loader:
            P.append(decode(model(x.to(dev)))); G.append(kk[..., :2].numpy()); V.append((kk[..., 2] > 0).numpy())
            k = kk.numpy()
            N.append(np.linalg.norm((k[:, 5, :2] + k[:, 6, :2]) / 2 - (k[:, 11, :2] + k[:, 12, :2]) / 2, axis=-1) + 1e-6)
    return pck(np.concatenate(P), np.concatenate(G), np.concatenate(V), np.concatenate(N), 0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--epochs", type=int); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config); tc = cfg["train"]
    if a.epochs: tc["epochs"] = a.epochs
    seed_everything(cfg["seed"]); dev = device(tc.get("device", "auto"))
    splits = load_split(cfg["data"]["dirs"], cfg["data"]["split"])
    if a.limit: splits = {k: v[: a.limit] for k, v in splits.items()}
    summary = split_summary(splits); print("split summary:", summary)
    hw = cfg["input_hw"]; nw = tc.get("num_workers", 0)
    tl = DataLoader(PoseDataset(splits["train"], hw, augment=True, seed=cfg["seed"]), tc["batch_size"], shuffle=True, num_workers=nw, drop_last=True)
    vl = DataLoader(PoseDataset(splits["val"], hw), tc["batch_size"], num_workers=nw)
    model = PoseHeatmapNet(imagenet_backbone=cfg["model"].get("imagenet_backbone", True)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tc["epochs"])
    run = new_run_dir("pose_own"); best, history = -1.0, []
    for ep in range(tc["epochs"]):
        model.train(); tot = 0.0; t0 = time.time()
        for x, hm, wt, _ in tl:
            x, hm, wt = x.to(dev), hm.to(dev), wt.to(dev)
            loss = (((model(x) - hm) ** 2).mean((2, 3)) * wt).sum() / wt.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        val = evaluate_pose(model, vl, dev)
        history.append({"epoch": ep, "loss": tot / max(1, len(tl)), "val_pck@0.2": val["pck"], "sec": time.time() - t0}); print(history[-1])
        if val["pck"] > best:
            best = val["pck"]; torch.save({"model": model.state_dict(), "cfg": cfg}, run / "best.pt")
    save_json(run / "summary.json", {"task": "pose_own", "config": cfg, "split_summary": summary, "history": history,
                                     "best_val_pck": best, "git": git_rev(), "device": str(dev)})
    print("run dir:", run.relative_to(ROOT))


if __name__ == "__main__":
    main()

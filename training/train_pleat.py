"""Train Model C (PleatNet: count / quality / secured / visible on pleat-region crops)."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import ROOT, load_config, seed_everything, device, load_split, split_summary, new_run_dir, save_json, git_rev
from datasets import PleatDataset
from metrics import classification_report, binary_report
from saree_ai.constants import PLEAT_QUALITY
from saree_ai.models import PleatNet


def pleat_loss(out, tgt, q):
    count, ql, sec, vis = out
    c_t, s_t, v_t = tgt[:, 0], tgt[:, 1], tgt[:, 2]
    loss = F.binary_cross_entropy_with_logits(vis[:, 0], v_t)
    m = c_t >= 0
    if m.any(): loss = loss + 0.2 * F.smooth_l1_loss(count[m, 0], c_t[m])
    if (q >= 0).any(): loss = loss + F.cross_entropy(ql, q, ignore_index=-100)
    m = s_t >= 0
    if m.any(): loss = loss + F.binary_cross_entropy_with_logits(sec[m, 0], s_t[m])
    return loss


def evaluate_pleat(model, loader, dev):
    model.eval()
    C, CT, Q, QT, S, ST, V, VT = [], [], [], [], [], [], [], []
    with torch.inference_mode():
        for x, tgt, q in loader:
            count, ql, sec, vis = model(x.to(dev))
            tgt = tgt.numpy(); q = q.numpy()
            m = tgt[:, 0] >= 0
            C += count[:, 0].cpu().numpy()[m].tolist(); CT += tgt[m, 0].tolist()
            mq = q >= 0
            Q += ql.argmax(1).cpu().numpy()[mq].tolist(); QT += q[mq].tolist()
            ms = tgt[:, 1] >= 0
            S += (sec[:, 0].cpu().numpy()[ms] > 0).tolist(); ST += (tgt[ms, 1] > 0.5).tolist()
            V += (vis[:, 0].cpu().numpy() > 0).tolist(); VT += (tgt[:, 2] > 0.5).tolist()
    C, CT = np.array(C), np.array(CT)
    pos = CT > 0
    return {
        "count_mae": float(np.abs(C[pos] - CT[pos]).mean()) if pos.any() else None,
        "count_acc_within_1": float((np.abs(np.round(C[pos]) - CT[pos]) <= 1).mean()) if pos.any() else None,
        "quality": classification_report(QT, Q, PLEAT_QUALITY) if QT else None,
        "secured": binary_report(ST, S) if ST else None,
        "visible": binary_report(VT, V),
    }


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
    tr = PleatDataset(splits["train"], cfg["input_size"], augment=True, seed=cfg["seed"])
    va = PleatDataset(splits["val"], cfg["input_size"])
    nw = tc.get("num_workers", 0)
    tl = DataLoader(tr, tc["batch_size"], shuffle=True, num_workers=nw, drop_last=True, persistent_workers=nw > 0)
    vl = DataLoader(va, tc["batch_size"], num_workers=nw)
    model = PleatNet(cfg["model"].get("width", 32)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tc["epochs"])
    run = new_run_dir("pleat"); best, history = -1e9, []
    for ep in range(tc["epochs"]):
        model.train(); tot = 0.0; t0 = time.time()
        for x, tgt, q in tl:
            loss = pleat_loss(model(x.to(dev)), tgt.to(dev), q.to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        val = evaluate_pleat(model, vl, dev)
        score = (val["quality"]["macro_f1"] or 0) if val["quality"] else 0
        score -= 0.05 * (val["count_mae"] or 0)
        history.append({"epoch": ep, "loss": tot / max(1, len(tl)), "val_quality_f1": val["quality"]["macro_f1"] if val["quality"] else None,
                        "val_count_mae": val["count_mae"], "sec": time.time() - t0})
        print(history[-1])
        if score > best:
            best = score; torch.save({"model": model.state_dict(), "cfg": cfg}, run / "best.pt")
    save_json(run / "summary.json", {"task": "pleat", "config": cfg, "split_summary": summary, "history": history,
                                     "git": git_rev(), "device": str(dev)})
    print("run dir:", run.relative_to(ROOT))


if __name__ == "__main__":
    main()

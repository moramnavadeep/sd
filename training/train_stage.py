"""Train Model D (draping-stage MLP on interpretable frame features).

Features are computed from ground-truth annotations (teacher forcing) with Gaussian feature
noise to approximate model errors. Recommended upgrade once Models B/C are trained on real data:
recompute features from their PREDICTIONS (--features-from-models) so train/test distributions match.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import ROOT, load_config, seed_everything, device, load_split, split_summary, new_run_dir, save_json, git_rev
from datasets import StageFeatureDataset
from metrics import classification_report
from saree_ai.constants import DRAPE_STAGES
from saree_ai.models import StageMLP


def evaluate_stage(model, ds, mu, sd, dev):
    model.eval()
    X = torch.stack([ds[i][0] for i in range(len(ds))]) if len(ds) else torch.zeros(0)
    y = [int(ds[i][1]) for i in range(len(ds))]
    if not y:
        return None
    with torch.inference_mode():
        p = model(((X - mu) / sd).to(dev)).argmax(1).cpu().numpy()
    return classification_report(y, p.tolist(), DRAPE_STAGES)


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
    cache: dict = {}
    tr = StageFeatureDataset(splits["train"], cache, noise=tc.get("feature_noise", 0.03), seed=cfg["seed"])
    va = StageFeatureDataset(splits["val"], cache)
    X = np.stack([f for f, _ in tr.items])
    mu, sd = torch.tensor(X.mean(0)), torch.tensor(X.std(0) + 1e-3)
    model = StageMLP(hidden=cfg["model"]["hidden"], dropout=cfg["model"]["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    counts = np.bincount([y for _, y in tr.items], minlength=len(DRAPE_STAGES)) + 1
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor((counts.sum() / counts) / len(counts), dtype=torch.float32, device=dev))
    tl = DataLoader(tr, tc["batch_size"], shuffle=True, drop_last=False)
    run = new_run_dir("stage"); best, history = -1.0, []
    for ep in range(tc["epochs"]):
        model.train(); tot = 0.0; t0 = time.time()
        for x, y in tl:
            loss = crit(model(((x - mu) / sd).to(dev)), y.to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += loss.item()
        val = evaluate_stage(model, va, mu, sd, dev)
        f1 = val["macro_f1"] or 0
        history.append({"epoch": ep, "loss": tot / max(1, len(tl)), "val_macro_f1": f1, "val_acc": val["accuracy"], "sec": time.time() - t0})
        if ep % 10 == 0 or ep == tc["epochs"] - 1: print(history[-1])
        if f1 > best:
            best = f1
            torch.save({"model": model.state_dict(), "cfg": cfg, "feature_norm": {"mean": mu.tolist(), "std": sd.tolist()}}, run / "best.pt")
    save_json(run / "summary.json", {"task": "stage", "config": cfg, "split_summary": summary, "history": history,
                                     "best_val_macro_f1": best, "git": git_rev(), "device": str(dev)})
    print("run dir:", run.relative_to(ROOT))


if __name__ == "__main__":
    main()

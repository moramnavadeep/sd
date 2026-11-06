"""Evaluate a trained run on the held-out TEST split (participant-level) and write a report.

  python training/evaluate.py --run runs/segmentation/<stamp> [--split test]

If the test split contains no real participants the report says so explicitly and the
metrics are labelled as NOT indicative of real-world performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import ROOT, load_split, split_summary, save_json, device
from datasets import SegDataset, PleatDataset, StageFeatureDataset, PoseDataset
from saree_ai.models import build_segmentation, PleatNet, StageMLP, PoseHeatmapNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    run = ROOT / a.run
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]; task = json.loads((run / "summary.json").read_text())["task"]
    dev = device()
    splits = load_split(cfg["data"]["dirs"], cfg["data"]["split"])
    summary = split_summary(splits)
    rows = splits[a.split]
    bs = cfg["train"]["batch_size"]
    note = None
    if summary[a.split]["real"] == 0:
        note = (f"The {a.split} split contains NO real data. These numbers measure fit to synthetic renderings only "
                "and say nothing about accuracy on real users.")
    if task == "segmentation":
        from train_segmentation import evaluate_seg
        m = build_segmentation(cfg["model"]["arch"], **cfg["model"]); m.load_state_dict(ck["model"]); m.to(dev)
        res = evaluate_seg(m, DataLoader(SegDataset(rows, cfg["input_hw"]), bs), dev)
        headline = {"miou": res["miou"], "miou_saree": res["miou_saree"], "macro_f1": res["macro_f1"]}
    elif task == "pleat":
        from train_pleat import evaluate_pleat
        m = PleatNet(cfg["model"].get("width", 32)); m.load_state_dict(ck["model"]); m.to(dev)
        res = evaluate_pleat(m, DataLoader(PleatDataset(rows, cfg["input_size"]), bs), dev)
        headline = {"quality_macro_f1": res["quality"]["macro_f1"] if res["quality"] else None,
                    "count_mae": res["count_mae"], "count_acc_within_1": res["count_acc_within_1"],
                    "secured_f1": res["secured"]["f1"] if res["secured"] else None}
    elif task == "stage":
        from train_stage import evaluate_stage
        m = StageMLP(hidden=cfg["model"]["hidden"], dropout=cfg["model"]["dropout"]); m.load_state_dict(ck["model"]); m.to(dev)
        fn = ck["feature_norm"]
        res = evaluate_stage(m, StageFeatureDataset(rows), torch.tensor(fn["mean"]), torch.tensor(fn["std"]), dev)
        headline = {"accuracy": res["accuracy"], "macro_f1": res["macro_f1"]} if res else {}
    elif task == "pose_own":
        from train_pose import evaluate_pose
        m = PoseHeatmapNet(imagenet_backbone=False); m.load_state_dict(ck["model"]); m.to(dev)
        res = evaluate_pose(m, DataLoader(PoseDataset(rows, cfg["input_hw"]), bs), dev)
        headline = {"pck@0.2": res["pck"]}
    else:
        raise SystemExit(f"unknown task {task}")
    report = {"task": task, "split": a.split, "split_summary": summary, "headline": headline, "details": res, "note": note}
    save_json(run / f"eval_{a.split}.json", report)
    md = [f"# Evaluation: {task} ({a.split} split)", ""]
    if note: md += [f"> **{note}**", ""]
    md += ["| split | samples | participants | real | synthetic |", "|---|---|---|---|---|"]
    md += [f"| {k} | {v['samples']} | {v['participants']} | {v['real']} | {v['synthetic']} |" for k, v in summary.items()]
    md += ["", "| metric | value |", "|---|---|"] + [f"| {k} | {'n/a' if v is None else round(v, 4)} |" for k, v in headline.items()]
    if res and "per_class" in res:
        md += ["", "| class | precision | recall | F1 | IoU | support |", "|---|---|---|---|---|---|"]
        f = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        md += [f"| {c} | {f(v['precision'])} | {f(v['recall'])} | {f(v['f1'])} | {f(v['iou'])} | {v['support']} |" for c, v in res["per_class"].items()]
    (run / f"eval_{a.split}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

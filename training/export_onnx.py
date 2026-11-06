"""Export a trained run to ONNX, verify parity with PyTorch, benchmark ONNX Runtime latency on
THIS machine, and register the model in models/registry.json with a DERIVED status.

  python training/export_onnx.py --run runs/segmentation/<stamp>

Status derivation (common.derive_status): no real training data -> smoke_test_synthetic;
no real test participants -> experimental; real test but below targets -> experimental;
otherwise validated. The status is never set by hand.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np
import torch

from common import ROOT, derive_status, save_json
from saree_ai import registry
from saree_ai.models import build_segmentation, PleatNet, StageMLP, PoseHeatmapNet
from saree_ai.features import FEATURE_DIM

LIMITATIONS = {
    "segmentation": ["Trained on the data listed in dataset; untested on fabrics/lighting not represented there.",
                     "Pixel classes are semantic regions; individual overlapping pleats are not separated.",
                     "Sheer/translucent fabric over other clothing is inherently ambiguous."],
    "pleat": ["Operates on a crop located by segmentation or pose; a wrong crop gives a wrong answer.",
              "Count is unreliable when folds overlap heavily (annotators mark count_certain=false)."],
    "stage": ["Features from ground-truth annotations at training time; predictions at runtime (domain gap).",
              "Front camera only; stages that differ mainly at the back are not separable."],
    "pose_own": ["Top-down: depends on the previous frame's box; recovers via full-frame crop."],
}


def build(task, cfg):
    if task == "segmentation":
        h, w = cfg["input_hw"]
        return build_segmentation(cfg["model"]["arch"], **{**cfg["model"], "imagenet_backbone": False}), torch.randn(1, 3, h, w), ["logits"], {"input_hw": [h, w]}
    if task == "pleat":
        s = cfg["input_size"]
        return PleatNet(cfg["model"].get("width", 32)), torch.randn(1, 3, s, s), ["count", "quality", "secured", "visible"], {"input_size": s}
    if task == "stage":
        return StageMLP(hidden=cfg["model"]["hidden"], dropout=cfg["model"]["dropout"]), torch.randn(1, FEATURE_DIM), ["logits"], {}
    if task == "pose_own":
        h, w = cfg["input_hw"]
        return PoseHeatmapNet(imagenet_backbone=False), torch.randn(1, 3, h, w), ["heatmaps"], {"input_hw": [h, w]}
    raise SystemExit(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    run = ROOT / a.run
    summary = json.loads((run / "summary.json").read_text())
    task = summary["task"]
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model, dummy, out_names, extra = build(task, cfg)
    model.load_state_dict(ck["model"]); model.eval()
    onnx_path = run / f"{task}.onnx"
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["input"], output_names=out_names,
                      opset_version=17, dynamic_axes={"input": {0: "batch"}}, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        ref = model(dummy)
    ref = ref if isinstance(ref, (tuple, list)) else [ref]
    got = sess.run(None, {"input": dummy.numpy()})
    max_err = max(float(np.abs(r.numpy() - g).max()) for r, g in zip(ref, got))
    if max_err > 1e-3:
        raise SystemExit(f"ONNX parity check failed: max abs diff {max_err}")
    x = dummy.numpy()
    for _ in range(5): sess.run(None, {"input": x})
    ts = []
    for _ in range(50):
        t0 = time.perf_counter(); sess.run(None, {"input": x}); ts.append((time.perf_counter() - t0) * 1000)
    lat = {"provider": "CPUExecutionProvider", "median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "runs": 50}

    ev_file = run / "eval_test.json"
    ev = json.loads(ev_file.read_text()) if ev_file.exists() else None
    targets = cfg.get("acceptance_targets", {})
    metrics = (ev or {}).get("headline", {})
    if ev is None:
        status, why = derive_status(summary["split_summary"], {}, {"_test_eval_missing": 1})
        if status == "validated": status, why = "experimental", "evaluate.py has not been run on the test split"
    else:
        status, why = derive_status(ev["split_summary"], metrics, targets)

    dest = registry.MODELS_DIR / task / f"{task}-{run.name}.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, dest)
    entry = {
        "task": task, "version": run.name, "file": str(dest.relative_to(ROOT)).replace("\\", "/"),
        "status": status, "status_reason": why,
        "trained_from": "scratch" if task in ("pleat", "stage") or cfg["model"].get("arch") == "tiny_unet" else "fine-tuned from torchvision ImageNet-1k MobileNetV3-Large backbone",
        "architecture": cfg["model"], "training_config": cfg["_config_path"], "config_sha256": cfg["_config_sha256"],
        "dataset": {"dirs": cfg["data"]["dirs"], "split_file": cfg["data"]["split"], "split_summary": summary["split_summary"],
                    "dataset_versions": cfg["data"].get("versions")},
        "metrics_test": metrics or None, "acceptance_targets": targets, "onnx_parity_max_abs_diff": max_err,
        "inference_latency": lat, "limitations": LIMITATIONS.get(task, []), "run_dir": a.run, **extra,
    }
    if task == "stage":
        entry["feature_norm"] = ck["feature_norm"]
    save_json(run / "export.json", entry)
    if not a.no_register:
        registry.register(task, entry)
    print(json.dumps({k: entry[k] for k in ("task", "version", "status", "status_reason", "file")}, indent=2))
    print("latency:", lat, "parity:", max_err)


if __name__ == "__main__":
    main()

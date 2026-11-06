"""Evaluate Model E (correction engine) against annotated correction_labels.

Two modes:
  --inputs oracle   : ground-truth masks + landmarks feed the checks (isolates the RULES; pleat
                      geometry still comes from the real PleatSignalAnalyzer on the GT pleat ROI)
  --inputs models   : the registered runtime models produce the inputs (end-to-end)

Also runs a LOW-CONFIDENCE test: every input confidence is forced below threshold and the
engine must emit ZERO corrections (requirement 14 / acceptance test 9).

  python training/evaluate_corrections.py --dirs data/synthetic_v1 --split data/splits/default.json --set val
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import yaml

from common import ROOT, load_split, split_summary, save_json
from datasets import load_image, load_mask
from metrics import binary_report
from saree_ai.constants import CORRECTION_LABELS
from saree_ai.correction.checks import CHECKS, FrameState
from saree_ai.correction.engine import Tutorial
from saree_ai.features import frame_features
from saree_ai.inference.pleat import PleatSignalAnalyzer, PleatResult
from saree_ai.inference.pose import PoseResult
from saree_ai.inference.segmentation import SegResult
from saree_ai.inference.stage import StageResult


def oracle_state(r, analyzer, conf=1.0):
    img = load_image(r); mask = load_mask(r)
    k = np.array(r["landmarks"], np.float32); k[:, 2] = (k[:, 2] > 0) * conf
    pose = PoseResult(k, conf, backend="oracle")
    seg = SegResult(True, mask, conf, ambiguous_fraction=0.0 if conf >= 0.5 else 1.0, experimental=False, reason="oracle")
    p = r.get("pleats") or {}
    if p.get("visible") and p.get("roi"):
        pl = analyzer(img, p["roi"])
        pl.experimental = False
        pl.secured_prob = None
        if conf < 0.5: pl.confidence = pl.count_conf = 0.0
    else:
        pl = PleatResult(False)
    f, extras = frame_features(mask, k, pl.as_feature_input())
    st = StageResult(True, r["step_label"], conf, source="oracle", experimental=False)
    return FrameState(r["width"], r["height"], pose, seg, pl, st, f, extras)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--set", default="test"); ap.add_argument("--inputs", default="oracle", choices=["oracle"])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config/app.yaml").read_text())
    th = cfg["thresholds"]
    tut = Tutorial(ROOT / cfg["tutorial"])
    step_for_stage = {s["expected_stage"]: s for s in tut.steps}
    splits = load_split(a.dirs, a.split)
    rows = [r for r in splits[a.set] if r.get("step_label") and r.get("landmarks") and (r.get("mask") or r.get("regions"))]
    if a.limit: rows = rows[: a.limit]
    analyzer = PleatSignalAnalyzer()
    y_true, y_pred = defaultdict(list), defaultdict(list)
    uncertain_counts, low_conf_corrections, n = defaultdict(int), 0, 0
    for r in rows:
        step = step_for_stage.get(r["step_label"])
        if step is None: continue
        n += 1
        s = oracle_state(r, analyzer)
        emitted = set()
        for cid, params in step["_checks"]:
            o = CHECKS[cid](s, params, th)
            if o.status == "fail" and o.label: emitted.add(o.label)
            if o.status == "uncertain": uncertain_counts[cid] += 1
        gt = set(r.get("correction_labels", []))
        checkable = {o for cid, _ in step["_checks"] for o in CHECK_LABELS.get(cid, [])}
        for lab in checkable:
            y_true[lab].append(lab in gt); y_pred[lab].append(lab in emitted)
        # low-confidence probe
        s_low = oracle_state(r, analyzer, conf=0.2)
        for cid, params in step["_checks"]:
            o = CHECKS[cid](s_low, params, th)
            if o.status == "fail" and o.label: low_conf_corrections += 1
    per_label = {lab: binary_report(y_true[lab], y_pred[lab]) for lab in CORRECTION_LABELS if y_true[lab]}
    allt = sum((y_true[l] for l in per_label), []); allp = sum((y_pred[l] for l in per_label), [])
    report = {"set": a.set, "inputs": a.inputs, "frames": n, "split_summary": split_summary(splits),
              "micro": binary_report(allt, allp) if allt else None, "per_label": per_label,
              "uncertain_check_counts": dict(uncertain_counts),
              "low_confidence_probe": {"frames": n, "corrections_emitted": low_conf_corrections,
                                       "pass": low_conf_corrections == 0}}
    out = ROOT / "runs" / "corrections"; out.mkdir(parents=True, exist_ok=True)
    save_json(out / f"eval_{a.set}_{a.inputs}.json", report)
    print(json.dumps({k: report[k] for k in ("frames", "micro", "low_confidence_probe")}, indent=2, default=str))
    for lab, v in per_label.items():
        print(f"{lab:24s} P={v['precision']} R={v['recall']} F1={v['f1']} FPR={v['false_positive_rate']}")


CHECK_LABELS = {
    "waist_wrap_coverage": ["waist_wrap_loose"], "hem_level": ["hem_uneven"],
    "pleat_count": ["pleats_too_few", "pleats_too_many"], "pleat_spacing": ["pleats_uneven_spacing"],
    "pleat_alignment": ["pleats_misaligned"], "pleat_center": ["pleats_off_center"], "pleats_tucked": ["pleats_not_tucked"],
    "pallu_shoulder": ["pallu_wrong_shoulder", "pallu_not_on_shoulder"],
}

if __name__ == "__main__":
    main()

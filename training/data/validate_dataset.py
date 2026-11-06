"""Validate a dataset directory against the schema and the labelling standard.

  python training/data/validate_dataset.py data/real_v1 [--strict]

Checks: JSON schema; files exist and dimensions match; mask values are valid class ids;
landmarks inside the image; pleat ROI inside the image; consent_id present for REAL data;
synthetic data clearly flagged; duplicate ids; per-attribute coverage report (colours, fabrics,
lighting, angles, step labels, correction labels) so gaps in diversity are visible.
Exit code 1 if any error is found.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from saree_ai.constants import SEG_CLASSES, SEG_IGNORE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    ap.add_argument("--check-pixels", type=int, default=200, help="open at most N images/masks for pixel checks (0 = all)")
    a = ap.parse_args()
    import jsonschema
    schema = json.loads((ROOT / "saree_ai/schema/sample.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    root = Path(a.dir) if Path(a.dir).is_absolute() else ROOT / a.dir
    errors, warnings, seen = [], [], set()
    cov = {k: Counter() for k in ("source", "lighting", "camera_angle", "darkness", "fabric", "pattern", "transparency",
                                   "step_label", "correction", "pleat_quality")}
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, r in enumerate(rows):
        rid = r.get("id", f"line{i + 1}")
        for e in validator.iter_errors(r):
            errors.append(f"{rid}: schema: {'/'.join(map(str, e.path))} {e.message}")
        if rid in seen: errors.append(f"{rid}: duplicate id")
        seen.add(rid)
        src = r.get("source", {})
        cov["source"][src.get("type")] += 1
        if src.get("type") == "real" and not src.get("consent_id"):
            errors.append(f"{rid}: real sample without consent_id - do not train on it")
        if src.get("type") == "synthetic" and not str(r.get("participant_id", "")).startswith("synth"):
            warnings.append(f"{rid}: synthetic participant ids should start with 'synth' to avoid confusion")
        c = r.get("capture") or {}; s = r.get("saree") or {}
        cov["lighting"][c.get("lighting")] += 1; cov["camera_angle"][c.get("camera_angle")] += 1
        for k in ("darkness", "fabric", "pattern", "transparency"): cov[k][s.get(k)] += 1
        cov["step_label"][r.get("step_label")] += 1
        for cl in r.get("correction_labels", []): cov["correction"][cl] += 1
        p = r.get("pleats") or {}
        cov["pleat_quality"][p.get("quality")] += 1
        W, Hh = r.get("width", 0), r.get("height", 0)
        if r.get("landmarks"):
            lm = np.array(r["landmarks"])
            vis = lm[:, 2] > 0
            if np.any((lm[vis, 0] < -1) | (lm[vis, 0] > W + 1) | (lm[vis, 1] < -1) | (lm[vis, 1] > Hh + 1)):
                errors.append(f"{rid}: labelled landmark outside image")
        if p.get("visible") and p.get("roi"):
            x0, y0, x1, y1 = p["roi"]
            if not (x0 < x1 and y0 < y1 and x0 >= -2 and y0 >= -2 and x1 <= W + 2 and y1 <= Hh + 2):
                errors.append(f"{rid}: invalid pleat roi {p['roi']}")
            if p.get("count") is not None and p.get("fold_lines") and abs(len(p["fold_lines"]) - 1 - p["count"]) > 1:
                warnings.append(f"{rid}: pleat count {p['count']} disagrees with {len(p['fold_lines'])} fold lines")
        if p.get("visible") and p.get("quality") is None:
            warnings.append(f"{rid}: visible pleats without a quality label")
        if a.check_pixels == 0 or i < a.check_pixels:
            img_p = root / r.get("image", "")
            img = cv2.imread(str(img_p)) if img_p.exists() else None
            if img is None:
                errors.append(f"{rid}: image missing/unreadable {r.get('image')}")
            elif img.shape[1] != W or img.shape[0] != Hh:
                errors.append(f"{rid}: image size {img.shape[1]}x{img.shape[0]} != declared {W}x{Hh}")
            if r.get("mask"):
                m = cv2.imread(str(root / r["mask"]), cv2.IMREAD_UNCHANGED)
                if m is None:
                    errors.append(f"{rid}: mask missing {r['mask']}")
                else:
                    if m.ndim != 2: errors.append(f"{rid}: mask must be single-channel")
                    bad = set(np.unique(m).tolist()) - set(range(len(SEG_CLASSES))) - {SEG_IGNORE}
                    if bad: errors.append(f"{rid}: mask has invalid values {sorted(bad)}")
                    if img is not None and m.shape[:2] != img.shape[:2]: errors.append(f"{rid}: mask/image size mismatch")
    report = {"dir": str(root), "samples": len(rows), "participants": len({r.get("participant_id") for r in rows}),
              "errors": len(errors), "warnings": len(warnings), "coverage": {k: dict(v) for k, v in cov.items()}}
    print(json.dumps(report, indent=2, default=str))
    for e in errors[:50]: print("ERROR", e)
    for w in warnings[:20]: print("WARN ", w)
    if errors or (a.strict and warnings):
        sys.exit(1)


if __name__ == "__main__":
    main()

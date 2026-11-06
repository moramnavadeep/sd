"""Dataset preparation.

Subcommands
  split     participant-level train/val/test split across one or more dataset dirs
  rasterize rasterise polygon `regions` into mask PNGs (for annotation tools that export polygons)

Split rules (requirement 12):
  * every frame of a participant goes to exactly one split (no near-duplicate leakage)
  * synthetic participants are only ever placed in train (or val with --synthetic-in-val)
  * test is real-only; if there are fewer than --min-test-participants real participants the
    test split is left EMPTY and the split file says so, rather than silently testing on synthetic
  * the split is deterministic given the seed and the set of participant ids

  python training/data/prepare_dataset.py split --dirs data/synthetic_v1 data/real_v1 --out data/splits/default.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "training"))
from common import read_manifest  # noqa: E402
from datasets import load_mask  # noqa: E402


def _rank(pid: str, seed: int) -> float:
    return int(hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()[:12], 16) / 16 ** 12


def cmd_split(a):
    rows = []
    for d in a.dirs:
        rows += read_manifest(ROOT / d)
    by_pid = {}
    for r in rows:
        t = r["source"]["type"]
        prev = by_pid.setdefault(r["participant_id"], t)
        if prev != t:
            raise SystemExit(f"participant {r['participant_id']} mixes real and synthetic samples")
    real = sorted([p for p, t in by_pid.items() if t == "real"], key=lambda p: _rank(p, a.seed))
    synth = sorted([p for p, t in by_pid.items() if t == "synthetic"], key=lambda p: _rank(p, a.seed))
    parts = {"train": [], "val": [], "test": []}
    notes = []
    n = len(real)
    if n:
        n_test = int(round(n * a.test_frac)) if n >= a.min_test_participants else 0
        n_val = max(1, int(round(n * a.val_frac))) if n - n_test >= 2 else 0
        parts["test"] = real[:n_test]; parts["val"] = real[n_test:n_test + n_val]; parts["train"] = real[n_test + n_val:]
        if n_test == 0:
            notes.append(f"Only {n} real participants (< {a.min_test_participants}); test split left EMPTY.")
    else:
        notes.append("No real participants: test split is EMPTY. Any metrics are synthetic-only.")
    if synth:
        n_sv = int(round(len(synth) * a.val_frac)) if (a.synthetic_in_val or not parts["val"]) else 0
        parts["val"] += synth[:n_sv]; parts["train"] += synth[n_sv:]
        if n_sv:
            notes.append(f"{n_sv} synthetic participants placed in val (val is not real-only).")
    counts = {k: Counter(r["source"]["type"] for r in rows if r["participant_id"] in set(v)) for k, v in parts.items()}
    out = {"seed": a.seed, "dirs": a.dirs, "participants": parts,
           "sample_counts": {k: dict(v) for k, v in counts.items()},
           "participant_counts": {k: len(v) for k, v in parts.items()}, "notes": notes,
           "rule": "participant-level; synthetic never in test"}
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("participant_counts", "sample_counts", "notes")}, indent=2))


def cmd_rasterize(a):
    root = ROOT / a.dir
    rows = read_manifest(root)
    (root / "masks").mkdir(exist_ok=True)
    n = 0
    for r in rows:
        if r.get("mask") or not r.get("regions"):
            continue
        m = load_mask(r)
        rel = f"masks/{r['id']}.png"
        cv2.imwrite(str(root / rel), m.astype(np.uint8))
        r["mask"] = rel; n += 1
    with open(root / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            r.pop("_root", None); fh.write(json.dumps(r) + "\n")
    print(f"rasterised {n} masks")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--dirs", nargs="+", required=True); s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--val-frac", type=float, default=0.15)
    s.add_argument("--test-frac", type=float, default=0.2); s.add_argument("--min-test-participants", type=int, default=5)
    s.add_argument("--synthetic-in-val", action="store_true")
    r = sub.add_parser("rasterize"); r.add_argument("--dir", required=True)
    a = ap.parse_args()
    {"split": cmd_split, "rasterize": cmd_rasterize}[a.cmd](a)


if __name__ == "__main__":
    main()

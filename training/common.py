"""Shared training utilities: config, seeding, manifests, splits, run directories."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch_home"))  # keep weight caches inside the project


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return cfg


def seed_everything(seed: int):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device(pref: str = "auto"):
    import torch
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_manifest(root: Path) -> list[dict]:
    rows = []
    with open(root / "manifest.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                r["_root"] = str(root)
                rows.append(r)
    return rows


def load_split(dataset_dirs: list[str], split_file: str | None) -> dict[str, list[dict]]:
    """Returns {'train': [...], 'val': [...], 'test': [...]} using a participant-level split file
    produced by training/data/prepare_dataset.py. Synthetic rows can only be in train/val."""
    rows = []
    for d in dataset_dirs:
        rows += read_manifest(ROOT / d)
    if split_file is None:
        raise SystemExit("A split file is required (run training/data/prepare_dataset.py split ...).")
    sp = json.loads((ROOT / split_file).read_text(encoding="utf-8"))
    out = {k: [] for k in ("train", "val", "test")}
    where = {pid: k for k in out for pid in sp["participants"][k]}
    for r in rows:
        k = where.get(r["participant_id"])
        if k is None:
            continue
        if k == "test" and r["source"]["type"] == "synthetic":
            raise SystemExit(f"split error: synthetic sample {r['id']} in test split")
        out[k].append(r)
    return out


def split_summary(splits: dict) -> dict:
    s = {}
    for k, rows in splits.items():
        s[k] = {"samples": len(rows), "participants": len({r["participant_id"] for r in rows}),
                "real": sum(r["source"]["type"] == "real" for r in rows),
                "synthetic": sum(r["source"]["type"] == "synthetic" for r in rows)}
    return s


def git_rev() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def new_run_dir(task: str) -> Path:
    d = ROOT / "runs" / task / time.strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


def derive_status(summary: dict, metrics: dict, targets: dict) -> tuple[str, str]:
    """Registry status is DERIVED from data provenance and metrics, never chosen by hand."""
    if summary["train"]["real"] == 0:
        return "smoke_test_synthetic", "trained without any real data"
    if summary["test"]["real"] == 0:
        return "experimental", "no real participant-held-out test data"
    failed = [k for k, v in targets.items() if metrics.get(k) is None or metrics[k] < v]
    if failed:
        return "experimental", "below acceptance targets: " + ", ".join(failed)
    return "validated", "met all acceptance targets on real held-out participants"


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")

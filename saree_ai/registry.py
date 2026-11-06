"""Model version registry (models/registry.json).

Every exported model gets an entry with its provenance, the exact dataset version it was trained
on, its evaluation numbers and a STATUS. The runtime refuses to load anything below the status
it is configured to accept, so a smoke-test model can never silently drive user-facing
corrections.

Status ladder (lowest -> highest):
  untrained              architecture only, no weights
  smoke_test_synthetic   trained on synthetic data only to prove the pipeline runs; NOT a model
  experimental           trained on real data, evaluated, but below acceptance targets
  validated              trained on real data and met acceptance targets on a participant-held-out test set
  pretrained_baseline    third-party pretrained weights we did not train (e.g. pose baseline)
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

STATUS_RANK = {
    "untrained": 0,
    "smoke_test_synthetic": 1,
    "experimental": 2,
    "validated": 3,
    "pretrained_baseline": 3,
}


def sha256_file(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"models": {}}


def save_registry(reg: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    tmp.replace(REGISTRY_PATH)


def register(task: str, entry: dict, make_active: bool = True) -> dict:
    """Add a version for `task` (segmentation|pleat|stage|pose_own|pose_baseline)."""
    reg = load_registry()
    t = reg["models"].setdefault(task, {"active": None, "versions": {}})
    version = entry.get("version") or time.strftime("%Y%m%d-%H%M%S")
    entry = {**entry, "version": version, "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if entry.get("file"):
        p = ROOT / entry["file"]
        if p.exists():
            entry["sha256"] = sha256_file(p)
    t["versions"][version] = entry
    if make_active:
        t["active"] = version
    save_registry(reg)
    return entry


def active(task: str) -> dict | None:
    t = load_registry()["models"].get(task)
    if not t or not t.get("active"):
        return None
    return t["versions"].get(t["active"])


def loadable(entry: dict | None, min_status: str) -> tuple[bool, str]:
    """Can this entry be used at runtime given the configured minimum status?"""
    if entry is None:
        return False, "no model registered"
    st = entry.get("status", "untrained")
    if STATUS_RANK.get(st, 0) < STATUS_RANK[min_status]:
        return False, f"status '{st}' is below required '{min_status}'"
    f = ROOT / entry.get("file", "")
    if not entry.get("file") or not f.exists():
        return False, f"weights file missing: {entry.get('file')}"
    if entry.get("sha256") and sha256_file(f) != entry["sha256"]:
        return False, "weights checksum mismatch"
    return True, "ok"

"""Content-addressed dataset versioning (no external service).

  python training/data/dataset_version.py data/real_v1 --name real

Hashes every file referenced by manifest.jsonl plus the manifest itself and writes
data/versions/<name>-<shorthash>.json (file list + sha256 + counts). Training configs should
list the version ids they used (data.versions) so every experiment is reproducible; if the
data changes, the hash changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from saree_ai.registry import sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--name", required=True)
    a = ap.parse_args()
    root = ROOT / a.dir
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    files = {"manifest.jsonl": sha256_file(root / "manifest.jsonl")}
    for r in rows:
        for k in ("image", "mask"):
            if r.get(k):
                files[r[k]] = sha256_file(root / r[k])
    h = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
    vid = f"{a.name}-{h[:12]}"
    out = {"version_id": vid, "dir": a.dir, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "content_sha256": h,
           "samples": len(rows), "participants": len({r["participant_id"] for r in rows}),
           "sources": dict(Counter(r["source"]["type"] for r in rows)), "files": files}
    d = ROOT / "data" / "versions"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{vid}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(vid)


if __name__ == "__main__":
    main()

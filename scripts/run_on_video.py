"""Run the full local pipeline on a recorded video (or image folder) - for demos and offline tests.

  python scripts/run_on_video.py --input rec.mp4 --out runs/demo/rec [--step 3] [--every 2]

Writes results.jsonl (one structured prediction per processed frame) and annotated.mp4
(un-mirrored, with landmarks, guides and status text) so behaviour can be reviewed frame by frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from saree_ai import hardware  # noqa: E402
from saree_ai.constants import SKELETON, KP  # noqa: E402
from saree_ai.inference.pipeline import Models, Session  # noqa: E402


def draw(img, out):
    h, w = img.shape[:2]
    k = out["pose"]["keypoints"]
    if k:
        for a, b in SKELETON:
            pa, pb = k[KP[a]], k[KP[b]]
            if pa[2] > 0.3 and pb[2] > 0.3:
                cv2.line(img, (int(pa[0] * w), int(pa[1] * h)), (int(pb[0] * w), int(pb[1] * h)), (255, 230, 150), 2)
    for g in out["guidance"]["guides"]:
        pts = g.get("points") or g.get("polygon") or ([g["at"]] if g.get("at") else [])
        P = [(int(x * w), int(y * h)) for x, y in pts]
        for p, q in zip(P, P[1:] + ([P[0]] if g["type"] == "zone" else [])):
            cv2.line(img, p, q, (0, 200, 255), 2)
        if len(P) == 1:
            cv2.circle(img, P[0], 8, (0, 200, 255), 2)
    g = out["guidance"]
    txt = f"[{g['status']}] {g['step']['title']}: {g['headline'][:70]}"
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def frames(inp: Path):
    if inp.is_dir():
        for p in sorted(inp.glob("*.jpg")) + sorted(inp.glob("*.png")):
            yield cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(inp))
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=0); ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config/app.yaml").read_text())
    sel = hardware.choose_profile(hardware.inspect())
    models = Models(cfg, sel, ROOT)
    sess = Session(models, cfg, ROOT)
    sess.engine.goto(a.step)
    out_dir = ROOT / a.out; out_dir.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as fh:
        for i, f in enumerate(frames(Path(a.input))):
            if i % a.every:
                continue
            out = sess.process(f)
            fh.write(json.dumps({"frame": i, **out}) + "\n")
            vis = draw(f.copy(), out)
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
            n += 1
            if a.limit and n >= a.limit:
                break
    if writer:
        writer.release()
    print(json.dumps({"frames": n, "models": models.status()}, indent=1))


if __name__ == "__main__":
    main()

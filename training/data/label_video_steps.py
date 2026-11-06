"""Local video step-labelling tool (OpenCV window, no network).

Annotators scrub through a consented recording and mark where each draping stage starts.
Frames are then sampled into a dataset directory with step_label set, ready for region /
landmark annotation in CVAT (self-hosted) or another tool.

  python training/data/label_video_steps.py --video rec.mp4 --participant P012 --consent C-2026-012 --out data/real_v1

  # recordings made with record_session.py: stage marks and deliberate-mistake intervals are
  # preloaded from markers.json (review them, then press s); participant/consent/saree come from session.json
  python training/data/label_video_steps.py --marks E:/saree_rec/P012/<session>/d2/markers.json --out data/real_v1
  # same, without a window (accept the live marks as recorded; review_status stays 'draft')
  python training/data/label_video_steps.py --marks E:/saree_rec/P012/<session>/d2/markers.json --out data/real_v1 --no-gui

Keys:  space play/pause | a/d step -/+1 frame | j/l -/+1 s | 0-6 mark stage start at current frame
       u undo last mark | s save & sample | q quit
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from saree_ai.constants import DRAPE_STAGES, SCHEMA_VERSION  # noqa: E402

CAMERA_ANGLE = {"front": "front", "front_left": "front_left", "front_right": "front_right", "left": "left", "right": "right"}


def _nearest_frame(ts_ms: np.ndarray, t_ms: float) -> int:
    i = int(np.searchsorted(ts_ms, t_ms))
    if i <= 0:
        return 0
    if i >= len(ts_ms):
        return len(ts_ms) - 1
    return i if abs(ts_ms[i] - t_ms) < abs(ts_ms[i - 1] - t_ms) else i - 1


def load_saree_attrs(csv_path: Path | None, saree_id: str | None) -> dict:
    """Saree attributes from data_collection/trackers/sarees.csv (headers may carry '(a|b)' hints)."""
    if not csv_path or not saree_id or not Path(csv_path).exists():
        return {}
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row = {k.split("(")[0].strip(): (v or "").strip() for k, v in row.items() if k}
            if row.get("saree_id") == saree_id:
                return {k: row[k] or None for k in ("primary_color", "darkness", "fabric", "pattern", "transparency") if k in row}
    return {}


def load_marks(markers_path: Path, camera: str | None = None, sarees_csv: Path | None = None) -> dict:
    """Read one record_session.py segment. Stage marks and mistake intervals (recorded as ms since
    session start) are converted to frame indices of the chosen camera (default: the first, front)
    via that camera's per-frame timestamps."""
    markers_path = Path(markers_path)
    seg_dir = markers_path.parent
    m = json.loads(markers_path.read_text(encoding="utf-8"))
    sess = json.loads((seg_dir.parent / "session.json").read_text(encoding="utf-8"))
    cam = camera or sess["cameras"][0]["name"]
    ts = np.loadtxt(seg_dir / f"{cam}.ts.csv", delimiter=",", skiprows=1, ndmin=2)
    if len(ts) == 0:
        raise SystemExit(f"{seg_dir / (cam + '.ts.csv')} has no frames")
    t = ts[:, 1]
    stage_marks = sorted((_nearest_frame(t, x["t_ms"]), DRAPE_STAGES.index(x["stage"])) for x in m["stage_marks"])
    mistakes = []
    for x in m["mistakes"]:
        if x.get("skipped") or "start_ms" not in x:
            continue
        mistakes.append({"label": x["label"], "f0": _nearest_frame(t, x["start_ms"]), "f1": _nearest_frame(t, x["end_ms"]),
                         "uncertain_end": bool(x.get("closed_at_segment_end"))})
    seg = m["segment"]
    saree_id = sess.get("saree", {}).get(seg.get("saree")) if seg.get("kind") == "drape" else None
    return {"video": seg_dir / f"{cam}.mp4", "ts_ms": t, "camera": cam, "stage_marks": stage_marks, "mistakes": mistakes,
            "participant_id": sess["participant_id"], "consent_id": sess["consent_id"], "session_id": sess["session_id"],
            "segment_id": seg["id"], "lighting": seg.get("lighting"), "l2_variant": sess.get("l2_variant"),
            "saree_id": saree_id, "saree": load_saree_attrs(sarees_csv, saree_id)}


def save_samples(video: Path, stage_marks: list, out: Path, participant: str, consent: str, *, video_id: str,
                 sample_every_s: float = 1.0, ts_ms=None, mistakes=(), camera_angle=None, lighting=None,
                 saree=None, notes=None) -> int:
    """Sample a frame every `sample_every_s` from the first stage mark on and append to
    <out>/manifest.jsonl. Frames within 0.5 s of a stage boundary get step_label_certain=false.
    Frames inside a deliberate-mistake interval get that correction label; frames within 0.5 s of
    an interval edge are skipped (the mistake is being set up or undone). Returns rows written."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if ts_ms is not None:
        n = min(n, len(ts_ms)) if n > 0 else len(ts_ms)
    half_s = fps * 0.5
    out = Path(out); (out / "images").mkdir(parents=True, exist_ok=True)
    step = max(1, int(round(fps * sample_every_s)))
    written = 0
    with open(out / "manifest.jsonl", "a", encoding="utf-8") as fh:
        for f in range(0, n, step):
            st = max([s for fm, s in stage_marks if fm <= f], default=None)
            if st is None:
                continue
            if any(abs(f - mk[e]) < half_s for mk in mistakes for e in ("f0", "f1")):
                continue
            labels = sorted({mk["label"] for mk in mistakes if mk["f0"] <= f <= mk["f1"]})
            cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
            if not ok:
                continue
            sid = f"{video_id}-f{f:06d}"
            cv2.imwrite(str(out / "images" / f"{sid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            near = any(abs(f - fm) < half_s for fm, _ in stage_marks)
            row = {"id": sid, "schema_version": SCHEMA_VERSION, "image": f"images/{sid}.jpg",
                   "width": img.shape[1], "height": img.shape[0], "video_id": video_id, "frame_index": f,
                   "timestamp_ms": float(ts_ms[f]) if ts_ms is not None else f / fps * 1000, "participant_id": participant,
                   "source": {"type": "real", "consent_id": consent, "license": "project-internal, participant consent"},
                   "step_label": DRAPE_STAGES[st], "step_label_certain": not near,
                   "correction_labels": labels, "review_status": "draft"}
            cap_info = {k: v for k, v in (("camera_angle", camera_angle), ("lighting", lighting)) if v}
            if cap_info:
                row["capture"] = cap_info
            if saree:
                row["saree"] = {**saree, "drape_style": "nivi"}
            if notes:
                row["notes"] = notes
            fh.write(json.dumps(row) + "\n")
            written += 1
    cap.release()
    return written


def lighting_value(info: dict) -> str | None:
    if info.get("lighting") == "L1":
        return "normal"
    return {"dim": "dim", "backlit": "backlit", "warm": "colored"}.get(info.get("l2_variant"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--participant"); ap.add_argument("--consent")
    ap.add_argument("--marks", help="markers.json written by record_session.py")
    ap.add_argument("--camera", help="camera name inside the segment (default: first / front)")
    ap.add_argument("--sarees", default=str(ROOT / "data_collection" / "trackers" / "sarees.csv"))
    ap.add_argument("--no-gui", action="store_true", help="with --marks: save directly from the live marks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-every-s", type=float, default=1.0)
    a = ap.parse_args()
    info = load_marks(Path(a.marks), a.camera, Path(a.sarees)) if a.marks else None
    video = str(info["video"]) if info else a.video
    participant = info["participant_id"] if info else a.participant
    consent = info["consent_id"] if info else a.consent
    if not (video and participant and consent):
        raise SystemExit("give --marks, or all of --video --participant --consent")
    out = ROOT / a.out
    if info:
        vid = f"{participant}-{info['session_id']}-{info['segment_id']}-{info['camera']}"
        extra = dict(ts_ms=info["ts_ms"], mistakes=info["mistakes"], camera_angle=CAMERA_ANGLE.get(info["camera"], "other"),
                     lighting=lighting_value(info), saree=info["saree"] or None,
                     notes=f"session={info['session_id']} segment={info['segment_id']} saree={info['saree_id']}")
    else:
        vid, extra = f"{participant}-{Path(video).stem}", {}
    marks: list[tuple[int, int]] = list(info["stage_marks"]) if info else []
    if a.no_gui:
        if not info:
            raise SystemExit("--no-gui needs --marks")
        n = save_samples(Path(video), marks, out, participant, consent, video_id=vid, sample_every_s=a.sample_every_s, **extra)
        print(f"saved {n} samples to {out}")
        return
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    mist = extra.get("mistakes", [])
    i, playing = 0, False
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok: i = max(0, n - 1); playing = False; continue
        cur = max([s for f, s in marks if f <= i], default=None)
        txt = f"{i}/{n}  stage: {DRAPE_STAGES[cur] if cur is not None else '-'}   [0-6 mark, s save, q quit]"
        cv2.putText(fr, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        active = [mk["label"] for mk in mist if mk["f0"] <= i <= mk["f1"]]
        if active:
            cv2.putText(fr, "MISTAKE: " + ", ".join(active), (10, fr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for k, s in enumerate(DRAPE_STAGES):
            cv2.putText(fr, f"{k}: {s}", (10, 60 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("label steps", fr)
        key = cv2.waitKey(int(1000 / fps) if playing else 0) & 0xFF
        if playing: i = min(n - 1, i + 1)
        if key == ord(" "): playing = not playing
        elif key == ord("a"): i = max(0, i - 1)
        elif key == ord("d"): i = min(n - 1, i + 1)
        elif key == ord("j"): i = max(0, i - int(fps))
        elif key == ord("l"): i = min(n - 1, i + int(fps))
        elif ord("0") <= key <= ord("6"): marks = sorted([m for m in marks if m[0] != i] + [(i, key - ord("0"))])
        elif key == ord("u") and marks: marks.pop()
        elif key == ord("q"): break
        elif key == ord("s"):
            cap.release(); cv2.destroyAllWindows()
            k = save_samples(Path(video), marks, out, participant, consent, video_id=vid, sample_every_s=a.sample_every_s, **extra)
            print(f"saved {k} samples to {out}")
            return
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

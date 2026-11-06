"""Local multi-camera session recorder (OpenCV; no network, nothing uploaded).

Drives the protocol in data_collection/protocol.yaml: shows the operator what comes next
(drape, saree, lighting, scripted mistake), records every camera with per-frame timestamps, and
captures live marks:

  space  start / stop the current segment (pose clip, d1..d4)
  0-6    stage starts now (index into saree_ai.constants.DRAPE_STAGES)
  m      toggle: scripted mistake being held (press when it is held, again when corrected)
  n      next prompt (pose clip) / skip a scripted mistake that was not performed
  p      pause / resume writing (e.g. participant asks for a break)
  q      abort session (already-saved segments are kept)

  python training/data/record_session.py --list-cameras
  python training/data/record_session.py --preview --cams 0 1
  python training/data/record_session.py --out E:/saree_rec --cams 0 1 \
      --participant P012 --consent C-2026-012 --saree-a S07 --saree-b S15 --l2 backlit

Output per segment:  <out>/<participant>/<session>/<segment>/<camera>.mp4, <camera>.ts.csv
(frame index, capture time in ms since session start) and markers.json. The save report checks
inter-camera timestamp offset and dropped frames (go/no-go criteria in DATA_COLLECTION_PLAN.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from saree_ai.constants import DRAPE_STAGES, CORRECTION_LABELS  # noqa: E402

BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def load_protocol(path=ROOT / "data_collection" / "protocol.yaml") -> dict:
    p = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for name, m in p["mistakes"].items():
        assert name in CORRECTION_LABELS, f"protocol mistake '{name}' is not a CORRECTION_LABEL"
        assert m["during_stage"] in DRAPE_STAGES, f"bad stage for {name}"
    return p


def participant_number(pid: str) -> int:
    digits = "".join(ch for ch in pid if ch.isdigit())
    if not digits:
        raise SystemExit(f"participant id {pid!r} must contain a number (e.g. P012)")
    return int(digits)


def session_plan(protocol: dict, pid: str, l2_variant: str) -> list[dict]:
    """Ordered segments for this participant, with mistake scripts resolved by rotation."""
    rot = protocol["mistake_rotation"]
    n = participant_number(pid)
    slots = {"mistakes_slot1": rot[n % len(rot)], "mistakes_slot2": rot[(n + 3) % len(rot)]}
    order = {s: i for i, s in enumerate(DRAPE_STAGES)}
    segs = [{"id": "pose", "kind": "pose", "prompts": protocol["pose_clip"]["prompts"], "seconds": protocol["pose_clip"]["seconds"]}]
    for d in protocol["drapes"]:
        mistakes = sorted(slots.get(d["script"], []), key=lambda m: order[protocol["mistakes"][m]["during_stage"]])
        segs.append({"id": d["id"], "kind": "drape", "saree": d["saree"], "lighting": d["lighting"],
                     "lighting_text": protocol["lighting_setups"][d["lighting"]] + (f"  [today's L2: {l2_variant}]" if d["lighting"] == "L2" else ""),
                     "script": d["script"],
                     "mistakes": [{"label": m, **protocol["mistakes"][m]} for m in mistakes]})
    return segs


class FileSource:
    """Stands in for a camera in tests/demos: replays a video file at its native fps."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.period = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.next_t = time.perf_counter()

    def read(self):
        now = time.perf_counter()
        if now < self.next_t:
            time.sleep(self.next_t - now)
        self.next_t += self.period
        ok, f = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, f = self.cap.read()
        return ok, f

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self.cap.release()


def open_camera(spec, resolution=None, fps=30):
    if isinstance(spec, str) and not spec.isdigit():
        return FileSource(spec)
    cap = cv2.VideoCapture(int(spec), BACKEND)
    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {spec}")
    return cap


class CameraThread(threading.Thread):
    """Reads continuously (so the preview is live); writes only while `recording`."""

    def __init__(self, name, cap, t0):
        super().__init__(daemon=True)
        self.name_, self.cap, self.t0 = name, cap, t0
        self.latest = None
        self.lock = threading.Lock()
        self.writer = None
        self.ts_fh = None
        self.frames = 0
        self.recording = False
        self.stop_flag = False

    def start_segment(self, seg_dir: Path, fps: float):
        with self.lock:
            f = self.latest
        if f is None:
            raise RuntimeError(f"camera {self.name_} has produced no frames yet")
        h, w = f.shape[:2]
        self.writer = cv2.VideoWriter(str(seg_dir / f"{self.name_}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self.ts_fh = open(seg_dir / f"{self.name_}.ts.csv", "w", encoding="utf-8")
        self.ts_fh.write("frame,t_ms\n")
        self.frames = 0
        self.recording = True

    def end_segment(self):
        self.recording = False
        with self.lock:
            if self.writer: self.writer.release(); self.writer = None
            if self.ts_fh: self.ts_fh.close(); self.ts_fh = None
        return self.frames

    def run(self):
        while not self.stop_flag:
            ok, f = self.cap.read()
            t = (time.perf_counter() - self.t0) * 1000
            if not ok:
                time.sleep(0.005); continue
            with self.lock:
                self.latest = f
                if self.recording and self.writer is not None:
                    self.writer.write(f)
                    self.ts_fh.write(f"{self.frames},{t:.2f}\n")
                    self.frames += 1


def sync_report(seg_dir: Path, cams: list[str], pauses=()) -> dict:
    """Per-camera measured fps and dropped-frame share (gaps > 1.5x the median frame interval,
    excluding gaps caused by an operator pause), plus nearest-frame offset between the first two
    cameras. Go/no-go thresholds: dropped < 2% per camera, nearest-frame offset p95 < 50 ms."""
    ts = {}
    for c in cams:
        a = np.loadtxt(seg_dir / f"{c}.ts.csv", delimiter=",", skiprows=1, ndmin=2)
        ts[c] = a[:, 1] if len(a) else np.zeros(0)
    rep = {"cameras": {}}
    for c, t in ts.items():
        if len(t) < 2:
            rep["cameras"][c] = {"frames": int(len(t)), "ok": False}; continue
        dt = np.diff(t)
        med = float(np.median(dt))
        gap = dt > 1.5 * med
        for p in pauses:   # a pause legitimately leaves a gap in the timestamps
            gap &= ~((t[:-1] <= p["end_ms"]) & (t[1:] >= p["start_ms"]))
        dropped = float(gap.sum() / len(dt))
        rep["cameras"][c] = {"frames": int(len(t)), "fps_measured": round(1000 / med, 2), "dropped_frac": round(dropped, 4),
                             "ok": dropped < 0.02}
    if len(cams) >= 2 and all(len(ts[c]) > 1 for c in cams):
        p, s = ts[cams[0]], ts[cams[1]]
        idx = np.clip(np.searchsorted(s, p), 1, len(s) - 1)
        nn = np.minimum(np.abs(s[idx] - p), np.abs(s[idx - 1] - p))
        rep["nearest_frame_offset_ms"] = {"median": round(float(np.median(nn)), 2), "p95": round(float(np.percentile(nn, 95)), 2)}
        rep["start_offset_ms"] = round(float(abs(p[0] - s[0])), 2)
        rep["sync_ok"] = rep["nearest_frame_offset_ms"]["p95"] < 50
    rep["ok"] = all(v["ok"] for v in rep["cameras"].values()) and rep.get("sync_ok", True)
    return rep


class Recorder:
    """All session state and file output. The OpenCV UI below only maps keys to these methods,
    so the logic is testable without a window or real cameras."""

    def __init__(self, out: Path, meta: dict, plan: list[dict], cams: dict[str, CameraThread], t0: float, fps: float = 30):
        self.out, self.meta, self.plan, self.cams, self.t0, self.fps = out, meta, plan, cams, t0, fps
        self.seg_i = 0
        self.active = False
        self.paused = False
        self.reset_marks()
        out.mkdir(parents=True, exist_ok=True)
        (out / "session.json").write_text(json.dumps({**meta, "plan": plan}, indent=2), encoding="utf-8")

    def now_ms(self):
        return (time.perf_counter() - self.t0) * 1000

    def reset_marks(self):
        self.stage_marks, self.mistakes, self.pauses, self.prompt_marks = [], [], [], []
        self.open_mistake = None
        self.mistake_i = 0
        self.prompt_i = 0
        self.current_stage = None

    @property
    def segment(self):
        return self.plan[self.seg_i] if self.seg_i < len(self.plan) else None

    def start(self):
        seg = self.segment
        if seg is None or self.active:
            return
        self.seg_dir = self.out / seg["id"]
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        self.reset_marks()
        self.seg_start = self.now_ms()
        for c in self.cams.values():
            c.start_segment(self.seg_dir, self.fps)
        self.active = True
        if seg["kind"] == "pose":
            self.prompt_marks.append({"t_ms": self.seg_start, "prompt": seg["prompts"][0]})

    def mark_stage(self, k: int):
        if self.active and self.segment["kind"] == "drape" and 0 <= k < len(DRAPE_STAGES):
            self.stage_marks.append({"t_ms": self.now_ms(), "stage": DRAPE_STAGES[k]})
            self.current_stage = DRAPE_STAGES[k]

    def next_mistake(self):
        seg = self.segment
        if not seg or seg["kind"] != "drape" or self.mistake_i >= len(seg["mistakes"]):
            return None
        return seg["mistakes"][self.mistake_i]

    def toggle_mistake(self):
        m = self.next_mistake()
        if not self.active or m is None:
            return
        if self.open_mistake is None:
            self.open_mistake = {"label": m["label"], "start_ms": self.now_ms(), "stage_at_start": self.current_stage}
        else:
            self.open_mistake["end_ms"] = self.now_ms()
            self.mistakes.append(self.open_mistake)
            self.open_mistake = None
            self.mistake_i += 1

    def skip(self):
        seg = self.segment
        if not self.active or seg is None:
            return
        if seg["kind"] == "pose":
            self.prompt_i = min(self.prompt_i + 1, len(seg["prompts"]) - 1)
            self.prompt_marks.append({"t_ms": self.now_ms(), "prompt": seg["prompts"][self.prompt_i]})
        elif self.next_mistake() is not None and self.open_mistake is None:
            self.mistakes.append({"label": self.next_mistake()["label"], "skipped": True, "t_ms": self.now_ms()})
            self.mistake_i += 1

    def toggle_pause(self):
        if not self.active:
            return
        self.paused = not self.paused
        for c in self.cams.values():
            c.recording = not self.paused
        if self.paused: self.pauses.append({"start_ms": self.now_ms()})
        else: self.pauses[-1]["end_ms"] = self.now_ms()

    def stop(self) -> dict:
        if not self.active:
            return {}
        if self.open_mistake is not None:   # operator forgot to close it: keep, but flag
            self.open_mistake.update(end_ms=self.now_ms(), closed_at_segment_end=True)
            self.mistakes.append(self.open_mistake); self.open_mistake = None
        if self.paused:
            self.toggle_pause()
        frames = {n: c.end_segment() for n, c in self.cams.items()}
        self.active = False
        seg = self.segment
        markers = {"segment": seg, "start_ms": self.seg_start, "end_ms": self.now_ms(), "frames": frames,
                   "stage_marks": self.stage_marks, "mistakes": self.mistakes, "pauses": self.pauses,
                   "prompts": self.prompt_marks, "time_base": "ms since session start; per-frame times in <camera>.ts.csv"}
        rep = sync_report(self.seg_dir, list(self.cams), self.pauses)
        markers["sync_report"] = rep
        missing = [m["label"] for m in seg.get("mistakes", []) if m["label"] not in {x["label"] for x in self.mistakes if not x.get("skipped")}]
        markers["mistakes_not_recorded"] = missing
        (self.seg_dir / "markers.json").write_text(json.dumps(markers, indent=2), encoding="utf-8")
        self.seg_i += 1
        return {"segment": seg["id"], "frames": frames, "sync": rep, "mistakes_not_recorded": missing}


# ----------------------------------------------------------------------------- operator UI
def overlay(frame, rec: Recorder):
    img = frame.copy()
    h, w = img.shape[:2]
    for frac, col in ((0.04, (0, 255, 255)), (0.97, (0, 255, 255))):   # head / feet guide lines
        cv2.line(img, (0, int(h * frac)), (w, int(h * frac)), col, 1)
    seg = rec.segment
    lines = []
    if seg is None:
        lines = ["SESSION COMPLETE - press q"]
    else:
        state = "REC" if rec.active and not rec.paused else ("PAUSED" if rec.paused else "ready - SPACE to start")
        lines.append(f"[{state}] segment {seg['id']} ({rec.seg_i + 1}/{len(rec.plan)})")
        if seg["kind"] == "pose":
            lines.append("PROMPT: " + seg["prompts"][rec.prompt_i] + "   (n = next prompt)")
        else:
            lines.append(f"saree {seg['saree']} | lighting {seg['lighting']}: {seg['lighting_text'][:60]}")
            lines.append(f"stage: {rec.current_stage or '-'}   (0-6 to mark)")
            m = rec.next_mistake()
            if seg["script"] != "correct" and m:
                tag = "HOLDING - m when fixed" if rec.open_mistake else f"at stage '{m['during_stage']}': m when held"
                lines.append(f"MISTAKE {rec.mistake_i + 1}/{len(seg['mistakes'])}: {m['label']} [{tag}]")
                lines.append("  " + m["prompt"][:90])
            elif seg["script"] == "correct":
                lines.append("CORRECT drape - no mistakes")
    for i, t in enumerate(lines):
        y = 24 + 22 * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if rec.active and not rec.paused:
        cv2.circle(img, (w - 20, 20), 8, (0, 0, 255), -1)
    return img


def list_cameras(n=6):
    for i in range(n):
        cap = cv2.VideoCapture(i, BACKEND)
        if cap.isOpened():
            ok, f = cap.read()
            print(f"camera {i}: {'%dx%d' % (f.shape[1], f.shape[0]) if ok else 'opened, no frame'}")
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--cams", nargs="+", default=["0"], help="camera indices (or video files for testing), in protocol order")
    ap.add_argument("--out"); ap.add_argument("--participant"); ap.add_argument("--consent")
    ap.add_argument("--saree-a"); ap.add_argument("--saree-b"); ap.add_argument("--l2", choices=["dim", "backlit", "warm"])
    ap.add_argument("--operator", default=None)
    a = ap.parse_args()
    if a.list_cameras:
        list_cameras(); return
    protocol = load_protocol()
    specs = protocol["cameras"][: len(a.cams)]
    t0 = time.perf_counter()
    threads = {}
    for spec, src in zip(specs, a.cams):
        th = CameraThread(spec["name"], open_camera(src, spec.get("resolution"), spec.get("fps", 30)), t0)
        th.start(); threads[spec["name"]] = th
    if a.preview:
        print("preview: head should be below the top line and feet above the bottom line. q to quit.")
        while True:
            frames = [t.latest for t in threads.values() if t.latest is not None]
            if frames:
                tiles = [cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1]))) for f in frames]
                for f in tiles:
                    for frac in (0.04, 0.97): cv2.line(f, (0, int(f.shape[0] * frac)), (f.shape[1], int(f.shape[0] * frac)), (0, 255, 255), 1)
                hmax = max(t.shape[0] for t in tiles)
                cv2.imshow("preview", np.hstack([cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]))
            if cv2.waitKey(30) & 0xFF == ord("q"): break
        return
    for req in ("out", "participant", "consent", "saree_a", "saree_b", "l2"):
        if not getattr(a, req):
            raise SystemExit(f"--{req.replace('_', '-')} is required for recording")
    session_id = time.strftime("%Y%m%d-%H%M%S")
    meta = {"participant_id": a.participant, "consent_id": a.consent, "session_id": session_id, "operator": a.operator,
            "saree": {"A": a.saree_a, "B": a.saree_b}, "l2_variant": a.l2, "protocol_version": protocol["protocol_version"],
            "cameras": [{**s, "source": str(src)} for s, src in zip(specs, a.cams)]}
    rec = Recorder(Path(a.out) / a.participant / session_id, meta, session_plan(protocol, a.participant, a.l2), threads, t0,
                   fps=float(specs[0].get("fps", 30)))
    main_cam = next(iter(threads.values()))
    while True:
        if main_cam.latest is not None:
            cv2.imshow("recorder (operator view)", overlay(main_cam.latest, rec))
        k = cv2.waitKey(15) & 0xFF
        if k == 255: continue
        if k == ord("q"):
            if rec.active: print(rec.stop())
            break
        if k == ord(" "):
            if rec.active: print(json.dumps(rec.stop(), indent=1))
            else: rec.start()
        elif ord("0") <= k <= ord("6"): rec.mark_stage(k - ord("0"))
        elif k == ord("m"): rec.toggle_mistake()
        elif k == ord("n"): rec.skip()
        elif k == ord("p"): rec.toggle_pause()
    for t in threads.values():
        t.stop_flag = True
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

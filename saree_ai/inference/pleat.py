"""Model C runtime: pleat recognition.

Two complementary components:
  1. PleatSignalAnalyzer - interpretable signal processing (no training). Measures fold count,
     spacing regularity, fold-line alignment and left/right symmetry from horizontal-gradient
     column profiles. ALWAYS labelled heuristic/experimental. It reports its own confidence
     (periodicity strength) and refuses (confidence ~0) on busy printed fabric where the
     gradient profile is dominated by the print instead of the folds.
  2. PleatModel - our trained PleatNet (ONNX) on the pleat crop: count / quality / secured.

The two are fused only when both are confident and agree; otherwise the more conservative
(lower-confidence) reading wins so the correction engine stays quiet.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..constants import PLEAT_QUALITY
from ..registry import ROOT, active, loadable


@dataclass
class PleatResult:
    available: bool
    source: str = "none"                  # signal | model | fused
    roi: list | None = None
    count: float | None = None
    count_conf: float = 0.0
    spacing_cv: float | None = None       # std/mean of fold spacing; 0 = perfectly even
    alignment_deg: float | None = None    # mean |deviation| of fold lines from their common direction
    tilt_deg: float | None = None         # common fold direction vs vertical
    symmetry: float | None = None         # 0..1 correlation of mirrored halves
    quality_probs: list | None = None
    secured_prob: float | None = None
    confidence: float = 0.0
    experimental: bool = True
    fold_x: list = field(default_factory=list)   # fold x positions (px, frame coords) at ROI mid-height
    reason: str = ""
    latency_ms: float = 0.0

    def as_feature_input(self):
        return {"count": self.count, "quality_probs": self.quality_probs, "secured_prob": self.secured_prob}


def _peaks(profile: np.ndarray, min_dist: int, prominence: float) -> np.ndarray:
    """Tiny dependency-free peak finder (local maxima with prominence & spacing)."""
    p = profile
    cand = np.nonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]))[0] + 1
    keep = []
    for i in cand[np.argsort(-p[cand])]:
        if any(abs(i - j) < min_dist for j in keep):
            continue
        lo = max(0, i - 2 * min_dist); hi = min(len(p), i + 2 * min_dist + 1)
        base = max(p[lo:i].min() if i > lo else p[i], p[i + 1:hi].min() if hi > i + 1 else p[i])
        if p[i] - base >= prominence:
            keep.append(i)
    return np.array(sorted(keep), dtype=int)


class PleatSignalAnalyzer:
    def __init__(self, min_fold_px=4):
        self.min_fold_px = min_fold_px

    def column_profile(self, gray: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        # Keep only predominantly vertical edges (fold edges) - suppress horizontal print lines.
        vert = np.abs(gx) * (np.abs(gx) > 1.5 * np.abs(gy))
        prof = vert.mean(0)
        k = max(3, (gray.shape[1] // 60) | 1)
        return cv2.GaussianBlur(prof[None], (k, 1), 0)[0]

    def __call__(self, bgr: np.ndarray, roi: list) -> PleatResult:
        t0 = time.perf_counter()
        x0, y0, x1, y1 = [int(v) for v in roi]
        crop = bgr[max(0, y0):y1, max(0, x0):x1]
        if crop.shape[0] < 24 or crop.shape[1] < 24:
            return PleatResult(False, "signal", roi, reason="pleat region too small")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (4, 4)).apply(gray)    # helps dark / low-contrast fabric
        w = gray.shape[1]
        bands = np.array_split(np.arange(gray.shape[0]), 3)
        profs = [self.column_profile(gray[b]) for b in bands]
        full = np.mean(profs, 0)
        full = full - cv2.GaussianBlur(full[None], (max(3, (w // 4) | 1), 1), 0)[0]  # detrend
        full = np.clip(full, 0, None)
        prom = 0.25 * (full.max() + 1e-6)
        min_dist = max(self.min_fold_px, w // 20)
        pk = _peaks(full, min_dist, prom)

        # Periodicity strength via normalised autocorrelation: folds give a clear secondary peak,
        # random print texture does not.
        z = full - full.mean()
        ac = np.correlate(z, z, "full")[len(z) - 1:]
        ac = ac / (ac[0] + 1e-6)
        sec = ac[min_dist: max(min_dist + 1, w // 2)]
        periodicity = float(np.clip(sec.max(), 0, 1)) if len(sec) else 0.0

        res = PleatResult(True, "signal", roi, experimental=True)
        if len(pk) >= 2:
            # Peaks are fold EDGES (incl. the two outer edges of the stack); a fold is the band
            # between two adjacent edges, matching the fold_lines annotation in docs/DATASET.md.
            sp = np.diff(pk).astype(float)
            res.count = float(len(pk) - 1)
            res.spacing_cv = float(sp.std() / (sp.mean() + 1e-6))
            res.fold_x = [float(x0 + p) for p in pk]
            # Alignment: track each fold's x position in top vs bottom band.
            pos = []
            for prof_b in (profs[0], profs[-1]):
                pb = _peaks(np.clip(prof_b - prof_b.mean(), 0, None), min_dist, 0.2 * (prof_b.max() + 1e-6))
                pos.append(pb)
            dy = (bands[-1].mean() - bands[0].mean()) or 1.0
            shifts = []
            for p in pk:
                if len(pos[0]) and len(pos[1]):
                    a = pos[0][np.argmin(np.abs(pos[0] - p))]
                    b = pos[1][np.argmin(np.abs(pos[1] - p))]
                    if abs(a - p) < min_dist and abs(b - p) < min_dist:
                        shifts.append(np.degrees(np.arctan2(b - a, dy)))
            if len(shifts) >= 2:
                shifts = np.array(shifts)
                res.tilt_deg = float(np.median(shifts))
                res.alignment_deg = float(np.mean(np.abs(shifts - np.median(shifts))))
            half = full[: w // 2]; other = full[w - len(half):][::-1]
            if half.std() > 0 and other.std() > 0:
                res.symmetry = float(np.clip(np.corrcoef(half, other)[0, 1], 0, 1))
        # Confidence: periodic signal AND a plausible fold count; uncertain otherwise.
        plausible = res.count is not None and 2 <= res.count <= 14
        res.confidence = float(periodicity * (1.0 if plausible else 0.3))
        res.count_conf = res.confidence
        res.reason = "ok" if plausible else "no clear periodic fold pattern"
        res.latency_ms = (time.perf_counter() - t0) * 1000
        return res


class PleatModel:
    def __init__(self, cfg: dict, providers=None):
        self.entry = active("pleat")
        ok, why = loadable(self.entry, cfg.get("min_status", "experimental"))
        self.available, self.reason = ok, why
        if ok:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(ROOT / self.entry["file"]), providers=providers or ["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
            self.size = int(self.entry.get("input_size", 128))
            self.experimental = self.entry.get("status") != "validated"

    def __call__(self, bgr, roi) -> PleatResult:
        if not self.available:
            return PleatResult(False, "model", roi, reason=self.reason)
        t0 = time.perf_counter()
        x0, y0, x1, y1 = [int(v) for v in roi]
        crop = bgr[max(0, y0):y1, max(0, x0):x1]
        if crop.size == 0:
            return PleatResult(False, "model", roi, reason="empty crop")
        x = cv2.resize(crop, (self.size, self.size))[:, :, ::-1].astype(np.float32) / 255
        x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
        count, q, sec, vis = self.sess.run(None, {self.inp: x})
        qe = np.exp(q[0] - q[0].max()); qp = qe / qe.sum()
        sig = lambda v: float(1 / (1 + np.exp(-float(v))))  # noqa: E731
        visible = sig(vis[0, 0])
        return PleatResult(True, "model", roi, count=float(max(0.0, count[0, 0])), count_conf=visible,
                           quality_probs=qp.tolist(), secured_prob=sig(sec[0, 0]),
                           confidence=float(visible * qp.max()), experimental=self.experimental,
                           reason="ok", latency_ms=(time.perf_counter() - t0) * 1000)


def fuse(signal: PleatResult, model: PleatResult) -> PleatResult:
    if not model.available:
        return signal
    if not signal.available or signal.confidence < 0.3:
        return model
    out = PleatResult(True, "fused", model.roi, experimental=model.experimental or signal.experimental)
    out.quality_probs, out.secured_prob = model.quality_probs, model.secured_prob
    out.spacing_cv, out.alignment_deg, out.tilt_deg = signal.spacing_cv, signal.alignment_deg, signal.tilt_deg
    out.symmetry, out.fold_x = signal.symmetry, signal.fold_x
    agree = signal.count is not None and model.count is not None and abs(signal.count - model.count) <= 1.5
    out.count = model.count if model.count_conf >= signal.count_conf else signal.count
    out.count_conf = max(model.count_conf, signal.count_conf) if agree else 0.5 * min(model.count_conf, signal.count_conf)
    out.confidence = min(model.confidence, signal.confidence) if not agree else max(model.confidence, signal.confidence)
    out.reason = "agree" if agree else "signal and model disagree on count"
    out.latency_ms = signal.latency_ms + model.latency_ms
    return out


def quality_label(res: PleatResult) -> tuple[str | None, float]:
    if res.quality_probs is None:
        return None, 0.0
    i = int(np.argmax(res.quality_probs))
    return PLEAT_QUALITY[i], float(res.quality_probs[i])

"""Model B runtime: saree region segmentation through ONNX Runtime."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..constants import SEG, SEG_CLASSES, SAREE_CLASS_IDS
from ..registry import ROOT, active, loadable

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class SegResult:
    available: bool
    mask: np.ndarray | None = None              # (H,W) uint8 class ids at frame resolution
    confidence: float = 0.0                     # mean max-softmax over predicted saree pixels
    class_confidence: dict = field(default_factory=dict)
    class_fraction: dict = field(default_factory=dict)
    ambiguous_fraction: float = 1.0             # share of saree pixels with max-prob < 0.5
    experimental: bool = True
    reason: str = ""
    latency_ms: float = 0.0
    model_version: str | None = None


def preprocess(bgr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    x = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


class Segmenter:
    def __init__(self, cfg: dict, providers=None):
        self.entry = active("segmentation")
        ok, why = loadable(self.entry, cfg.get("min_status", "experimental"))
        self.available, self.reason = ok, why
        if ok:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(ROOT / self.entry["file"]), providers=providers or ["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
            self.hw = tuple(self.entry.get("input_hw", [384, 256]))
            self.experimental = self.entry.get("status") != "validated"

    def __call__(self, bgr: np.ndarray) -> SegResult:
        if not self.available:
            return SegResult(False, reason=self.reason)
        t0 = time.perf_counter()
        logits = self.sess.run(None, {self.inp: preprocess(bgr, self.hw)})[0][0]  # (C,h,w)
        e = np.exp(logits - logits.max(0, keepdims=True))
        prob = e / e.sum(0, keepdims=True)
        H, W = bgr.shape[:2]
        lab_small = prob.argmax(0).astype(np.uint8)
        maxp_small = prob.max(0)
        mask = cv2.resize(lab_small, (W, H), interpolation=cv2.INTER_NEAREST)
        saree = np.isin(lab_small, SAREE_CLASS_IDS)
        conf = float(maxp_small[saree].mean()) if saree.any() else 0.0
        amb = float((maxp_small[saree] < 0.5).mean()) if saree.any() else 1.0
        cls_conf, cls_frac = {}, {}
        n = lab_small.size
        for i, c in enumerate(SEG_CLASSES):
            m = lab_small == i
            cls_frac[c] = float(m.sum() / n)
            cls_conf[c] = float(maxp_small[m].mean()) if m.any() else 0.0
        return SegResult(True, mask, conf, cls_conf, cls_frac, amb, self.experimental, "ok",
                         (time.perf_counter() - t0) * 1000, self.entry.get("version"))


def region_bbox(mask: np.ndarray, cls: str, min_px: int = 50):
    ys, xs = np.nonzero(mask == SEG[cls])
    if len(xs) < min_px:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def region_contours(mask: np.ndarray, cls: str, max_points: int = 40, min_area_frac: float = 0.002):
    """Simplified polygons (pixel coords) for overlay rendering."""
    m = (mask == SEG[cls]).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out, area_min = [], min_area_frac * mask.size
    for c in cnts:
        if cv2.contourArea(c) < area_min:
            continue
        eps = 0.01 * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True)[:, 0, :]
        if len(c) > max_points:
            c = c[np.linspace(0, len(c) - 1, max_points).astype(int)]
        out.append(c.tolist())
    return out

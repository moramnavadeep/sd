"""Per-session pipeline: Camera frame -> Pose -> Segmentation -> Pleats -> Stage -> Correction.

Everything executes in-process on this machine. Nothing here performs network I/O.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..correction.checks import FrameState
from ..correction.engine import CorrectionEngine, Tutorial
from ..features import frame_features, body_geometry
from ..constants import KP, KEYPOINTS
from .pose import build_pose, PoseResult
from .segmentation import Segmenter, region_bbox, region_contours
from .pleat import PleatSignalAnalyzer, PleatModel, PleatResult, fuse
from .stage import StageRecognizer


class Models:
    """Loaded once per server; shared by sessions (stateless per frame)."""

    def __init__(self, cfg: dict, hw: dict, root: Path):
        min_status = cfg["min_model_status"]
        self.pose, self.pose_msg = build_pose({**cfg["pose"], "min_status": min_status}, hw)
        self.seg = Segmenter({"min_status": min_status})
        self.pleat_model = PleatModel({"min_status": min_status})
        self.pleat_signal = PleatSignalAnalyzer()
        self.stage_cfg = {**cfg.get("stage", {}), "min_status": min_status}
        self.root = root

    def status(self) -> dict:
        def st(name, ok, why, exp=None, entry=None):
            return {"name": name, "available": ok, "detail": why, "experimental": exp,
                    "version": (entry or {}).get("version"), "registry_status": (entry or {}).get("status")}
        return {
            "pose": st(getattr(self.pose, "name", "none"), self.pose.name != "none", self.pose_msg, self.pose.experimental),
            "segmentation": st("saree_segmentation", self.seg.available, self.seg.reason, getattr(self.seg, "experimental", None), self.seg.entry),
            "pleat_model": st("pleat_net", self.pleat_model.available, self.pleat_model.reason, getattr(self.pleat_model, "experimental", None), self.pleat_model.entry),
            "pleat_signal": st("pleat_signal_analyzer", True, "heuristic signal analysis (always experimental)", True),
        }


class Session:
    def __init__(self, models: Models, cfg: dict, root: Path):
        self.m = models
        self.cfg = cfg
        self.th = cfg["thresholds"]
        self.stage = StageRecognizer(models.stage_cfg)
        self.engine = CorrectionEngine(Tutorial(root / cfg["tutorial"]), self.th, cfg["corrections"], cfg["speech"])
        self.frames = 0

    def pleat_roi(self, seg_res, pose_res: PoseResult, w, h):
        if seg_res.available and seg_res.mask is not None:
            b = region_bbox(seg_res.mask, "pleats", min_px=int(0.002 * w * h))
            if b:
                return b, "segmentation"
        k = pose_res.keypoints
        if k is not None:
            g = body_geometry(k, self.th["keypoint"])
            if g.valid:
                lr = k[KP["left_hip"], :2] - k[KP["right_hip"], :2]
                lr = lr / (np.linalg.norm(lr) + 1e-6)
                c = g.hip_center + lr * 0.1 * g.hip_width
                knee_y = np.mean([k[KP["left_knee"], 1], k[KP["right_knee"], 1]])
                x0, x1 = c[0] - 0.35 * g.hip_width, c[0] + 0.35 * g.hip_width
                y0 = g.hip_center[1] + 0.05 * g.torso_len
                y1 = knee_y if knee_y > y0 + 10 else y0 + g.torso_len
                return [int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))], "pose_zone"
        return None, None

    def process(self, bgr: np.ndarray, visible: list | None = None) -> dict:
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        pose = self.m.pose(bgr)
        seg = self.m.seg(bgr)
        roi, roi_src = self.pleat_roi(seg, pose, w, h)
        if roi is not None:
            pl = fuse(self.m.pleat_signal(bgr, roi), self.m.pleat_model(bgr, roi))
        else:
            pl = PleatResult(False, reason="no pleat region located")
        feats, extras = (frame_features(seg.mask, pose.keypoints, pl.as_feature_input()) if seg.available
                         else (None, {}))
        st = self.stage(feats, seg.available)
        if visible and len(visible) == 4:   # part of the camera image actually shown on screen (object-fit crop)
            extras["visible_px"] = [visible[0] * w, visible[1] * h, visible[2] * w, visible[3] * h]
        state = FrameState(w, h, pose, seg, pl, st, feats, extras)
        guidance = self.engine.evaluate(state)
        self.frames += 1

        regions = {}
        if seg.available and seg.mask is not None:
            for c in ("waist_wrap", "pleats", "pallu"):
                polys = region_contours(seg.mask, c)
                if polys:
                    regions[c] = [[[x / w, y / h] for x, y in p] for p in polys]

        kps = None
        if pose.keypoints is not None:
            kps = [[float(x) / w, float(y) / h, float(c)] for x, y, c in pose.keypoints]
        return {
            "frame_size": [w, h],
            "pose": {"keypoints": kps, "names": KEYPOINTS, "confidence": round(pose.confidence, 3),
                     "orientation": pose.orientation, "orientation_confidence": round(pose.orientation_conf, 3),
                     "backend": pose.backend, "experimental": pose.experimental, "latency_ms": round(pose.latency_ms, 1),
                     "trusted": pose.confidence >= self.th["body_landmarks"]},
            "segmentation": {"available": seg.available, "confidence": round(seg.confidence, 3),
                             "ambiguous_fraction": round(seg.ambiguous_fraction, 3), "experimental": seg.experimental,
                             "reason": seg.reason, "regions": regions, "latency_ms": round(seg.latency_ms, 1)},
            "pleats": {"available": pl.available, "source": pl.source, "roi_source": roi_src,
                       "count": None if pl.count is None else round(pl.count, 1), "confidence": round(pl.confidence, 3),
                       "spacing_cv": pl.spacing_cv, "alignment_deg": pl.alignment_deg, "tilt_deg": pl.tilt_deg,
                       "symmetry": pl.symmetry, "experimental": pl.experimental, "reason": pl.reason},
            "stage": {"available": st.available, "stage": st.stage, "confidence": round(st.confidence, 3),
                      "source": st.source, "experimental": st.experimental, "reason": st.reason},
            "guidance": guidance,
            "timing_ms": {"total": round((time.perf_counter() - t0) * 1000, 1)},
        }

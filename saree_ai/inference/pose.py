"""Model A: human pose.

Backends (choose with config `pose.backend`):
  keypointrcnn  - torchvision Keypoint R-CNN ResNet50-FPN, COCO-pretrained. PRETRAINED BASELINE,
                  not ours. Weights are read from models/pose_baseline/ (never downloaded at
                  runtime; run scripts/download_models.py once).
  own           - our PoseHeatmapNet exported to ONNX (models/registry.json task 'pose_own').
  none          - always returns "no pose"; app degrades to neutral guidance.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..constants import KP
from ..registry import ROOT, active, loadable

BASELINE_DIR = ROOT / "models" / "pose_baseline"
BASELINE_FILE = "keypointrcnn_resnet50_fpn_coco-fc266e95.pth"


@dataclass
class PoseResult:
    keypoints: np.ndarray | None          # (17,3) x_px, y_px, conf in [0,1]
    person_score: float = 0.0
    bbox: list | None = None               # [x0,y0,x1,y1] px
    backend: str = "none"
    experimental: bool = False
    latency_ms: float = 0.0
    orientation: str = "unknown"
    orientation_conf: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Body confidence = person score x mean conf of the 8 torso/limb anchors we rely on."""
        if self.keypoints is None:
            return 0.0
        idx = [KP[n] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
                               "left_knee", "right_knee", "left_ankle", "right_ankle")]
        return float(self.person_score * self.keypoints[idx, 2].mean())


def estimate_orientation(k: np.ndarray, min_c=0.3) -> tuple[str, float]:
    """Coarse body orientation from 2D geometry; image coords are UN-mirrored camera coords.
    Returns one of front | back | turned_left | turned_right | side | unknown."""
    ls, rs, lh, rh = (k[KP[n]] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    if min(ls[2], rs[2], lh[2], rh[2]) < min_c:
        return "unknown", 0.0
    torso = np.linalg.norm((ls[:2] + rs[:2]) / 2 - (lh[:2] + rh[:2]) / 2) + 1e-6
    width_ratio = abs(ls[0] - rs[0]) / torso          # ~0.7-0.9 facing camera, ~0.1 in profile
    face = np.mean([k[KP[n], 2] for n in ("nose", "left_eye", "right_eye")])
    conf = float(min(ls[2], rs[2], lh[2], rh[2]))
    if width_ratio < 0.25:
        return "side", conf * 0.8
    # In an un-mirrored camera image a person facing the camera has their LEFT shoulder on the
    # image RIGHT (larger x). If the order flips they have their back to the camera.
    facing_camera = ls[0] > rs[0]
    if not facing_camera:
        return ("back", conf * 0.7) if face < 0.3 else ("unknown", 0.2)
    if width_ratio < 0.5:
        nose_x = k[KP["nose"], 0]
        mid = (ls[0] + rs[0]) / 2
        return ("turned_left" if nose_x > mid else "turned_right"), conf * 0.6
    return "front", conf


class _Base:
    name = "none"
    experimental = False

    def __call__(self, bgr: np.ndarray) -> PoseResult:
        return PoseResult(None, backend=self.name)


class NullPose(_Base):
    pass


class KeypointRCNNPose(_Base):
    name = "keypointrcnn_baseline"

    def __init__(self, device="cuda", long_side=640, min_person_score=0.6):
        import torch
        import torchvision
        from torchvision.models.detection import keypointrcnn_resnet50_fpn
        wfile = BASELINE_DIR / BASELINE_FILE
        if not wfile.exists():
            raise FileNotFoundError(
                f"Pose baseline weights not found at {wfile}. Run `python scripts/download_models.py` "
                "once while online; the app never downloads models at runtime.")
        self.torch = torch
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        m = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_keypoints=17)
        m.load_state_dict(torch.load(wfile, map_location="cpu", weights_only=True))
        m.eval().to(self.device)
        if self.device == "cuda":
            m.half()
        self.model = m
        self.long_side = long_side
        self.min_person_score = min_person_score
        _ = torchvision  # noqa: keep import explicit for licence traceability

    def __call__(self, bgr):
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        s = self.long_side / max(h, w)
        import cv2
        img = cv2.resize(bgr, (int(w * s), int(h * s))) if s < 1 else bgr
        s = img.shape[1] / w
        x = self.torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255)
        x = x.to(self.device)
        if self.device == "cuda":
            x = x.half()
        with self.torch.inference_mode():
            out = self.model([x])[0]
        scores = out["scores"].float().cpu().numpy()
        lat = (time.perf_counter() - t0) * 1000
        if len(scores) == 0 or scores[0] < self.min_person_score:
            return PoseResult(None, float(scores[0]) if len(scores) else 0.0, backend=self.name, latency_ms=lat)
        # Largest confident person = the user standing at the mirror.
        boxes = out["boxes"].float().cpu().numpy()
        ok = np.nonzero(scores >= self.min_person_score)[0]
        areas = (boxes[ok, 2] - boxes[ok, 0]) * (boxes[ok, 3] - boxes[ok, 1])
        i = int(ok[np.argmax(areas)])
        kp = out["keypoints"][i].float().cpu().numpy()          # (17,3) x,y,vis
        ks = out["keypoints_scores"][i].float().cpu().numpy()   # heatmap logits (uncalibrated)
        k = np.zeros((17, 3), np.float32)
        k[:, :2] = kp[:, :2] / s
        k[:, 2] = 1 / (1 + np.exp(-(ks - 1.0)))                 # squash to [0,1]; see docs/MODEL_CARDS.md
        ori, oc = estimate_orientation(k)
        return PoseResult(k, float(scores[i]), (boxes[i] / s).tolist(), self.name, False, lat, ori, oc,
                          {"num_people": int(len(ok))})


class OwnPoseONNX(_Base):
    """Top-down: needs a person box. Uses the previous frame's keypoint box, else full frame."""
    name = "own_pose"

    def __init__(self, entry: dict, providers=None):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(ROOT / entry["file"]), providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.h, self.w = entry.get("input_hw", [256, 192])
        self.experimental = entry.get("status") != "validated"
        self.prev_box = None

    def __call__(self, bgr):
        import cv2
        t0 = time.perf_counter()
        H, W = bgr.shape[:2]
        x0, y0, x1, y1 = self.prev_box or [0, 0, W, H]
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            self.prev_box = None
            return PoseResult(None, backend=self.name, experimental=self.experimental)
        inp = cv2.resize(crop, (self.w, self.h))[:, :, ::-1].astype(np.float32) / 255
        inp = ((inp - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
        hm = self.sess.run(None, {self.inp: inp})[0][0]            # (17, h/4, w/4)
        k = np.zeros((17, 3), np.float32)
        hh, ww = hm.shape[1:]
        for j in range(17):
            idx = int(hm[j].argmax()); py, px = divmod(idx, ww)
            k[j] = [x0 + (px + 0.5) / ww * (x1 - x0), y0 + (py + 0.5) / hh * (y1 - y0),
                    float(1 / (1 + np.exp(-hm[j].max())))]
        good = k[k[:, 2] > 0.3]
        if len(good) >= 6:
            bx0, by0 = good[:, :2].min(0); bx1, by1 = good[:, :2].max(0)
            pw, ph = (bx1 - bx0) * 0.25, (by1 - by0) * 0.15
            self.prev_box = [bx0 - pw, by0 - ph, bx1 + pw, by1 + ph]
            score = 1.0
        else:
            self.prev_box, score = None, 0.0
        ori, oc = estimate_orientation(k)
        return PoseResult(k, score, self.prev_box, self.name, self.experimental,
                          (time.perf_counter() - t0) * 1000, ori, oc)


def build_pose(cfg: dict, hw: dict) -> tuple[_Base, str]:
    backend = cfg.get("backend", "auto")
    msgs = []
    if backend in ("own", "auto"):
        e = active("pose_own")
        ok, why = loadable(e, cfg.get("min_status", "experimental"))
        if ok:
            return OwnPoseONNX(e), "own pose model"
        msgs.append(f"own pose model unavailable ({why})")
    if backend in ("keypointrcnn", "auto"):
        try:
            return KeypointRCNNPose(hw.get("pose_device", "cpu"), hw.get("pose_input_long_side", 640),
                                    cfg.get("min_person_score", 0.6)), "keypointrcnn baseline"
        except Exception as e:  # noqa: BLE001 - surface any load failure to the UI
            msgs.append(f"pose baseline unavailable ({e})")
    return NullPose(), "; ".join(msgs) or "pose disabled"


if os.environ.get("SAREE_AI_NO_TORCH_HOME") is None:
    # Make any torchvision weight cache live inside the project, never a shared global cache.
    os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch_home"))

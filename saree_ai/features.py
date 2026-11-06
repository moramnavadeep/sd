"""Hand-designed, interpretable per-frame features.

The SAME function is used (a) at training time on ground-truth masks/landmarks and
(b) at inference time on model predictions, so the draping-stage model and the rule-based
correction engine see identical feature definitions. All geometric quantities are normalised
by body size (hip width / torso length), so they are camera-resolution independent.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .constants import KP, SEG, SEG_CLASSES, PLEAT_QUALITY, PLEAT_MAX_COUNT

FEATURE_NAMES = (
    [f"frac_{c}" for c in SEG_CLASSES]                  # 7: share of person-bbox pixels
    + ["pallu_left_shoulder", "pallu_right_shoulder"]  # 2: pallu coverage near each shoulder
    + ["pleat_present", "pleat_cx_rel", "pleat_top_rel"]  # 3
    + ["waist_coverage"]                               # 1
    + ["pleat_count_norm"] + [f"pleat_q_{q}" for q in PLEAT_QUALITY] + ["pleat_secured"]  # 6
    + ["wrist_to_pleat_l", "wrist_to_pleat_r", "orientation_ratio", "pose_conf"]  # 4
)
FEATURE_DIM = len(FEATURE_NAMES)  # 23


@dataclass
class Geometry:
    """Body geometry derived from landmarks (pixel coordinates)."""
    hip_center: np.ndarray
    shoulder_center: np.ndarray
    hip_width: float
    torso_len: float
    l_shoulder: np.ndarray
    r_shoulder: np.ndarray
    valid: bool

    def to_dict(self):
        d = asdict(self)
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def body_geometry(kps: np.ndarray | None, min_conf: float = 0.3) -> Geometry:
    """kps: (17, 3) array of [x, y, conf_or_visibility]."""
    nan2 = np.array([np.nan, np.nan])
    if kps is None:
        return Geometry(nan2, nan2, np.nan, np.nan, nan2, nan2, False)
    kps = np.asarray(kps, dtype=np.float32)
    need = [KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]]
    ok = all(kps[i, 2] >= min_conf for i in need)
    ls, rs = kps[KP["left_shoulder"], :2], kps[KP["right_shoulder"], :2]
    lh, rh = kps[KP["left_hip"], :2], kps[KP["right_hip"], :2]
    hc, sc = (lh + rh) / 2, (ls + rs) / 2
    hip_w = float(np.linalg.norm(lh - rh))
    torso = float(np.linalg.norm(sc - hc))
    ok = ok and hip_w > 1 and torso > 1
    return Geometry(hc, sc, hip_w, torso, ls, rs, bool(ok))


def _disc_fraction(mask_bool: np.ndarray, center: np.ndarray, radius: float) -> float:
    h, w = mask_bool.shape
    if not np.all(np.isfinite(center)) or radius <= 0:
        return 0.0
    yy, xx = np.ogrid[:h, :w]
    disc = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
    n = disc.sum()
    return float((mask_bool & disc).sum() / n) if n else 0.0


def frame_features(mask: np.ndarray | None,
                   kps: np.ndarray | None,
                   pleat: dict | None = None,
                   seg_probs_available: bool = True) -> tuple[np.ndarray, dict]:
    """Compute the FEATURE_DIM vector.

    mask : (H, W) uint8 class-index map (prediction or ground truth), or None.
    kps  : (17, 3) [x, y, conf] in the SAME pixel space as mask, or None.
    pleat: dict with optional keys count (float), quality_probs (list[4]), secured_prob (float).
    Returns (vector, extras) where extras holds human-readable intermediate values that the
    correction engine uses (e.g. pleat centroid offset in hip-widths).
    """
    f = np.zeros(FEATURE_DIM, dtype=np.float32)
    extras: dict = {}
    geo = body_geometry(kps)
    extras["geometry_valid"] = geo.valid

    if mask is not None:
        person_px = mask != SEG["background"]
        ys, xs = np.nonzero(person_px)
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            crop = mask[y0:y1, x0:x1]
            valid = crop != 255
            total = max(int(valid.sum()), 1)
            for i in range(len(SEG_CLASSES)):
                f[i] = float((crop == i).sum() / total)
        pallu = mask == SEG["pallu"]
        pleats = mask == SEG["pleats"]
        if geo.valid:
            r = 0.35 * geo.hip_width + 0.1 * geo.torso_len
            f[7] = _disc_fraction(pallu, geo.l_shoulder, r)
            f[8] = _disc_fraction(pallu, geo.r_shoulder, r)
        n_pleat = int(pleats.sum())
        if n_pleat > 0:
            pys, pxs = np.nonzero(pleats)
            f[9] = 1.0
            if geo.valid:
                cx_rel = (pxs.mean() - geo.hip_center[0]) / geo.hip_width
                top_rel = (pys.min() - geo.hip_center[1]) / geo.torso_len
                f[10], f[11] = float(np.clip(cx_rel, -3, 3)), float(np.clip(top_rel, -3, 3))
                extras["pleat_cx_rel"] = float(cx_rel)
                extras["pleat_top_rel"] = float(top_rel)
                extras["pleat_centroid_px"] = [float(pxs.mean()), float(pys.mean())]
        if geo.valid:
            # Waist band coverage: fraction of a horizontal strip at hip height (between hips,
            # extended 25% each side) that is saree waist_wrap / pleats / saree_body.
            y = int(round(geo.hip_center[1]))
            band_h = max(2, int(0.08 * geo.torso_len))
            xa = int(geo.hip_center[0] - 0.75 * geo.hip_width)
            xb = int(geo.hip_center[0] + 0.75 * geo.hip_width)
            h, w = mask.shape
            ya, yb = max(0, y - band_h), min(h, y + band_h)
            xa, xb = max(0, xa), min(w, xb)
            if yb > ya and xb > xa:
                strip = mask[ya:yb, xa:xb]
                cover = np.isin(strip, [SEG["waist_wrap"], SEG["pleats"], SEG["saree_body"], SEG["pallu"]])
                f[12] = float(cover.mean())

    if pleat:
        if pleat.get("count") is not None:
            f[13] = float(pleat["count"]) / PLEAT_MAX_COUNT
        q = pleat.get("quality_probs")
        if q is not None:
            f[14:18] = np.asarray(q, dtype=np.float32)
        if pleat.get("secured_prob") is not None:
            f[18] = float(pleat["secured_prob"])

    if kps is not None and geo.valid:
        k = np.asarray(kps, dtype=np.float32)
        target = extras.get("pleat_centroid_px")
        target = np.array(target) if target is not None else geo.hip_center
        for j, name in enumerate(["left_wrist", "right_wrist"]):
            wv = k[KP[name]]
            f[19 + j] = float(np.clip(np.linalg.norm(wv[:2] - target) / geo.torso_len, 0, 5)) if wv[2] > 0.3 else 5.0
        sw = float(np.linalg.norm(geo.l_shoulder - geo.r_shoulder))
        f[21] = float(np.clip(sw / geo.torso_len, 0, 3))
        f[22] = float(np.mean(k[:, 2].clip(0, 1)))
    return f, extras

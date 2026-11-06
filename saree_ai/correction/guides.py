"""Overlay primitives anchored to live landmarks.

All coordinates are returned NORMALISED to [0,1] in the UN-mirrored camera frame. The frontend
alone applies the mirror + object-fit transform, so there is exactly one place where mirroring
happens (frontend/js/geometry.js).

Primitive schema (JSON):
  {"type": "marker", "id", "at": [x,y], "label"?}
  {"type": "arrow",  "id", "points": [[x,y], ...], "label"?}       polyline with arrow head
  {"type": "zone",   "id", "polygon": [[x,y], ...], "label"?}
  {"type": "line",   "id", "points": [[x,y],[x,y]], "label"?}
  {"type": "folds",  "id", "lines": [[[x,y],[x,y]], ...], "quality": [...]}
  {"type": "frame",  "id", "box": [x0,y0,x1,y1], "ok": bool}
"""
from __future__ import annotations

import numpy as np

from ..constants import KP
from ..features import body_geometry


def _n(pt, w, h):
    return [float(pt[0]) / w, float(pt[1]) / h]


def build_guides(guide_ids, state, th, focus=None):
    w, h = state.frame_w, state.frame_h
    k = state.pose.keypoints
    out = []
    geo = body_geometry(k, th["keypoint"]) if k is not None else None
    for gid in guide_ids:
        if gid == "frame_box":
            out.append({"type": "frame", "id": gid, "box": None, "ok": False})  # frontend owns geometry
            continue
        if geo is None or not geo.valid:
            continue  # cannot anchor -> draw nothing rather than something misplaced
        hc, sc, hw_, tl = geo.hip_center, geo.shoulder_center, geo.hip_width, geo.torso_len
        up = (sc - hc) / (np.linalg.norm(sc - hc) + 1e-6)          # body "up" direction
        lr = (k[KP["left_hip"], :2] - k[KP["right_hip"], :2])      # toward user's LEFT
        lr = lr / (np.linalg.norm(lr) + 1e-6)
        navel = hc + up * 0.18 * tl
        ankle_y = max(k[KP["left_ankle"], 1], k[KP["right_ankle"], 1]) if min(k[KP["left_ankle"], 2], k[KP["right_ankle"], 2]) >= th["keypoint"] else None

        if gid == "tuck_start_marker":
            out.append({"type": "marker", "id": gid, "at": _n(navel - lr * 0.25 * hw_, w, h), "label": "Tuck here"})
        elif gid == "hem_line" and ankle_y is not None:
            y = ankle_y + 0.06 * tl
            out.append({"type": "line", "id": gid, "points": [_n([hc[0] - 0.9 * hw_, y], w, h), _n([hc[0] + 0.9 * hw_, y], w, h)],
                        "label": "Hem just touching the floor"})
        elif gid == "wrap_arrow":
            a = navel - lr * 0.25 * hw_
            b = navel + lr * 0.75 * hw_
            out.append({"type": "arrow", "id": gid, "points": [_n(a, w, h), _n(b, w, h)], "label": "Across to your left, round the back"})
        elif gid in ("pleat_zone", "pleat_tuck_marker"):
            c = navel + lr * 0.10 * hw_
            if gid == "pleat_tuck_marker":
                out.append({"type": "marker", "id": gid, "at": _n(c, w, h), "label": "Tuck pleats here"})
            else:
                half = 0.32 * hw_
                bottom_y = ankle_y if ankle_y is not None else hc[1] + 2.2 * tl
                poly = [c - lr * half, c + lr * half, [c[0] + half * 1.15 * lr[0], bottom_y], [c[0] - half * 1.15 * lr[0], bottom_y]]
                out.append({"type": "zone", "id": gid, "polygon": [_n(p, w, h) for p in poly], "label": "Pleats: 5-7 folds"})
        elif gid == "pleat_folds" and state.pleat.available and state.pleat.fold_x and state.pleat.roi \
                and state.pleat.confidence >= th["pleat"]:  # never draw folds we are unsure about
            x0, y0, x1, y1 = state.pleat.roi
            lines = [[_n([x, y0], w, h), _n([x, y1], w, h)] for x in state.pleat.fold_x]
            out.append({"type": "folds", "id": gid, "lines": lines, "confidence": state.pleat.confidence,
                        "experimental": state.pleat.experimental})
        elif gid == "pallu_path":
            rs, ls = k[KP["right_shoulder"], :2], k[KP["left_shoulder"], :2]
            start = hc - lr * 0.55 * hw_
            mid = (start + ls) / 2 + up * 0.05 * tl
            out.append({"type": "arrow", "id": gid, "points": [_n(start, w, h), _n(mid, w, h), _n(ls - up * 0.02 * tl, w, h)],
                        "label": "Across the chest, over the LEFT shoulder"})
            _ = rs
        elif gid == "shoulder_pin_marker":
            out.append({"type": "marker", "id": gid, "at": _n(k[KP["left_shoulder"], :2], w, h), "label": "Pin here"})
    for g in out:
        g["focus"] = (g["id"] == focus)
    return out

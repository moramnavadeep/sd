"""Model E building blocks: interpretable per-frame checks.

Every check returns an Outcome with one of four statuses:
  pass      - inputs were confident and the condition holds
  fail      - inputs were confident and the condition does NOT hold   -> candidate correction
  uncertain - an input was missing / below threshold                    -> NEVER a correction
  n/a       - check does not apply in the current situation

A check may only return 'fail' after verifying every input it uses is above its threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import KP, SEG
from ..features import FEATURE_NAMES, body_geometry
from ..i18n import T

FI = {n: i for i, n in enumerate(FEATURE_NAMES)}


@dataclass
class FrameState:
    frame_w: int
    frame_h: int
    pose: object                  # PoseResult
    seg: object                   # SegResult
    pleat: object                 # PleatResult
    stage: object                 # StageResult
    feats: np.ndarray | None
    extras: dict = field(default_factory=dict)


@dataclass
class Outcome:
    check: str
    status: str                   # pass | fail | uncertain | n/a
    message: str = ""
    label: str | None = None      # correction label id (constants.CORRECTION_LABELS) when failing
    confidence: float = 0.0
    experimental: bool = False
    needs: list = field(default_factory=list)
    focus: str | None = None      # guide id to highlight


def _pose_ok(s: FrameState, th):
    return s.pose.keypoints is not None and s.pose.confidence >= th["body_landmarks"]


def _seg_ok(s: FrameState, th):
    return (s.seg.available and s.seg.confidence >= th["segmentation"]
            and s.seg.ambiguous_fraction <= th["segmentation_max_ambiguous"])


def _uncertain(name, why, needs):
    return Outcome(name, "uncertain", why, needs=needs)


# ----------------------------------------------------------------------------- checks
def full_body_visible(s: FrameState, p, th):
    if s.pose.keypoints is None:
        return Outcome("full_body_visible", "fail", T("I can't see you. Step into the frame."), None, 0.0,
                       needs=["pose"], focus="frame_box")
    k = s.pose.keypoints
    need = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"]
    vis = [k[KP[n], 2] >= th["keypoint"] for n in need]
    x0, y0, x1, y1 = s.extras.get("visible_px", [0, 0, s.frame_w, s.frame_h])
    inside = all(x0 <= k[KP[n], 0] < x1 and y0 <= k[KP[n], 1] < y1 for n in need)
    c = s.pose.confidence
    if all(vis) and inside and c >= th["body_landmarks"]:
        return Outcome("full_body_visible", "pass", T("Whole body visible."), None, c)
    if all(vis) and not inside:
        return Outcome("full_body_visible", "fail", T("Part of you is outside the mirror view. Step back so your head and feet are on screen."),
                       None, c, needs=["pose"], focus="frame_box")
    missing = [n.replace("_", " ") for n, v in zip(need, vis) if not v]
    if any("ankle" in m for m in missing):
        msg = T("Step back a little so your feet are visible.")
    elif any(m == "nose" for m in missing):
        msg = T("Move so your head is inside the frame.")
    else:
        msg = T("Face the camera and stand inside the frame. Better lighting may help.")
    # This is a positioning request, not a draping correction -> no correction label.
    return Outcome("full_body_visible", "fail", msg, None, c, needs=["pose"], focus="frame_box")


def waist_wrap_coverage(s, p, th):
    if not (_pose_ok(s, th) and _seg_ok(s, th)):
        return _uncertain("waist_wrap_coverage", T("I can't check the waist wrap reliably yet."), ["pose", "segmentation"])
    cov = float(s.feats[FI["waist_coverage"]])
    lim = p.get("min", 0.6)
    exp = s.seg.experimental
    if cov >= lim:
        return Outcome("waist_wrap_coverage", "pass", T("Waist wrap looks complete from the front."), None, s.seg.confidence, exp)
    return Outcome("waist_wrap_coverage", "fail", T("The wrap doesn't seem to go all the way across your waist. Bring it fully round to the front."),
                   "waist_wrap_loose", s.seg.confidence, exp, focus="wrap_arrow")


def hem_level(s, p, th):
    if not (_pose_ok(s, th) and _seg_ok(s, th)):
        return _uncertain("hem_level", T("I can't check the hem reliably yet."), ["pose", "segmentation"])
    k = s.pose.keypoints
    if min(k[KP["left_ankle"], 2], k[KP["right_ankle"], 2]) < th["keypoint"]:
        return _uncertain("hem_level", T("I need to see both ankles to check the hem."), ["pose"])
    m = np.isin(s.seg.mask, [SEG["saree_body"], SEG["pleats"], SEG["waist_wrap"]])
    geo = body_geometry(k)
    xs = [k[KP["right_ankle"], 0], k[KP["left_ankle"], 0]]
    bottoms = []
    for x in xs:
        x0, x1 = int(max(0, x - 0.15 * geo.hip_width)), int(min(s.frame_w, x + 0.15 * geo.hip_width))
        cols = m[:, x0:x1]
        rows = np.nonzero(cols.any(1))[0]
        if len(rows) == 0:
            return _uncertain("hem_level", T("I can't find the saree hem near your feet."), ["segmentation"])
        bottoms.append(rows.max())
    diff = abs(bottoms[0] - bottoms[1]) / geo.torso_len
    if diff <= p.get("max_diff", 0.15):
        return Outcome("hem_level", "pass", T("Hem looks level."), None, s.seg.confidence, s.seg.experimental)
    higher = "right" if bottoms[0] < bottoms[1] else "left"
    return Outcome("hem_level", "fail", T("The hem looks higher on your {higher} side. Lower it so it just touches the floor all round.", higher=higher),
                   "hem_uneven", s.seg.confidence, s.seg.experimental, focus="hem_line")


def _pleat_ok(s, th):
    return s.pleat.available and s.pleat.confidence >= th["pleat"]


def pleat_count(s, p, th):
    if not _pleat_ok(s, th) or s.pleat.count is None or s.pleat.count_conf < th["pleat"]:
        return _uncertain("pleat_count", T("I can't count the pleats clearly. Hold them spread out in front of you, or confirm manually."), ["pleat"])
    n = int(round(s.pleat.count))
    lo, hi = p.get("min", 5), p.get("max", 9)
    exp = s.pleat.experimental
    if lo <= n <= hi:
        return Outcome("pleat_count", "pass", T("About {n} pleats - good.", n=n), None, s.pleat.count_conf, exp)
    if n < lo:
        return Outcome("pleat_count", "fail", T("I can see about {n} pleats. Aim for {lo} to 7.", lo=lo, n=n), "pleats_too_few", s.pleat.count_conf, exp, focus="pleat_folds")
    return Outcome("pleat_count", "fail", T("I can see about {n} pleats; they may be too narrow. Aim for 5 to 7 pleats about 5 inches wide.", n=n), "pleats_too_many", s.pleat.count_conf, exp, focus="pleat_folds")


def pleat_spacing(s, p, th):
    if not _pleat_ok(s, th):
        return _uncertain("pleat_spacing", T("I can't measure pleat spacing reliably."), ["pleat"])
    cv = s.pleat.spacing_cv
    q = s.pleat.quality_probs
    q_uneven = q[1] if q is not None else None
    if cv is None and q_uneven is None:
        return _uncertain("pleat_spacing", T("I can't measure pleat spacing reliably."), ["pleat"])
    bad = (cv is not None and cv > p.get("max_cv", 0.35)) or (q_uneven is not None and q_uneven > 0.6)
    exp = s.pleat.experimental
    if not bad:
        return Outcome("pleat_spacing", "pass", T("Pleat widths look even."), None, s.pleat.confidence, exp)
    return Outcome("pleat_spacing", "fail", T("Some pleats look wider than others. Try to make each fold the same width."),
                   "pleats_uneven_spacing", s.pleat.confidence, exp, focus="pleat_folds")


def pleat_alignment(s, p, th):
    if not _pleat_ok(s, th):
        return _uncertain("pleat_alignment", T("I can't check pleat alignment reliably."), ["pleat"])
    dev, tilt = s.pleat.alignment_deg, s.pleat.tilt_deg
    q = s.pleat.quality_probs
    q_mis = q[2] if q is not None else None
    if dev is None and q_mis is None:
        return _uncertain("pleat_alignment", T("I can't check pleat alignment reliably."), ["pleat"])
    bad = ((dev is not None and dev > p.get("max_dev_deg", 8)) or
           (tilt is not None and abs(tilt) > p.get("max_tilt_deg", 12)) or
           (q_mis is not None and q_mis > 0.6))
    exp = s.pleat.experimental
    if not bad:
        return Outcome("pleat_alignment", "pass", T("Pleats hang straight."), None, s.pleat.confidence, exp)
    return Outcome("pleat_alignment", "fail", T("The pleats are slanting. Line up the top edges and let them fall straight down."),
                   "pleats_misaligned", s.pleat.confidence, exp, focus="pleat_folds")


def pleat_center(s, p, th):
    if not (_pose_ok(s, th) and _seg_ok(s, th)) or "pleat_cx_rel" not in s.extras:
        return _uncertain("pleat_center", T("I can't locate the pleats relative to your waist reliably."), ["pose", "segmentation"])
    cx = s.extras["pleat_cx_rel"]          # + = toward user's LEFT hip (un-mirrored camera image)
    target, tol = p.get("target", 0.10), p.get("tolerance", 0.35)
    exp = s.seg.experimental
    if abs(cx - target) <= tol:
        return Outcome("pleat_center", "pass", T("Pleats are well placed at the navel."), None, s.seg.confidence, exp)
    side = "left" if cx > target else "right"
    return Outcome("pleat_center", "fail", T("The pleats are too far to your {side}. Move them to just left of your navel.", side=side),
                   "pleats_off_center", s.seg.confidence, exp, focus="pleat_tuck_marker")


def pleats_tucked(s, p, th):
    have_model = s.pleat.available and s.pleat.secured_prob is not None and s.pleat.confidence >= th["pleat"]
    have_geom = _pose_ok(s, th) and _seg_ok(s, th) and "pleat_top_rel" in s.extras
    if not (have_model or have_geom):
        return _uncertain("pleats_tucked", T("I can't tell whether the pleats are tucked. Confirm manually when done."), ["pleat", "segmentation"])
    # pleat_top_rel: (top of pleats - hip line) / torso length. Tucked pleats start at the waist,
    # i.e. slightly ABOVE the hip keypoints (negative) - pleats held in the hand start lower.
    ok_geom = (not have_geom) or s.extras["pleat_top_rel"] < p.get("max_top_rel", 0.15)
    ok_model = (not have_model) or s.pleat.secured_prob >= 0.5
    exp = (s.pleat.experimental if have_model else False) or (s.seg.experimental if have_geom else False)
    conf = max(s.pleat.confidence if have_model else 0, s.seg.confidence if have_geom else 0)
    if ok_geom and ok_model:
        return Outcome("pleats_tucked", "pass", T("Pleats are tucked in."), None, conf, exp)
    return Outcome("pleats_tucked", "fail", T("The pleats don't look tucked in yet. Push the top of the pleats into the petticoat at your navel."),
                   "pleats_not_tucked", conf, exp, focus="pleat_tuck_marker")


def pallu_shoulder(s, p, th):
    if not (_pose_ok(s, th) and _seg_ok(s, th)):
        return _uncertain("pallu_shoulder", T("I can't see the pallu clearly enough to check it."), ["pose", "segmentation"])
    k = s.pose.keypoints
    if min(k[KP["left_shoulder"], 2], k[KP["right_shoulder"], 2]) < th["keypoint"]:
        return _uncertain("pallu_shoulder", T("I need to see both shoulders to check the pallu."), ["pose"])
    left, right = float(s.feats[FI["pallu_left_shoulder"]]), float(s.feats[FI["pallu_right_shoulder"]])
    exp = s.seg.experimental
    if left >= p.get("min_cover", 0.15) and left >= right:
        return Outcome("pallu_shoulder", "pass", T("Pallu is over your left shoulder."), None, s.seg.confidence, exp)
    if right >= p.get("min_cover", 0.15) and right > left:
        return Outcome("pallu_shoulder", "fail", T("The pallu is on your right shoulder. In the Nivi style it goes over the LEFT shoulder."),
                       "pallu_wrong_shoulder", s.seg.confidence, exp, focus="pallu_path")
    if not p.get("require_on_shoulder", True):
        # While the user is still moving the pallu, "not on the shoulder yet" is progress, not an error.
        return Outcome("pallu_shoulder", "n/a", T("Bring the pallu up across your chest to your left shoulder."), None, s.seg.confidence, exp)
    return Outcome("pallu_shoulder", "fail", T("Bring the pallu up across your chest and over your left shoulder."),
                   "pallu_not_on_shoulder", s.seg.confidence, exp, focus="pallu_path")


def pallu_length_manual(s, p, th):
    # A front-facing camera cannot see the pallu's length behind the user; be honest about it.
    return Outcome("pallu_length_manual", "uncertain",
                   T("I can't see the pallu length from the front. Turn sideways and check it reaches the back of your knees."),
                   needs=["manual"])


CHECKS = {f.__name__: f for f in [full_body_visible, waist_wrap_coverage, hem_level, pleat_count, pleat_spacing,
                                    pleat_alignment, pleat_center, pleats_tucked, pallu_shoulder, pallu_length_manual]}

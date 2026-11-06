"""Procedural SYNTHETIC data generator.

Purpose: exercise and debug the full training / evaluation / export pipeline before real,
consented data exists, and optionally pre-train before fine-tuning on real data.

What it is NOT: a substitute for real data. The renderings are flat, cartoon-like figures with
procedurally textured fabric. Metrics on this data say nothing about real-world accuracy.
Every sample is written with source.type = "synthetic"; prepare_dataset.py never places
synthetic samples in the test split.

Usage:
  python training/data/synth_generate.py --out data/synthetic_v1 --participants 60 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from saree_ai.constants import SEG, KEYPOINTS, DRAPE_STAGES, SCHEMA_VERSION  # noqa: E402

GEN_VERSION = "synth-v1"
W, H = 256, 384

TEMPLATE = {  # facing camera: the person's LEFT side is image RIGHT
    "nose": (128, 50), "left_eye": (134, 45), "right_eye": (122, 45), "left_ear": (142, 50), "right_ear": (114, 50),
    "left_shoulder": (158, 88), "right_shoulder": (98, 88), "left_elbow": (166, 142), "right_elbow": (90, 142),
    "left_wrist": (170, 192), "right_wrist": (86, 192), "left_hip": (148, 192), "right_hip": (108, 192),
    "left_knee": (146, 272), "right_knee": (110, 272), "left_ankle": (144, 352), "right_ankle": (112, 352),
}


class Person:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.scale = rng.uniform(0.78, 1.0)
        self.dx, self.dy = rng.uniform(-30, 30), rng.uniform(-8, 20)
        self.width = rng.uniform(0.9, 1.15)
        self.skin = np.array([rng.uniform(60, 120), rng.uniform(100, 160), rng.uniform(150, 215)])  # BGR
        self.blouse = rng.uniform(20, 235, 3)
        self.petticoat = rng.uniform(20, 235, 3)
        self.saree = self._fabric_spec()
        self.bg = rng.uniform(30, 230, 3), rng.uniform(30, 230, 3)

    def _fabric_spec(self):
        r = self.rng
        h = r.uniform(0, 180)
        s = r.uniform(60, 255)
        v = r.choice([r.uniform(40, 90), r.uniform(120, 250)])  # dark or light sarees
        base = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0].astype(float)
        return {"base": base, "pattern": r.choice(["plain", "striped", "printed", "woven_border", "checked"]),
                "accent": r.uniform(0, 255, 3), "alpha": r.choice([1.0, 1.0, 1.0, r.uniform(0.6, 0.85)]),
                "darkness": "dark" if v < 100 else "light"}

    def keypoints(self, arm_pose: str) -> np.ndarray:
        k = np.array([TEMPLATE[n] for n in KEYPOINTS], np.float32)
        c = np.array([128, 200])
        k[:, 0] = c[0] + (k[:, 0] - c[0]) * self.width
        k = c + (k - c) * self.scale + [self.dx, self.dy]
        idx = {n: i for i, n in enumerate(KEYPOINTS)}
        hc = (k[idx["left_hip"]] + k[idx["right_hip"]]) / 2
        T = np.linalg.norm((k[idx["left_shoulder"]] + k[idx["right_shoulder"]]) / 2 - hc)
        if arm_pose == "hold_pleats":    # both hands in front, below the navel
            k[idx["left_wrist"]] = hc + [8, 0.35 * T]; k[idx["right_wrist"]] = hc + [-8, 0.35 * T]
            k[idx["left_elbow"]] = hc + [30, -0.05 * T]; k[idx["right_elbow"]] = hc + [-30, -0.05 * T]
        elif arm_pose == "pallu":        # right hand lifting fabric toward the left shoulder
            k[idx["right_wrist"]] = (k[idx["left_shoulder"]] + hc) / 2 + [0, -10]
            k[idx["right_elbow"]] = hc + [-40, -0.45 * T]
        k += self.rng.normal(0, 1.5, k.shape)
        return np.concatenate([k, np.full((17, 1), 2.0)], 1)


def texture(spec, h, w, rng, phase=0.0):
    img = np.ones((h, w, 3), np.float32) * spec["base"]
    yy, xx = np.mgrid[:h, :w]
    p = spec["pattern"]
    if p == "striped":
        img[((xx + yy * 0.3 + phase) // 6 % 2) == 0] = spec["accent"] * 0.5 + spec["base"] * 0.5
    elif p == "printed":
        n = cv2.resize(rng.uniform(0, 1, (max(2, h // 8), max(2, w // 8))).astype(np.float32), (w, h))
        img = img * (0.75 + 0.5 * n[..., None])
    elif p == "checked":
        m = ((xx // 10 + yy // 10) % 2) == 0
        img[m] = img[m] * 0.8
    elif p == "woven_border":
        img[-max(3, h // 12):] = spec["accent"]
    return img


class Canvas:
    def __init__(self, person: Person, rng):
        a, b = person.bg
        g = np.linspace(0, 1, H)[:, None, None]
        self.img = (a * (1 - g) + b * g) * np.ones((H, W, 3))
        self.img += rng.normal(0, 6, self.img.shape)
        self.mask = np.full((H, W), SEG["background"], np.uint8)

    def poly(self, pts, color, cls, alpha=1.0, tex=None):
        pts = np.round(np.asarray(pts)).astype(np.int32)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        self.fill(m.astype(bool), color, cls, alpha, tex)

    def fill(self, m, color, cls, alpha=1.0, tex=None):
        src = tex if tex is not None else np.ones((H, W, 3)) * color
        self.img[m] = self.img[m] * (1 - alpha) + src[m] * alpha
        if cls is not None:
            self.mask[m] = cls

    def limb(self, a, b, r, color, cls):
        m = np.zeros((H, W), np.uint8)
        cv2.line(m, tuple(np.int32(a)), tuple(np.int32(b)), 1, int(2 * r))
        self.fill(m.astype(bool), color, cls)


def render(person: Person, stage: str, variant: dict, rng) -> tuple:
    arm = {"pleats_forming": "hold_pleats", "pallu_in_progress": "pallu"}.get(stage, "down")
    k = person.keypoints(arm)
    I = {n: i for i, n in enumerate(KEYPOINTS)}
    P = lambda n: k[I[n], :2]  # noqa: E731
    cv = Canvas(person, rng)
    hc = (P("left_hip") + P("right_hip")) / 2
    sc = (P("left_shoulder") + P("right_shoulder")) / 2
    T = np.linalg.norm(sc - hc)
    hw = np.linalg.norm(P("left_hip") - P("right_hip"))
    lr = (P("left_hip") - P("right_hip")) / hw
    floor = max(P("left_ankle")[1], P("right_ankle")[1]) + 8
    skin = person.skin

    # legs + petticoat (other clothing)
    for s in ("left", "right"):
        cv.limb(P(f"{s}_hip"), P(f"{s}_knee"), 11 * person.scale, skin, SEG["person"])
        cv.limb(P(f"{s}_knee"), P(f"{s}_ankle"), 9 * person.scale, skin, SEG["person"])
    cv.poly([hc - lr * 0.75 * hw + [0, -0.05 * T], hc + lr * 0.75 * hw + [0, -0.05 * T],
             [hc[0] + 0.9 * hw, floor - 6], [hc[0] - 0.9 * hw, floor - 6]], person.petticoat, SEG["other_clothing"])
    # torso skin, blouse, head
    cv.poly([P("right_shoulder"), P("left_shoulder"), P("left_hip") + [4, 0], P("right_hip") - [4, 0]], skin, SEG["person"])
    cv.poly([P("right_shoulder") + [-6, -4], P("left_shoulder") + [6, -4], P("left_shoulder") + [2, 0.45 * T],
             P("right_shoulder") + [-2, 0.45 * T]], person.blouse, SEG["other_clothing"])
    cv.limb(sc + [0, -4], P("nose") + [0, 12], 7 * person.scale, skin, SEG["person"])
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, tuple(np.int32(P("nose") + [0, -4])), (int(17 * person.scale), int(22 * person.scale)), 0, 0, 360, 1, -1)
    cv.fill(m.astype(bool), skin, SEG["person"])
    hair = np.zeros((H, W), np.uint8)
    cv2.ellipse(hair, tuple(np.int32(P("nose") + [0, -14])), (int(18 * person.scale), int(14 * person.scale)), 0, 180, 360, 1, -1)
    cv.fill(hair.astype(bool), np.array([20, 20, 25]), SEG["person"])

    def arms():
        for s in ("left", "right"):
            cv.limb(P(f"{s}_shoulder"), P(f"{s}_elbow"), 7 * person.scale, skin, SEG["person"])
            cv.limb(P(f"{s}_elbow"), P(f"{s}_wrist"), 6 * person.scale, skin, SEG["person"])
    arms()

    spec = person.saree
    tex = texture(spec, H, W, rng)
    a = spec["alpha"]
    pleat_ann, corrections = None, []
    si = DRAPE_STAGES.index(stage)
    waist_top, waist_bot = hc[1] - 0.10 * T, hc[1] + 0.07 * T
    hem_tilt = variant.get("hem_tilt", 0.0)

    if si >= 1:  # wrapping started: partial wrap from right-of-navel across to the left
        x_from = hc[0] - lr[0] * 0.3 * hw
        x_to = hc[0] + lr[0] * (0.75 if si >= 2 else rng.uniform(0.0, 0.4)) * hw
        x_left = hc[0] - 0.75 * hw if si >= 2 else x_from
        cv.poly([[x_left - 0.1 * hw, waist_bot], [x_to + 0.15 * hw, waist_bot],
                 [x_to + 0.3 * hw, floor + hem_tilt * T], [x_left - 0.25 * hw, floor - hem_tilt * T]], None, SEG["saree_body"], a, tex)
        cv.poly([[x_left, waist_top], [x_to, waist_top], [x_to, waist_bot], [x_left, waist_bot]], None, SEG["waist_wrap"], a,
                texture(spec, H, W, rng, phase=3) * 0.92)
        if hem_tilt:
            corrections.append("hem_uneven")
        if variant.get("loose_wrap") and si >= 2:
            corrections.append("waist_wrap_loose")

    if si >= 3:
        n = variant["n_pleats"]
        q = variant["pleat_quality"]
        fold_w = rng.uniform(0.09, 0.12) * hw * 6 / max(n, 1) * (1.0 if n <= 7 else 0.8)
        widths = np.full(n, fold_w)
        if q == "uneven_spacing":
            widths = widths * rng.uniform(0.45, 1.7, n)
        cx = hc[0] + lr[0] * (variant["pleat_offset"]) * hw
        total = widths.sum()
        tucked = si >= 4 and not variant.get("untucked", False)
        top = waist_top + 0.02 * T if tucked else hc[1] + 0.30 * T
        x = cx - total / 2
        tilts = np.zeros(n)
        if q == "misaligned":
            tilts = rng.uniform(-0.35, 0.35, n) + rng.choice([-0.25, 0.25])
        fold_lines = []
        for j in range(n):
            x0, x1 = x, x + widths[j]
            dx = tilts[j] * (floor - top)
            shade = 0.72 if j % 2 else 1.08
            if q == "loose_or_unformed":
                shade = rng.uniform(0.9, 1.05)
            cv.poly([[x0, top], [x1, top], [x1 + dx, floor], [x0 + dx, floor]], None, SEG["pleats"], a, np.clip(tex * shade, 0, 255))
            fold_lines.append([[float(x0), float(top)], [float(x0 + dx), float(floor)]])
            x = x1
        fold_lines.append([[float(x), float(top)], [float(x + tilts[-1] * (floor - top)), float(floor)]])
        xs = [p[0] for l in fold_lines for p in l]
        pleat_ann = {"visible": True, "roi": [float(min(xs)) - 4, float(top) - 4, float(max(xs)) + 4, float(floor)],
                     "fold_lines": fold_lines, "count": int(n), "count_certain": q != "loose_or_unformed",
                     "quality": q, "secured": bool(tucked)}
        if q == "uneven_spacing": corrections.append("pleats_uneven_spacing")
        if q == "misaligned": corrections.append("pleats_misaligned")
        if n < 5: corrections.append("pleats_too_few")
        if n > 9: corrections.append("pleats_too_many")
        if abs(variant["pleat_offset"] - 0.1) > 0.35: corrections.append("pleats_off_center")
        if si >= 4 and not tucked: corrections.append("pleats_not_tucked")

    if si >= 5:
        target = P("right_shoulder") if variant.get("wrong_shoulder") else P("left_shoulder")
        start = hc - lr * 0.6 * hw + [0, -0.05 * T]
        end = target if si >= 6 else start + (target - start) * rng.uniform(0.45, 0.7)
        d = end - start; nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
        wdt = 0.28 * hw
        cv.poly([start - nrm * wdt, start + nrm * wdt, end + nrm * wdt * 0.8 + [0, -6], end - nrm * wdt * 0.8 + [0, -6]],
                None, SEG["pallu"], a, texture(spec, H, W, rng, phase=7) * 1.05)
        if si >= 6 and variant.get("wrong_shoulder"): corrections.append("pallu_wrong_shoulder")

    if arm != "down":
        arms()  # hands occlude fabric

    # global photometric variation (lighting conditions)
    img = cv.img
    light = rng.choice(["bright", "normal", "dim", "colored", "backlit"])
    if light == "bright": img = img * rng.uniform(1.1, 1.3)
    elif light == "dim": img = img * rng.uniform(0.35, 0.6)
    elif light == "colored": img = img * rng.uniform(0.7, 1.2, 3)
    elif light == "backlit":
        g = np.linspace(-1, 1, W)[None, :, None]
        img = img * (0.6 + 0.4 * np.abs(g))
    img = img + rng.normal(0, rng.uniform(2, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, cv.mask, k, pleat_ann, sorted(set(corrections)), light


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--participants", type=int, default=60)
    ap.add_argument("--frames-per-stage", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    lines = []
    for p in range(a.participants):
        person = Person(rng)
        pid = f"synth-p{p:04d}"
        variant = {
            "n_pleats": int(rng.choice([3, 4, 5, 5, 6, 6, 7, 7, 8, 10])),
            "pleat_quality": str(rng.choice(["neat", "neat", "uneven_spacing", "misaligned", "loose_or_unformed"])),
            "pleat_offset": float(rng.choice([0.1, 0.1, 0.0, 0.2, rng.uniform(-0.7, 0.8)])),
            "wrong_shoulder": bool(rng.random() < 0.2),
            "untucked": bool(rng.random() < 0.15),
            "hem_tilt": float(rng.choice([0.0, 0.0, 0.0, rng.uniform(-0.2, 0.2)])),
        }
        fi = 0
        for stage in DRAPE_STAGES:
            for _ in range(a.frames_per_stage):
                img, mask, k, pleat, corr, light = render(person, stage, variant, rng)
                sid = f"{pid}-f{fi:03d}"
                cv2.imwrite(str(out / "images" / f"{sid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                cv2.imwrite(str(out / "masks" / f"{sid}.png"), mask)
                lines.append({
                    "id": sid, "schema_version": SCHEMA_VERSION, "image": f"images/{sid}.jpg", "width": W, "height": H,
                    "video_id": pid, "frame_index": fi, "timestamp_ms": fi * 500.0, "participant_id": pid,
                    "source": {"type": "synthetic", "consent_id": None, "license": "generated", "collector": None,
                               "generator": f"{GEN_VERSION} seed={a.seed}"},
                    "capture": {"camera_angle": "front", "camera_height": "chest", "lighting": light if light != "colored" else "colored",
                                "body_position": "standing", "device": None},
                    "saree": {"primary_color": None, "darkness": person.saree["darkness"], "fabric": "synthetic",
                              "pattern": {"striped": "striped", "printed": "printed", "woven_border": "woven_border",
                                          "checked": "checked"}.get(person.saree["pattern"], "plain"),
                              "transparency": "opaque" if person.saree["alpha"] == 1.0 else "translucent", "drape_style": "nivi"},
                    "mask": f"masks/{sid}.png",
                    "landmarks": [[round(float(x), 2), round(float(y), 2), 2] for x, y, _ in k],
                    "regions": None,
                    "pleats": pleat if pleat else {"visible": False, "roi": None, "fold_lines": [], "count": 0,
                                                   "count_certain": True, "quality": None, "secured": None},
                    "step_label": stage,
                    "step_label_certain": not (variant["untucked"] and DRAPE_STAGES.index(stage) >= 4),
                    "correction_labels": corr, "annotator": GEN_VERSION, "review_status": "reviewed", "notes": None,
                })
                fi += 1
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    (out / "README.txt").write_text(
        f"SYNTHETIC DATA generated by {GEN_VERSION} (seed {a.seed}). Not real people. "
        "Never report metrics on this data as real-world accuracy.\n", encoding="utf-8")
    print(f"wrote {len(lines)} synthetic samples for {a.participants} synthetic participants to {out}")


if __name__ == "__main__":
    main()

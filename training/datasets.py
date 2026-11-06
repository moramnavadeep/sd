"""PyTorch datasets for each model. All read the same manifest schema."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from saree_ai.constants import SEG, SEG_IGNORE, FLIP_PAIRS, PLEAT_QUALITY, STAGE
from saree_ai.features import frame_features

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

_REGION_CLS = {"person": SEG["person"], "other_clothing": SEG["other_clothing"], "saree_body": SEG["saree_body"],
               "waist_wrap": SEG["waist_wrap"], "pleats": SEG["pleats"], "pallu": SEG["pallu"]}
_DRAW_ORDER = ["person", "other_clothing", "saree_body", "waist_wrap", "pleats", "pallu"]


def load_image(r) -> np.ndarray:
    img = cv2.imread(str(Path(r["_root"]) / r["image"]), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(r["image"])
    return img


def load_mask(r) -> np.ndarray | None:
    if r.get("mask"):
        m = cv2.imread(str(Path(r["_root"]) / r["mask"]), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(r["mask"])
        return m
    if r.get("regions"):
        m = np.full((r["height"], r["width"]), SEG["background"], np.uint8)
        for cls in _DRAW_ORDER:
            for reg in r["regions"]:
                if reg["class"] == cls:
                    cv2.fillPoly(m, [np.round(np.array(reg["polygon"])).astype(np.int32)], _REGION_CLS[cls])
        return m
    return None


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    x = img_bgr[:, :, ::-1].astype(np.float32) / 255
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy())


def photometric(img, rng):
    """Lighting / colour robustness: brightness, contrast, gamma, hue shift, noise, blur."""
    img = img.astype(np.float32)
    img = img * rng.uniform(0.6, 1.4) + rng.uniform(-25, 25)
    img = np.clip(img, 0, 255)
    img = 255 * (img / 255) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.5:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + rng.integers(-12, 13)) % 180
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    img += rng.normal(0, rng.uniform(0, 6), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


class SegDataset(Dataset):
    """Note on flips: our classes are left/right symmetric EXCEPT that the pallu should be on the
    LEFT shoulder. Horizontal flip is still valid for segmentation (pixels keep their class); it is
    the correction engine, not the segmenter, that judges which shoulder."""

    def __init__(self, rows, hw=(384, 256), augment=False, seed=0):
        self.rows = [r for r in rows if r.get("mask") or r.get("regions")]
        self.hw, self.augment = tuple(hw), augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img, mask = load_image(r), load_mask(r)
        h, w = self.hw
        if self.augment:
            rng = self.rng
            s = rng.uniform(0.8, 1.2)
            ang = rng.uniform(-8, 8)
            M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), ang, s)
            M[:, 2] += rng.uniform(-0.08, 0.08, 2) * [img.shape[1], img.shape[0]]
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderValue=SEG_IGNORE)
            if rng.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            img = photometric(np.ascontiguousarray(img), rng)
        img = cv2.resize(np.ascontiguousarray(img), (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(np.ascontiguousarray(mask), (w, h), interpolation=cv2.INTER_NEAREST)
        return to_tensor(img), torch.from_numpy(mask.astype(np.int64))


def pleat_crop_box(r, rng=None, jitter=0.0):
    p = r.get("pleats") or {}
    if p.get("visible") and p.get("roi"):
        x0, y0, x1, y1 = p["roi"]
    else:  # negative sample: crop from the same body area the runtime would use
        lm = np.array(r["landmarks"]) if r.get("landmarks") else None
        if lm is None:
            return None
        hc = (lm[11, :2] + lm[12, :2]) / 2
        hw = np.linalg.norm(lm[11, :2] - lm[12, :2]) + 1
        x0, x1 = hc[0] - 0.35 * hw, hc[0] + 0.35 * hw
        y0, y1 = hc[1], max(lm[13, 1], lm[14, 1])
    if rng is not None and jitter:
        bw, bh = x1 - x0, y1 - y0
        x0 += rng.uniform(-jitter, jitter) * bw; x1 += rng.uniform(-jitter, jitter) * bw
        y0 += rng.uniform(-jitter, jitter) * bh; y1 += rng.uniform(-jitter, jitter) * bh
    return [int(max(0, x0)), int(max(0, y0)), int(min(r["width"], x1)), int(min(r["height"], y1))]


class PleatDataset(Dataset):
    """Targets: count (float, -1 = unknown), quality (int, -100 = ignore), secured (float, -1 = n/a), visible (0/1)."""

    def __init__(self, rows, size=128, augment=False, seed=0):
        self.rows = [r for r in rows if r.get("pleats") is not None and pleat_crop_box(r) is not None]
        self.size, self.augment = size, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = load_image(r)
        x0, y0, x1, y1 = pleat_crop_box(r, self.rng if self.augment else None, 0.08)
        crop = img[y0:max(y1, y0 + 2), x0:max(x1, x0 + 2)]
        if self.augment:
            crop = photometric(crop, self.rng)
        crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
        p = r["pleats"]
        vis = bool(p.get("visible"))
        count = float(p["count"]) if vis and p.get("count") is not None and p.get("count_certain", True) else (-1.0 if vis else 0.0)
        q = PLEAT_QUALITY.index(p["quality"]) if vis and p.get("quality") else -100
        sec = float(p["secured"]) if vis and p.get("secured") is not None else -1.0
        return to_tensor(crop), torch.tensor([count, sec, float(vis)], dtype=torch.float32), torch.tensor(q)


def gt_features(r) -> np.ndarray | None:
    """Stage-model features computed from GROUND TRUTH annotations (teacher forcing).
    Pleat quality/secured from labels; at inference these come from the models instead."""
    mask = load_mask(r)
    if mask is None or not r.get("landmarks"):
        return None
    k = np.array(r["landmarks"], np.float32)
    k[:, 2] = (k[:, 2] > 0).astype(np.float32)
    p = r.get("pleats") or {}
    pf = None
    if p.get("visible"):
        qp = [0.0] * 4
        if p.get("quality"):
            qp[PLEAT_QUALITY.index(p["quality"])] = 1.0
        pf = {"count": p.get("count"), "quality_probs": qp, "secured_prob": None if p.get("secured") is None else float(p["secured"])}
    f, _ = frame_features(mask, k, pf)
    return f


class StageFeatureDataset(Dataset):
    def __init__(self, rows, cache: dict | None = None, noise=0.0, seed=0):
        self.items = []
        for r in rows:
            if r.get("step_label") is None or not r.get("step_label_certain", True):
                continue
            f = cache.get(r["id"]) if cache is not None and r["id"] in cache else gt_features(r)
            if f is None:
                continue
            if cache is not None:
                cache[r["id"]] = f
            self.items.append((f, STAGE[r["step_label"]]))
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, y = self.items[i]
        if self.noise:
            f = f + self.rng.normal(0, self.noise, f.shape).astype(np.float32)
        return torch.from_numpy(f.astype(np.float32)), torch.tensor(y)


class PoseDataset(Dataset):
    """Top-down crops around the annotated person; Gaussian heatmap targets at 1/4 resolution."""

    def __init__(self, rows, hw=(256, 192), sigma=2.0, augment=False, seed=0):
        self.rows = [r for r in rows if r.get("landmarks")]
        self.hw, self.sigma, self.augment = tuple(hw), sigma, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = load_image(r)
        k = np.array(r["landmarks"], np.float32)
        lab = k[k[:, 2] > 0]
        x0, y0 = lab[:, :2].min(0); x1, y1 = lab[:, :2].max(0)
        pw, ph = (x1 - x0) * 0.2 + 8, (y1 - y0) * 0.12 + 8
        if self.augment:
            pw *= self.rng.uniform(0.8, 1.3); ph *= self.rng.uniform(0.8, 1.3)
        box = np.array([x0 - pw, y0 - ph, x1 + pw, y1 + ph])
        h, w = self.hw
        src = np.float32([[box[0], box[1]], [box[2], box[1]], [box[0], box[3]]])
        dst = np.float32([[0, 0], [w, 0], [0, h]])
        M = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        kk = k.copy()
        kk[:, :2] = (np.c_[k[:, :2], np.ones(17)] @ M.T)
        if self.augment and self.rng.random() < 0.5:
            crop = crop[:, ::-1]; kk[:, 0] = w - 1 - kk[:, 0]
            for a, b in FLIP_PAIRS:
                kk[[a, b]] = kk[[b, a]]
        if self.augment:
            crop = photometric(np.ascontiguousarray(crop), self.rng)
        hh, ww = h // 4, w // 4
        yy, xx = np.mgrid[:hh, :ww]
        hm = np.zeros((17, hh, ww), np.float32)
        wt = np.zeros(17, np.float32)
        for j in range(17):
            if kk[j, 2] > 0:
                cx, cy = kk[j, 0] / 4, kk[j, 1] / 4
                hm[j] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * self.sigma ** 2))
                wt[j] = 1.0
        return to_tensor(np.ascontiguousarray(crop)), torch.from_numpy(hm), torch.from_numpy(wt), torch.from_numpy(kk)

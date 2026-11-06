"""Metric implementations (numpy only, so evaluation does not depend on training code)."""
from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred, n, ignore=None) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if ignore is not None:
        keep = y_true != ignore
        y_true, y_pred = y_true[keep], y_pred[keep]
    return np.bincount(n * y_true.astype(np.int64) + y_pred, minlength=n * n).reshape(n, n)


def per_class_from_confusion(cm: np.ndarray, names: list[str]) -> dict:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    support = cm.sum(1)
    out = {}
    for i, n in enumerate(names):
        p = tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] else None
        r = tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] else None
        f1 = 2 * p * r / (p + r) if p is not None and r is not None and (p + r) else (0.0 if p is not None or r is not None else None)
        iou = tp[i] / (tp[i] + fp[i] + fn[i]) if tp[i] + fp[i] + fn[i] else None
        out[n] = {"precision": p, "recall": r, "f1": f1, "iou": iou, "support": int(support[i])}
    return out


def macro(per_class: dict, key: str, skip=()) -> float | None:
    vals = [v[key] for k, v in per_class.items() if k not in skip and v[key] is not None and v["support"] > 0]
    return float(np.mean(vals)) if vals else None


def classification_report(y_true, y_pred, names) -> dict:
    cm = confusion(y_true, y_pred, len(names))
    pc = per_class_from_confusion(cm, names)
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else None
    return {"accuracy": acc, "macro_precision": macro(pc, "precision"), "macro_recall": macro(pc, "recall"),
            "macro_f1": macro(pc, "f1"), "per_class": pc, "confusion": cm.tolist(), "labels": names, "n": int(cm.sum())}


def binary_report(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum()); fp = int((~y_true & y_pred).sum()); fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if (p is not None and r is not None) else None)
    fpr = fp / (fp + tn) if fp + tn else None
    return {"precision": p, "recall": r, "f1": f1, "false_positive_rate": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def pck(pred: np.ndarray, gt: np.ndarray, vis: np.ndarray, norm: np.ndarray, thr=0.2) -> dict:
    """Percentage of Correct Keypoints. pred/gt (N,17,2); vis (N,17) bool; norm (N,) torso size."""
    d = np.linalg.norm(pred - gt, axis=-1) / norm[:, None]
    ok = (d <= thr) & vis
    per_kp = ok.sum(0) / np.maximum(vis.sum(0), 1)
    return {"pck": float(ok.sum() / max(vis.sum(), 1)), "per_keypoint": per_kp.tolist(), "threshold": thr}

"""Model D runtime: draping-stage recognition with temporal filtering."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import DRAPE_STAGES, STAGE
from ..features import FEATURE_NAMES
from ..registry import ROOT, active, loadable

FI = {n: i for i, n in enumerate(FEATURE_NAMES)}


@dataclass
class StageResult:
    available: bool
    stage: str | None = None
    confidence: float = 0.0
    probs: list | None = None          # per-frame
    filtered: list | None = None       # after temporal filter
    source: str = "none"               # model | heuristic | none
    experimental: bool = True
    reason: str = ""


def heuristic_stage_probs(f: np.ndarray) -> np.ndarray:
    """Interpretable fallback used when no trained stage model is available.
    Produces a soft distribution whose max is capped at 0.55 so it can never by itself pass
    the stage-confidence threshold for corrections (0.6 by default)."""
    p = np.full(len(DRAPE_STAGES), 0.02, np.float32)
    saree = f[FI["frac_saree_body"]] + f[FI["frac_waist_wrap"]] + f[FI["frac_pleats"]] + f[FI["frac_pallu"]]
    if saree < 0.03:
        p[STAGE["not_started"]] = 1
    elif f[FI["frac_pallu"]] > 0.03:
        p[STAGE["pallu_complete" if f[FI["pallu_left_shoulder"]] > 0.15 else "pallu_in_progress"]] = 1
    elif f[FI["pleat_present"]] > 0:
        tucked = f[FI["pleat_secured"]] > 0.5 or abs(f[FI["pleat_top_rel"]]) < 0.25
        p[STAGE["pleats_secured" if tucked else "pleats_forming"]] = 1
    elif f[FI["waist_coverage"]] < 0.5:
        p[STAGE["wrapping_started"]] = 1
    else:
        p[STAGE["waist_wrap_complete"]] = 1
    p = p / p.sum()
    top = p.argmax()
    if p[top] > 0.55:  # cap
        rest = (1 - 0.55) / (len(p) - 1)
        p[:] = rest; p[top] = 0.55
    return p


def transition_matrix(n=len(DRAPE_STAGES), stay=0.90, fwd=0.07, back=0.02) -> np.ndarray:
    """Tutorial-order prior: mostly stay, sometimes advance one stage, rarely go back."""
    T = np.full((n, n), 1e-3)
    for i in range(n):
        T[i, i] = stay
        if i + 1 < n: T[i, i + 1] = fwd
        if i - 1 >= 0: T[i, i - 1] = back
    return T / T.sum(1, keepdims=True)


class StageRecognizer:
    def __init__(self, cfg: dict, providers=None):
        self.entry = active("stage")
        ok, why = loadable(self.entry, cfg.get("min_status", "experimental"))
        self.model_ok, self.reason = ok, why
        if ok:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(ROOT / self.entry["file"]), providers=providers or ["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
            self.experimental = self.entry.get("status") != "validated"
            norm = self.entry.get("feature_norm") or {}
            self.mu = np.array(norm.get("mean", [0.0] * len(FEATURE_NAMES)), np.float32)
            self.sd = np.array(norm.get("std", [1.0] * len(FEATURE_NAMES)), np.float32)
        self.T = transition_matrix()
        self.temper = float(cfg.get("likelihood_temper", 0.7))
        self.reset()

    def reset(self, start_stage: int | None = None):
        n = len(DRAPE_STAGES)
        self.belief = np.full(n, 1.0 / n)
        if start_stage is not None:
            self.belief = np.full(n, 0.02); self.belief[start_stage] = 1; self.belief /= self.belief.sum()

    def __call__(self, feats: np.ndarray | None, seg_available: bool) -> StageResult:
        if feats is None or not seg_available:
            return StageResult(False, reason="needs saree segmentation, which is unavailable")
        if self.model_ok:
            x = ((feats - self.mu) / self.sd).astype(np.float32)[None]
            logits = self.sess.run(None, {self.inp: x})[0][0]
            e = np.exp(logits - logits.max()); p = e / e.sum(); src, exp = "model", self.experimental
        else:
            p, src, exp = heuristic_stage_probs(feats), "heuristic", True
        lik = np.power(p + 1e-4, self.temper)
        b = (self.belief @ self.T) * lik
        self.belief = b / b.sum()
        i = int(self.belief.argmax())
        conf = float(min(self.belief[i], p[i] if src == "heuristic" else 1.0))
        return StageResult(True, DRAPE_STAGES[i], conf, p.tolist(), self.belief.tolist(), src, exp,
                           "ok" if self.model_ok else f"heuristic fallback ({self.reason})")

"""Model E: interpretable, rule-based correction engine.

Inputs : FrameState (pose, segmentation, pleats, stage, features) for the current frame.
Output : a structured guidance message for the frontend + optional speech.

Trust rules enforced here (requirement 14):
  * Positioning problems (pose below threshold) suppress ALL draping feedback.
  * A check that is 'uncertain' never becomes a correction; the user is offered manual confirm.
  * A failing check must persist for persist_ratio of the last `window` frames.
  * If a failing check depends on an EXPERIMENTAL model and allow_experimental_corrections is
    false, it is reported as 'unverified', not as a correction.
  * The engine never auto-advances the tutorial; it only suggests "say next".

Upgrade path: once enough frames carry `correction_labels`, training/evaluate_corrections.py
measures this engine's precision / false-positive rate; a learned model can then replace
individual checks (same Outcome interface).
"""
from __future__ import annotations

import time
from collections import deque, defaultdict
from pathlib import Path

import yaml

from ..constants import STAGE
from ..i18n import T, render, DEFAULT_LANG, LANGS
from .checks import CHECKS, FrameState, Outcome, full_body_visible
from .guides import build_guides


class Tutorial:
    def __init__(self, path: str | Path):
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.id, self.name, self.steps = d["id"], d["name"], d["steps"]
        self.explanations = d.get("explanations", {})
        # Per-language overlays: <name>.<lang>.yaml with the same step ids (title/instruction/speech) + explanations.
        self.overlays = {}
        p = Path(path)
        for lang in LANGS:
            op = p.with_name(f"{p.stem}.{lang}{p.suffix}")
            if lang != "en" and op.exists():
                o = yaml.safe_load(op.read_text(encoding="utf-8"))
                self.overlays[lang] = {"steps": {x["id"]: x for x in o.get("steps", [])},
                                       "explanations": o.get("explanations", {})}
        for s in self.steps:
            s["_checks"] = [(c, {}) if isinstance(c, str) else (c["id"], {k: v for k, v in c.items() if k != "id"})
                            for c in s.get("checks", [])]
            for cid, _ in s["_checks"]:
                if cid not in CHECKS:
                    raise ValueError(f"tutorial step {s['id']} references unknown check {cid}")

    def text(self, step: dict, field: str, lang: str) -> str:
        """title / instruction / speech in `lang`, falling back to English."""
        ov = self.overlays.get(lang, {}).get("steps", {}).get(step["id"], {})
        return ov.get(field) or step[field]

    def explanation(self, key: str, lang: str) -> str:
        return self.overlays.get(lang, {}).get("explanations", {}).get(key) or self.explanations.get(key, "")


class CorrectionEngine:
    def __init__(self, tutorial: Tutorial, thresholds: dict, ccfg: dict, scfg: dict):
        self.t = tutorial
        self.th = thresholds
        self.window = int(ccfg.get("window", 12))
        self.persist = float(ccfg.get("persist_ratio", 0.7))
        self.allow_exp = bool(ccfg.get("allow_experimental_corrections", False))
        self.min_interval = float(scfg.get("min_interval_s", 6))
        self.repeat_suppress = float(scfg.get("repeat_suppress_s", 20))
        self.step_index = 0
        self.lang = DEFAULT_LANG
        self.manual_confirmed: dict[str, set] = defaultdict(set)
        self.step_confirmed: set[str] = set()
        self._reset_history()
        self._last_speech_t = 0.0
        self._spoken: dict[str, float] = {}

    # ------------------------------------------------------------------ navigation
    def _reset_history(self):
        self.hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self.stage_hist: deque = deque(maxlen=self.window)

    @property
    def step(self):
        return self.t.steps[self.step_index]

    def goto(self, idx: int):
        self.step_index = max(0, min(len(self.t.steps) - 1, idx))
        self._reset_history()
        return self.step

    def next(self):
        return self.goto(self.step_index + 1)

    def previous(self):
        return self.goto(self.step_index - 1)

    def restart(self):
        self.manual_confirmed.clear()
        self.step_confirmed.clear()
        return self.goto(0)

    def confirm_manual(self, check_id: str | None = None):
        """User says 'this looks right' for a check (or the whole step) the camera can't verify."""
        sid = self.step["id"]
        if check_id is None:
            self.step_confirmed.add(sid)
            self.manual_confirmed[sid] |= {c for c, _ in self.step["_checks"] if c != "full_body_visible"}
        else:
            self.manual_confirmed[sid].add(check_id)

    def set_language(self, lang: str):
        if lang in LANGS:
            self.lang = lang
            self._spoken.clear()

    def r(self, msg) -> str:
        """Render a message template in the session language."""
        return render(msg, self.lang)

    def say(self, field: str, step: dict | None = None) -> str:
        return self.t.text(step or self.step, field, self.lang)

    # ------------------------------------------------------------------ speech gating
    def _maybe_speak(self, text: str | None, force=False):
        if not text:
            return None
        now = time.monotonic()
        if not force:
            if now - self._last_speech_t < self.min_interval:
                return None
            if now - self._spoken.get(text, -1e9) < self.repeat_suppress:
                return None
        self._last_speech_t = now
        self._spoken[text] = now
        return text

    def instruction_speech(self):
        return self._maybe_speak(self.say("speech"), force=True)

    # ------------------------------------------------------------------ main
    def evaluate(self, s: FrameState) -> dict:
        step = self.step
        sid = step["id"]
        outcomes: list[Outcome] = []

        # 1) Positioning gate.
        body = full_body_visible(s, {}, self.th)
        pose_ok = body.status == "pass"
        self.hist["full_body_visible"].append(body.status)

        if not pose_ok:
            guides = build_guides(["frame_box"], s, self.th, focus="frame_box")
            return self._msg(step, "need_position", self.r(body.message), [body], [], guides,
                             speech=self._maybe_speak(self.r(body.message)), experimental=False)

        # 2) Step checks.
        for cid, params in step["_checks"]:
            if cid == "full_body_visible":
                outcomes.append(body); continue
            if cid in self.manual_confirmed[sid]:
                outcomes.append(Outcome(cid, "pass", T("Confirmed by you."), confidence=1.0)); continue
            o = CHECKS[cid](s, params, self.th)
            if o.status == "fail" and o.experimental and not self.allow_exp:
                o = Outcome(cid, "unverified", o.message, o.label, o.confidence, True, o.needs, o.focus)
            outcomes.append(o)
            self.hist[cid].append(o.status)

        # 3) Persistence filter: only stable failures become corrections.
        def stable(cid, status):
            h = self.hist[cid]
            return len(h) >= max(3, self.window // 2) and sum(x == status for x in h) / len(h) >= self.persist

        corrections = [o for o in outcomes if o.status == "fail" and o.check != "full_body_visible" and stable(o.check, "fail")]
        passed = [o for o in outcomes if o.status == "pass"]
        uncertain = [o for o in outcomes if o.status in ("uncertain", "unverified")]

        # 4) Stage.
        stage_info = None
        if s.stage.available:
            self.stage_hist.append(s.stage.stage if s.stage.confidence >= self.th["stage"] else None)
            stage_info = {"stage": s.stage.stage, "confidence": s.stage.confidence, "source": s.stage.source,
                          "experimental": s.stage.experimental, "trusted": s.stage.confidence >= self.th["stage"]}
        done_stage = step.get("done_stage")
        stage_done = bool(done_stage) and len(self.stage_hist) >= self.window // 2 and \
            sum(1 for x in self.stage_hist if x and STAGE[x] >= STAGE[done_stage]) / max(1, len(self.stage_hist)) >= self.persist
        all_checks_pass = all(o.status in ("pass", "n/a") for o in outcomes)

        focus = corrections[0].focus if corrections else None
        guides = build_guides(step.get("guides", []), s, self.th, focus)
        experimental = any(o.experimental for o in outcomes) or bool(stage_info and stage_info["experimental"])

        if corrections:
            c = corrections[0]
            return self._msg(step, "correction", self.r(c.message), outcomes, [c.label for c in corrections if c.label], guides,
                             self._maybe_speak(self.r(c.message)), experimental, stage_info)
        if all_checks_pass and (stage_done or not done_stage or sid in self.step_confirmed):
            msg = self.r(T("This step looks done. Say \"next\" when you're ready."))
            return self._msg(step, "step_complete", msg, outcomes, [], guides, self._maybe_speak(msg), experimental, stage_info)
        if uncertain:
            msg = self.r(self._check_text(uncertain[0])) + " " + self.r(T("You can confirm this step manually."))
            return self._msg(step, "uncertain", msg, outcomes, [], guides, None, experimental, stage_info)
        return self._msg(step, "ok", self.say("instruction"), outcomes, [], guides, None, experimental, stage_info)

    def _msg(self, step, status, headline, outcomes, labels, guides, speech, experimental, stage_info=None):
        return {
            "step": {"id": step["id"], "title": self.say("title", step), "instruction": self.say("instruction", step),
                     "index": self.step_index, "total": len(self.t.steps)},
            "lang": self.lang,
            "status": status,
            "headline": headline,
            "checks": [{"id": o.check, "status": o.status, "message": self.r(self._check_text(o)), "label": o.label,
                        "confidence": round(o.confidence, 3), "experimental": o.experimental, "needs": o.needs}
                       for o in outcomes],
            "corrections": labels,
            "guides": guides,
            "speech": speech,
            "stage": stage_info,
            "experimental": experimental,
            "manual_confirm_available": True,
        }

    @staticmethod
    def _check_text(o: Outcome):
        # Unverified = a failing check that relied on an experimental model: never a correction.
        return T("(experimental model) {inner}", inner=o.message) if o.status == "unverified" else o.message

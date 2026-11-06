"""Local speech recognition (Vosk, Apache-2.0). Audio never leaves the machine.

The browser streams 16 kHz mono PCM16 over the loopback WebSocket. One grammar-constrained Vosk
recogniser per language turns it into a known command phrase (or [unk]). In Telugu mode the English
recogniser runs in parallel on the same audio, because people mix English commands into Telugu.

Note: grammars MUST be serialised with ensure_ascii=False - with the default, Telugu words become
"\\u0c24..." escapes that Vosk treats as unknown words (found and fixed 2026-11-06).
"""
from __future__ import annotations

import json

from ..registry import ROOT
from .intents import grammar_phrases, parse


class VoskCommandRecognizer:
    SAMPLE_RATE = 16000

    def __init__(self, model_dir: str, lang: str):
        path = ROOT / model_dir
        if not path.exists():
            raise FileNotFoundError(f"Vosk model not found at {path}. Run `python scripts/download_models.py`.")
        import vosk
        vosk.SetLogLevel(-1)
        self.lang = lang
        self.model = vosk.Model(str(path))
        self.grammar = json.dumps(grammar_phrases(lang), ensure_ascii=False)

    def recognizer(self):
        import vosk
        rec = vosk.KaldiRecognizer(self.model, self.SAMPLE_RATE, self.grammar)
        rec.SetWords(True)
        return rec


class CommandStream:
    """Feeds the same audio to one recogniser per active language and arbitrates between them.

    Rules (from a failure found in testing: Telugu "మళ్లీ మొదలు పెట్టు" (start again) was heard by the
    parallel English recogniser as "repeat do", fuzzy-matched to *repeat*, and won because it finished first):
      * a secondary-language (English) result counts only if it matches a command phrase EXACTLY;
      * a secondary result is held for HOLD_CHUNKS of further audio, so the session-language
        recogniser can finish the same utterance and take priority.
    """
    HOLD_CHUNKS = 6   # ~0.6 s at 100 ms chunks

    def __init__(self, recognizers: list[VoskCommandRecognizer], session_lang: str):
        self.recs = [(r.lang, r.recognizer()) for r in recognizers]
        self.session_lang = session_lang
        self.pending = None   # (chunks_left, candidate)

    @staticmethod
    def _result(rec_json: str):
        r = json.loads(rec_json)
        text = r.get("text", "").replace("[unk]", "").strip()
        if not text:
            return None
        conf = min((w.get("conf", 0.0) for w in r.get("result", [])), default=0.0)
        return text, round(float(conf), 3)

    def feed(self, pcm16: bytes) -> dict | None:
        primary, secondary = None, None
        for lang, rec in self.recs:
            if not rec.AcceptWaveform(pcm16):
                continue
            got = self._result(rec.Result())
            if not got:
                continue
            p = parse(got[0], self.session_lang)
            cand = {"text": got[0], "confidence": got[1], "asr_lang": lang, "intent": p["intent"], "score": p["score"]}
            if lang == self.session_lang:
                primary = cand
            elif p["intent"] is not None and p["score"] == 1.0:
                secondary = cand
        if primary is not None and (primary["intent"] is not None or self.pending is None):
            self.pending = None
            return primary
        if secondary is not None and self.pending is None:
            if len(self.recs) == 1:
                return secondary
            self.pending = [self.HOLD_CHUNKS, secondary]
            return primary   # may be a non-command transcript (shown to the user), or None
        if self.pending is not None:
            self.pending[0] -= 1
            if self.pending[0] <= 0:
                cand, self.pending = self.pending[1], None
                return cand
        return primary


class ASR:
    def __init__(self, cfg: dict):
        self.recognizers: dict[str, VoskCommandRecognizer] = {}
        self.status: dict[str, str] = {}
        for lang, d in (cfg.get("vosk_models") or {"en": cfg.get("vosk_model_dir")}).items():
            try:
                self.recognizers[lang] = VoskCommandRecognizer(d, lang)
                self.status[lang] = "vosk (local)"
            except Exception as e:  # noqa: BLE001
                self.status[lang] = f"unavailable: {e}"

    def stream(self, session_lang: str) -> CommandStream | None:
        langs = [session_lang] + (["en"] if session_lang != "en" else [])
        recs = [self.recognizers[lg] for lg in langs if lg in self.recognizers]
        return CommandStream(recs, session_lang) if recs else None


def build_asr(cfg: dict):
    if cfg.get("asr_backend", "vosk") != "vosk":
        return None, "speech recognition disabled in config"
    asr = ASR(cfg)
    if not asr.recognizers:
        return None, "; ".join(f"{k}: {v}" for k, v in asr.status.items())
    return asr, ", ".join(f"{k}: {v}" for k, v in asr.status.items())

"""Local text-to-speech for languages the OS has no voice for (Telugu).

English uses the browser's speechSynthesis restricted to OS-installed (localService) voices.
Windows ships no Telugu voice, so Telugu is synthesised here with Piper (piper-tts, GPL-3.0-or-later)
and the te_IN-maya-medium voice, on CPU via ONNX Runtime. The voice's training data (IIT Madras
IndicTTS) has a licence we could not read - see docs/DEPENDENCIES.md before any distribution.
"""
from __future__ import annotations

import io
import threading
import wave
from collections import OrderedDict

from ..registry import ROOT


class PiperTTS:
    def __init__(self, model_path: str, max_cache: int = 64):
        from piper import PiperVoice
        p = ROOT / model_path
        if not p.exists():
            raise FileNotFoundError(f"Piper voice not found at {p}. Run `python scripts/download_models.py`.")
        self.voice = PiperVoice.load(str(p))   # piper's default: use_cuda=False -> CPUExecutionProvider only
        self.lock = threading.Lock()
        self.cache: OrderedDict[str, bytes] = OrderedDict()
        self.max_cache = max_cache

    def wav(self, text: str) -> bytes:
        text = text.strip()[:600]
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]
        buf = io.BytesIO()
        with self.lock, wave.open(buf, "wb") as w:
            self.voice.synthesize_wav(text, w)
        data = buf.getvalue()
        self.cache[text] = data
        if len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        return data


def build_tts(cfg: dict) -> tuple[dict, dict]:
    engines, status = {}, {}
    for lang, path in (cfg.get("piper_voices") or {}).items():
        try:
            engines[lang] = PiperTTS(path)
            status[lang] = "piper (local)"
        except Exception as e:  # noqa: BLE001
            status[lang] = f"unavailable: {e}"
    return engines, status

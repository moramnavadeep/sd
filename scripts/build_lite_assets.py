"""Build the static assets for the FREE online demo (lite mode).

1. Packs the English Vosk model as a model.tar.gz archive for vosk-browser (recognition in the browser):
     frontend/asr/en.tar.gz   (Telugu is recognised on the server in lite mode; see backend/server.py)
2. Pre-records every Telugu sentence the tutorial engine can speak WITHOUT the camera, using the local
   Piper voice:  frontend/tts/te/<sha1 of text>.wav
   The sentence list is not hand-written: it is collected by running every voice command on every
   step through the real engine (saree_ai.voice.intents.respond), so audio and text always match.

  python scripts/build_lite_assets.py
"""
from __future__ import annotations

import hashlib
import sys
import tarfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saree_ai.correction.engine import CorrectionEngine, Tutorial  # noqa: E402
from saree_ai.voice.intents import INTENTS, respond  # noqa: E402


BROWSER_ASR_LANGS = ["en"]   # Telugu is recognised on the server: the 2022 WASM engine can't decode the te model


def pack_models(cfg):
    out = ROOT / "frontend" / "asr"
    out.mkdir(parents=True, exist_ok=True)
    for lang, d in cfg["voice"]["vosk_models"].items():
        if lang not in BROWSER_ASR_LANGS:
            continue
        src = ROOT / d
        if not src.exists():
            raise SystemExit(f"missing {src}: run scripts/download_models.py first")
        dst = out / f"{lang}.tar.gz"
        with tarfile.open(dst, "w:gz") as t:
            t.add(src, arcname="model")
        print(f"asr {lang}: {dst.stat().st_size // 1024} KB")


def speakable_sentences(cfg, lang: str) -> set[str]:
    tut = Tutorial(ROOT / cfg["tutorial"])
    eng = CorrectionEngine(tut, cfg["thresholds"], cfg["corrections"], {"min_interval_s": 0, "repeat_suppress_s": 0})
    eng.set_language(lang)
    texts = set()
    for i in range(len(tut.steps)):
        eng.goto(i)
        texts.add(eng.say("speech"))                       # hello / language switch / repeat
        for intent in list(INTENTS["en"]) + [None]:
            eng.goto(i)
            r = respond(intent, eng, None)
            if r.get("speech"):
                texts.add(r["speech"])
    return {t.strip() for t in texts if t and t.strip()}


def render_tts(cfg, lang: str):
    from saree_ai.voice.tts import PiperTTS
    tts = PiperTTS(cfg["voice"]["piper_voices"][lang])
    out = ROOT / "frontend" / "tts" / lang
    out.mkdir(parents=True, exist_ok=True)
    texts = speakable_sentences(cfg, lang)
    total = 0
    for t in sorted(texts):
        f = out / f"{hashlib.sha1(t.encode('utf-8')).hexdigest()}.wav"
        if not f.exists():
            f.write_bytes(tts.wav(t))
        total += f.stat().st_size
    print(f"tts {lang}: {len(texts)} sentences, {total // 1024} KB")


def main():
    cfg = yaml.safe_load((ROOT / "config" / "app.yaml").read_text(encoding="utf-8"))
    pack_models(cfg)
    for lang in cfg["voice"].get("piper_voices", {}):
        render_tts(cfg, lang)


if __name__ == "__main__":
    main()

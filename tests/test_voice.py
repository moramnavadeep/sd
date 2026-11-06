"""Voice + language tests (English and Telugu). Everything runs locally."""
import ast
import io
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saree_ai.i18n import T, render, catalog
from saree_ai.voice.intents import INTENTS, parse, grammar_phrases, normalise

CFG = yaml.safe_load((ROOT / "config/app.yaml").read_text(encoding="utf-8"))
TE_ASR = ROOT / CFG["voice"]["vosk_models"]["te"]
TE_TTS = ROOT / CFG["voice"]["piper_voices"]["te"]


def code_templates():
    """Every T("...") template in saree_ai/, including implicitly concatenated literals."""
    out = set()
    pat = re.compile(r'\bT\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)')
    for p in (ROOT / "saree_ai").rglob("*.py"):
        if p.name == "i18n.py":
            continue
        for m in pat.finditer(p.read_text(encoding="utf-8")):
            out.add(ast.literal_eval("(" + m.group(1) + ")"))
    return out


def test_every_message_has_a_telugu_translation():
    tmpl = code_templates()
    assert len(tmpl) > 40
    missing = sorted(t for t in tmpl if t not in catalog("te")["messages"])
    assert not missing, missing


def test_telugu_templates_keep_their_placeholders():
    for en, te in catalog("te")["messages"].items():
        assert set(re.findall(r"\{(\w+)\}", en)) == set(re.findall(r"\{(\w+)\}", te)), en


def test_render_translates_params_and_nesting():
    m = T("The pleats are too far to your {side}. Move them to just left of your navel.", side="right")
    assert "కుడి" in render(m, "te") and "right" in render(m, "en")
    nested = T("(experimental model) {inner}", inner=T("Hem looks level."))
    assert render(nested, "te") == "(ప్రయోగాత్మక మోడల్) అంచు సమానంగా ఉంది."


def test_tutorial_has_telugu_for_every_step():
    en = yaml.safe_load((ROOT / "config/tutorials/nivi.yaml").read_text(encoding="utf-8"))
    te = yaml.safe_load((ROOT / "config/tutorials/nivi.te.yaml").read_text(encoding="utf-8"))
    te_ids = {s["id"]: s for s in te["steps"]}
    for s in en["steps"]:
        assert s["id"] in te_ids and all(te_ids[s["id"]].get(f) for f in ("title", "instruction", "speech")), s["id"]
    assert set(en["explanations"]) <= set(te["explanations"])


@pytest.mark.parametrize("text,lang,intent", [
    ("తరువాత", "te", "next"), ("తర్వాతి దశ", "te", "next"), ("వెనక్కి", "te", "previous"),
    ("మళ్ళీ చెప్పండి", "te", "repeat"),                     # ళ్ళ / ళ్ల spelling variants
    ("కుచ్చిళ్లు ఎలా చేయాలి", "te", "explain_pleats"), ("పైట ఎలా వేయాలి", "te", "show_pallu"),
    ("మళ్లీ మొదలు పెట్టు", "te", "restart"), ("నేను సరిగ్గా చేస్తున్నానా", "te", "check"),
    ("next step", "te", "next"),                            # English still works in Telugu mode
    ("next next", "en", "next"), ("తరువాత తరువాత", "te", "next"),   # repeated word (seen in a real test)
    ("next", "en", "next"), ("తరువాత", "en", None),          # but Telugu is not active in English mode
])
def test_parse_multilingual(text, lang, intent):
    assert parse(text, lang)["intent"] == intent


def test_normalise_keeps_telugu():
    assert normalise("తరువాత!") == "తరువాత"


def test_telugu_grammar_json_is_not_ascii_escaped():
    from saree_ai.voice import asr
    src = Path(asr.__file__).read_text(encoding="utf-8")
    assert "ensure_ascii=False" in src
    assert all("\\u" not in p for p in grammar_phrases("te"))


def _pcm16_16k(wav_bytes):
    w = wave.open(io.BytesIO(wav_bytes))
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    sr = w.getframerate()
    idx = np.arange(0, len(x), sr / 16000)
    return np.interp(idx, np.arange(len(x)), x).astype(np.int16).tobytes()


@pytest.mark.skipif(not (TE_ASR.exists() and TE_TTS.exists()), reason="Telugu models not downloaded")
def test_telugu_commands_end_to_end_with_local_tts_and_asr():
    """Synthetic loop: Piper (local) speaks each command -> Vosk Telugu (local) -> intent.
    This proves the plumbing; it is NOT evidence of accuracy on real human voices."""
    from saree_ai.voice.asr import ASR
    from saree_ai.voice.tts import PiperTTS
    tts = PiperTTS(CFG["voice"]["piper_voices"]["te"])
    asr = ASR(CFG["voice"])
    phrases = {"next": "తరువాతి దశ", "previous": "వెనక్కి", "repeat": "మళ్లీ చెప్పండి",
               "explain_pleats": "కుచ్చిళ్లు ఎలా చేయాలి", "show_pallu": "పైట ఎలా వేయాలి",
               "restart": "మళ్లీ మొదలు పెట్టు", "check": "నేను సరిగ్గా చేస్తున్నానా"}
    got = {}
    for intent, text in phrases.items():
        st = asr.stream("te")
        pcm = _pcm16_16k(tts.wav(text)) + b"\x00\x00" * 16000   # 1 s of trailing silence ends the utterance
        res = None
        for i in range(0, len(pcm), 3200):
            res = st.feed(pcm[i:i + 3200]) or res
        got[intent] = res and res["intent"]
    assert got == {k: k for k in phrases}, got


@pytest.mark.skipif(not TE_TTS.exists(), reason="Telugu TTS not downloaded")
def test_server_language_switch_tts_and_voice_next():
    from fastapi.testclient import TestClient
    from backend.server import create_app, load_config
    c = TestClient(create_app(load_config(ROOT / "config/app.yaml", False)))
    st = c.get("/api/status").json()
    assert st["languages"]["te"] == "తెలుగు" and st["tts_server"]["te"].startswith("piper")
    r = c.post("/api/tts", json={"text": "తరువాతి దశ", "lang": "te"}, headers={"origin": "http://127.0.0.1:8765"})
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    assert c.post("/api/tts", json={"text": "x", "lang": "te"}, headers={"origin": "https://evil.example"}).status_code == 403
    # No camera frames at all: voice "next" must still move the tutorial (the reported bug).
    with c.websocket_connect("/ws/session?lang=te", headers={"origin": "http://127.0.0.1:8765"}) as ws:
        hello = json.loads(ws.receive_text())
        assert hello["lang"] == "te" and hello["step"]["title"] == "సిద్ధం అవ్వండి"
        ws.send_text(json.dumps({"type": "utterance", "text": "తరువాత", "intent": "next"}))
        v = json.loads(ws.receive_text())
        assert v["intent"] == "next" and v["step"]["index"] == 1 and v["speech"].startswith("దశ 2.")
        ws.send_text(json.dumps({"type": "set_language", "lang": "en"}))
        v = json.loads(ws.receive_text())
        assert v["lang"] == "en" and v["step"]["title"] == "Tuck the plain end"
        ws.send_text(json.dumps({"type": "utterance", "text": "next step"}))
        assert json.loads(ws.receive_text())["step"]["index"] == 2

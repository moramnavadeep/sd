"""Free-hosting (lite) mode: no vision models or torch, honest privacy text, Telugu recognised on the server,
pre-recorded Telugu replies match exactly what the engine says."""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from backend.server import create_app, load_config
    app = create_app(load_config(ROOT / "config/app.yaml", False, hosted=True, lite=True))
    return TestClient(app, base_url="https://sd.onrender.com")


def test_lite_status_and_privacy(client):
    st = client.get("/api/status").json()
    assert st["lite"] and st["hosted"]
    assert all(not m["available"] for m in st["models"].values())
    assert "camera picture never leaves your device" in st["privacy"]
    assert "Telugu voice commands are recognised on this app's server" in st["privacy"]
    assert st["asr"]["languages"]["en"] == "browser"
    assert st["asr"]["server_languages"] == (["te"] if (ROOT / "models/asr/vosk-model-small-te-0.42").exists() else [])


def test_health_endpoints(client):
    assert client.get("/healthz").json() == {"ok": True}
    assert client.get("/health").json() == {"ok": True}      # Render's default path (seen in the Render log)


def test_lite_grammar_endpoint(client):
    g = client.get("/api/grammar?lang=te").json()
    assert "తరువాత" in g["phrases"] and g["phrases"][-1] == "[unk]"


def test_lite_browser_utterance_moves_step_and_ignores_noise(client):
    h = {"origin": "https://sd.onrender.com", "host": "sd.onrender.com"}
    with client.websocket_connect("/ws/session?lang=en", headers=h) as ws:
        assert json.loads(ws.receive_text())["type"] == "hello"
        ws.send_text(json.dumps({"type": "utterance", "text": "yes yes"}))        # not a command: no speech, no step change
        assert json.loads(ws.receive_text()) == {"type": "heard", "text": "yes yes", "intent": None}
        ws.send_text(json.dumps({"type": "utterance", "text": "next step"}))      # browser recogniser sends text only
        v = json.loads(ws.receive_text())
        assert v["intent"] == "next" and v["step"]["index"] == 1


def test_lite_does_not_import_torch():
    import subprocess
    code = ("import sys; sys.path.insert(0, r'%s'); from backend.server import create_app, load_config; "
            "from pathlib import Path; create_app(load_config(Path(r'%s'), False, hosted=True, lite=True)); "
            "print('torch' in sys.modules)") % (ROOT, ROOT / "config/app.yaml")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().splitlines()[-1] == "False", out.stderr[-500:]


def test_prerecorded_telugu_covers_every_engine_sentence():
    tts_dir = ROOT / "frontend/tts/te"
    if not tts_dir.exists():
        pytest.skip("run scripts/build_lite_assets.py first")
    from build_lite_assets import speakable_sentences
    cfg = yaml.safe_load((ROOT / "config/app.yaml").read_text(encoding="utf-8"))
    missing = [t for t in speakable_sentences(cfg, "te")
               if not (tts_dir / f"{hashlib.sha1(t.encode('utf-8')).hexdigest()}.wav").exists()]
    assert not missing, missing[:3]

"""Hosted (Render) mode: same-origin only, honest privacy text, session cap."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import os
    os.environ["MAX_SESSIONS"] = "1"
    from backend.server import create_app, load_config
    app = create_app(load_config(ROOT / "config/app.yaml", False, hosted=True))
    os.environ.pop("MAX_SESSIONS")
    return TestClient(app, base_url="https://saree-mirror.onrender.com")


def test_hosted_status_discloses_server_processing(client):
    st = client.get("/api/status").json()
    assert st["hosted"] is True
    assert "sent over an encrypted connection to this app's own server" in st["privacy"]
    assert "on this computer" not in st["privacy"]
    assert client.get("/healthz").json() == {"ok": True}


def test_hosted_accepts_same_origin_rejects_others(client):
    good = {"origin": "https://saree-mirror.onrender.com", "host": "saree-mirror.onrender.com"}
    with client.websocket_connect("/ws/session", headers=good) as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello" and "own server" in hello["privacy"]
        # session cap = 1: a second user is told the demo is busy
        with client.websocket_connect("/ws/session", headers=good) as ws2:
            assert json.loads(ws2.receive_text())["type"] == "busy"
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/session", headers={"origin": "https://evil.example", "host": "saree-mirror.onrender.com"}) as ws:
            ws.receive_text()
    r = client.post("/api/tts", json={"text": "x", "lang": "te"}, headers={"origin": "https://evil.example"})
    assert r.status_code == 403


def test_local_mode_still_refuses_public_bind():
    src = (ROOT / "backend/server.py").read_text(encoding="utf-8")
    assert "Refusing to bind to a non-loopback address in local mode" in src

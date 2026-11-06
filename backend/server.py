"""Local backend: serves the mirror UI and runs all inference on this machine.

Run:  python -m backend.server            (http://127.0.0.1:8765)  - LOCAL mode (default)
      python -m backend.server --allow-synthetic-models   (dev only)
      python -m backend.server --hosted     - HOSTED mode (e.g. Render), see below

LOCAL mode privacy: binds to loopback only, rejects WebSocket connections from non-local origins,
makes no outbound network calls, and never writes camera frames to disk.

HOSTED mode (opt-in, chosen by the project owner on 2026-11-06 to get a public link): binds 0.0.0.0
on $PORT, accepts same-origin connections only, and the UI states plainly that camera frames and
audio are sent to this app's server. That is cloud processing, which the local-first design avoids:
frames are still processed in memory, never stored, and never sent to any third-party AI service.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import struct
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saree_ai import hardware  # noqa: E402
from saree_ai.inference.pipeline import Models, Session  # noqa: E402
from saree_ai.voice.asr import build_asr  # noqa: E402
from saree_ai.voice.intents import parse, respond, HELP_TEXT  # noqa: E402
from saree_ai.voice.tts import build_tts  # noqa: E402
from saree_ai.i18n import LANGS  # noqa: E402

log = logging.getLogger("uvicorn.error")   # shows in the server console next to the access log

PRIVACY = ("All processing happens on this computer. Camera frames and microphone audio are sent only to "
           "this local server (127.0.0.1), are processed in memory, and are never stored or transmitted "
           "to any external service. No external AI API is used.")
PRIVACY_LITE = ("Free online demo: your camera picture never leaves your device, and body tracking is switched off. "
                "English voice commands are recognised inside your browser; only the command text is sent to this "
                "app's server. Telugu voice commands are recognised on this app's server (the in-browser engine "
                "cannot handle Telugu yet), so in Telugu mode your microphone audio is sent there; it is processed "
                "in memory and never stored. No third-party AI service is used.")
PRIVACY_HOSTED = ("This is the HOSTED demo: your camera frames and microphone audio are sent over an encrypted "
                  "connection to this app's own server for processing. They are processed in memory, never "
                  "stored, and never sent to any third-party AI service. For fully on-device processing, run "
                  "the app locally (see README).")


def load_config(path: Path, allow_synthetic: bool, hosted: bool = False, lite: bool = False) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["hosted"] = bool(hosted or lite)
    cfg["lite"] = bool(lite)
    if allow_synthetic:
        cfg["min_model_status"] = "smoke_test_synthetic"
        cfg["corrections"]["allow_experimental_corrections"] = False  # still never correct from synthetic-only models
    return cfg


class LiteModels:
    """Free-hosting mode: no vision models on the server (they need ~2 GB RAM)."""

    def __init__(self, cfg):
        self.stage_cfg = {**cfg.get("stage", {}), "min_status": cfg["min_model_status"]}

    def status(self):
        off = {"name": "off", "available": False, "detail": "disabled in the free online demo (needs ~2 GB RAM)",
               "experimental": None, "version": None, "registry_status": None}
        return {"pose": off, "segmentation": off, "pleat_model": off, "pleat_signal": off}


def create_app(cfg: dict) -> FastAPI:
    lite = bool(cfg.get("lite"))
    info = hardware.inspect(gpu=not lite)
    sel = hardware.choose_profile(info)
    vcfg = cfg.get("voice", {})
    if lite:
        # English recognition runs in the visitor's browser (frontend/asr). The 2022 browser engine cannot decode
        # the Telugu model (tested 2026-11-06), so Telugu is recognised here (~3% of one core per listener,
        # ~225 MB RAM). Speech output is pre-recorded (frontend/tts).
        models, tts, tts_status = LiteModels(cfg), {}, {}
        server_langs = [x for x in os.environ.get("LITE_SERVER_ASR", "te").split(",") if x]
        asr, asr_msg = build_asr({**vcfg, "vosk_models": {lg: vcfg["vosk_models"][lg] for lg in server_langs
                                                          if lg in vcfg.get("vosk_models", {})}})
    else:
        models = Models(cfg, sel, ROOT)
        asr, asr_msg = build_asr(vcfg)
        tts, tts_status = build_tts(vcfg)
    default_lang = vcfg.get("default_language", "en")
    infer_lock = threading.Lock()   # one GPU model instance shared by sessions
    port = cfg["server"]["port"]
    hosted = bool(cfg.get("hosted"))
    privacy = PRIVACY_LITE if lite else (PRIVACY_HOSTED if hosted else PRIVACY)
    allowed_origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
    for o in [os.environ.get("RENDER_EXTERNAL_URL", "")] + os.environ.get("ALLOWED_ORIGINS", "").split(","):
        if o.strip():
            allowed_origins.add(o.strip().rstrip("/"))
    max_sessions = int(os.environ.get("MAX_SESSIONS", "2" if hosted else "8"))
    max_asr = int(os.environ.get("MAX_ASR_STREAMS", "2" if hosted else "8"))
    active = {"n": 0, "asr": 0}

    def origin_allowed(origin: str | None, host: str | None) -> bool:
        if origin is None or origin in allowed_origins:
            return True
        # hosted: accept the page's own origin (same-origin), whatever domain Render assigns
        return hosted and host is not None and urlparse(origin).netloc == host

    app = FastAPI(title="Saree Mirror (local)", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "frontend" / "index.html")

    @app.get("/healthz")
    @app.get("/health")          # Render's default health-check path
    def healthz():
        return {"ok": True}

    @app.get("/api/status")
    def status():
        return JSONResponse({
            "hardware": info, "selection": sel, "models": models.status(),
            "asr": {"available": asr is not None, "detail": asr_msg,
                    "languages": {lg: (asr.status.get(lg, "browser" if lite else "unavailable") if asr else
                                       ("browser" if lite else "unavailable")) for lg in LANGS},
                    "server_languages": sorted(asr.recognizers) if asr else []},
            "tts_server": tts_status,
            "languages": LANGS, "default_language": default_lang,
            "min_model_status": cfg["min_model_status"],
            "thresholds": cfg["thresholds"],
            "privacy": privacy,
            "hosted": hosted,
            "lite": lite,
            "help": HELP_TEXT,
        })

    @app.get("/api/grammar")
    def grammar(lang: str = "en"):
        from saree_ai.voice.intents import grammar_phrases
        return {"lang": lang, "phrases": grammar_phrases(lang if lang in LANGS else "en")}

    @app.post("/api/tts")
    async def tts_endpoint(body: dict, request: Request):
        if not origin_allowed(request.headers.get("origin"), request.headers.get("host")):
            return JSONResponse({"error": "forbidden origin"}, status_code=403)
        lang, text = body.get("lang"), str(body.get("text", ""))
        if lang not in tts or not text.strip():
            return JSONResponse({"error": f"no local TTS for {lang!r}"}, status_code=404)
        wav = await asyncio.to_thread(tts[lang].wav, text)
        return Response(wav, media_type="audio/wav")

    def origin_ok(ws: WebSocket) -> bool:
        return origin_allowed(ws.headers.get("origin"), ws.headers.get("host"))

    @app.websocket("/ws/session")
    async def ws_session(ws: WebSocket):
        if not origin_ok(ws):
            await ws.close(code=1008); return
        await ws.accept()
        if active["n"] >= max_sessions:
            await ws.send_text(json.dumps({"type": "busy", "detail": f"The demo server is busy ({max_sessions} users). Please try again in a few minutes."}))
            await ws.close(code=1013); return
        active["n"] += 1
        sess = Session(models, cfg, ROOT)
        lang = ws.query_params.get("lang", default_lang)
        sess.engine.set_language(lang if lang in LANGS else default_lang)
        last_guidance = None

        def step_info():
            e = sess.engine
            return {"id": e.step["id"], "index": e.step_index, "title": e.say("title"),
                    "instruction": e.say("instruction"), "total": len(e.t.steps)}
        await ws.send_text(json.dumps({"type": "hello", "step": step_info(), "lang": sess.engine.lang,
                                       "speech": sess.engine.instruction_speech(), "privacy": privacy}))
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None and lite:
                    continue   # no vision models in lite mode; the frontend does not send frames
                if msg.get("bytes") is not None:
                    data = msg["bytes"]
                    (hlen,) = struct.unpack("<I", data[:4])
                    header = json.loads(data[4:4 + hlen])
                    img = cv2.imdecode(np.frombuffer(data[4 + hlen:], np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        await ws.send_text(json.dumps({"type": "error", "frame_id": header.get("frame_id"), "error": "bad frame"}))
                        continue

                    def run():
                        with infer_lock:
                            return sess.process(img, header.get("visible"))
                    out = await asyncio.to_thread(run)
                    del img  # frames are never persisted
                    last_guidance = out["guidance"]
                    out.update({"type": "result", "frame_id": header.get("frame_id"), "client_ts": header.get("ts")})
                    await ws.send_text(json.dumps(out))
                elif msg.get("text") is not None:
                    m = json.loads(msg["text"])
                    if m.get("type") == "set_language" and m.get("lang") in LANGS:
                        sess.engine.set_language(m["lang"])
                        log.info("[voice] session language -> %s", m["lang"])
                        await ws.send_text(json.dumps({"type": "voice", "action": "language", "intent": None,
                                                       "speech": sess.engine.say("speech"), "lang": sess.engine.lang,
                                                       "step": step_info()}))
                        continue
                    if m.get("type") == "command":
                        intent, heard = m.get("intent"), None
                    elif m.get("type") == "utterance":
                        # the audio socket already parsed it; re-parse so typed text works too
                        p = parse(m.get("text", ""), sess.engine.lang)
                        intent, heard = (m.get("intent") or p["intent"]), p
                        if intent is None:
                            # browser-side recognition sends every transcript; ignore non-commands quietly
                            log.info("[voice] heard (browser) %r -> no command", m.get("text"))
                            await ws.send_text(json.dumps({"type": "heard", "text": m.get("text"), "intent": None}))
                            continue
                    else:
                        continue
                    before = sess.engine.step_index
                    r = respond(intent, sess.engine, last_guidance)
                    log.info("[voice] session cmd type=%s text=%r intent=%s step %d -> %d (lang=%s)", m.get("type"),
                             m.get("text"), intent, before + 1, sess.engine.step_index + 1, sess.engine.lang)
                    if r["action"] in ("step", "confirm"):
                        sess.stage.reset()
                    await ws.send_text(json.dumps({"type": "voice", "intent": intent, "heard": heard, **r,
                                                   "lang": sess.engine.lang, "step": step_info()}))
        except WebSocketDisconnect:
            pass
        finally:
            active["n"] -= 1

    @app.websocket("/ws/audio")
    async def ws_audio(ws: WebSocket):
        if not origin_ok(ws):
            await ws.close(code=1008); return
        await ws.accept()
        if asr is None:
            await ws.send_text(json.dumps({"type": "asr_unavailable", "detail": asr_msg}))
            await ws.close(); return
        lang = ws.query_params.get("lang", default_lang)
        if active["asr"] >= max_asr:
            await ws.send_text(json.dumps({"type": "asr_unavailable", "detail": "the voice server is busy - please use the buttons or try again later"}))
            await ws.close(); return
        stream = asr.stream(lang if lang in LANGS else default_lang)
        if stream is None:
            await ws.send_text(json.dumps({"type": "asr_unavailable", "detail": f"no recogniser for {lang}: {asr_msg}"}))
            await ws.close(); return
        await ws.send_text(json.dumps({"type": "asr_ready", "languages": [lg for lg, _ in stream.recs]}))
        log.info("[voice] mic stream connected, recognisers=%s", [lg for lg, _ in stream.recs])
        active["asr"] += 1
        chunks, peak, t_last = 0, 0.0, time.monotonic()
        try:
            while True:
                data = await ws.receive_bytes()
                chunks += 1
                pcm = np.frombuffer(data, np.int16)
                if len(pcm):
                    peak = max(peak, float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768) ** 2))))
                if time.monotonic() - t_last >= 5:
                    # level ~0.000 = silence / muted mic; ~0.01-0.2 = speech reaching the server
                    log.info("[voice] audio: %d chunks in last 5 s, peak level %.3f", chunks, peak)
                    chunks, peak, t_last = 0, 0.0, time.monotonic()
                r = await asyncio.to_thread(stream.feed, data)
                if r:
                    log.info("[voice] heard %r (asr=%s, conf=%s) -> intent=%s", r["text"], r["asr_lang"], r["confidence"], r["intent"])
                    await ws.send_text(json.dumps({"type": "transcript", **r}))
        except WebSocketDisconnect:
            log.info("[voice] mic stream closed")
        finally:
            active["asr"] -= 1

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "app.yaml"))
    ap.add_argument("--allow-synthetic-models", action="store_true",
                    help="DEV ONLY: load models trained only on synthetic data (outputs tagged experimental)")
    ap.add_argument("--hosted", action="store_true", default=os.environ.get("SAREE_HOSTED") == "1",
                    help="HOSTED mode for a public server (Render): binds 0.0.0.0:$PORT; UI discloses server-side processing")
    ap.add_argument("--lite", action="store_true", default=os.environ.get("SAREE_LITE") == "1",
                    help="FREE-HOSTING mode (512 MB): no vision models; speech runs in the browser")
    a = ap.parse_args()
    cfg = load_config(Path(a.config), a.allow_synthetic_models, hosted=a.hosted or a.lite, lite=a.lite)
    import uvicorn
    if a.hosted or a.lite:
        uvicorn.run(create_app(cfg), host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "10000")), log_level="info",
                    proxy_headers=True, forwarded_allow_ips="*")
        return
    host = cfg["server"]["host"]
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("Refusing to bind to a non-loopback address in local mode: camera data must stay on this device. "
                         "Use --hosted only for a deliberate public deployment.")
    uvicorn.run(create_app(cfg), host=host, port=cfg["server"]["port"], log_level="info")


if __name__ == "__main__":
    main()

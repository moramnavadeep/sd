import json
import socket
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "training"))

from saree_ai.constants import KP, SEG
from saree_ai.correction.checks import FrameState
from saree_ai.correction.engine import CorrectionEngine, Tutorial
from saree_ai.features import FEATURE_DIM, FEATURE_NAMES, frame_features
from saree_ai.inference.pleat import PleatSignalAnalyzer, PleatResult
from saree_ai.inference.pose import PoseResult, estimate_orientation
from saree_ai.inference.segmentation import SegResult
from saree_ai.inference.stage import StageResult, heuristic_stage_probs, StageRecognizer
from saree_ai.voice.intents import parse

CFG = yaml.safe_load((ROOT / "config/app.yaml").read_text())
TH = CFG["thresholds"]

KP_FRONT = np.array([[128, 50], [134, 45], [122, 45], [142, 50], [114, 50], [158, 88], [98, 88], [166, 142], [90, 142],
                     [170, 192], [86, 192], [148, 192], [108, 192], [146, 272], [110, 272], [144, 352], [112, 352]], np.float32)


def kps(conf=0.95):
    return np.c_[KP_FRONT, np.full(17, conf)].astype(np.float32)


def state(pose_conf=0.95, seg_conf=0.9, pleat_conf=0.9, pallu_side="left", pleat_count=6.0, spacing_cv=0.1):
    W, H = 256, 384
    mask = np.zeros((H, W), np.uint8)
    mask[180:200, 90:170] = SEG["waist_wrap"]
    mask[200:360, 90:170] = SEG["saree_body"]
    mask[175:360, 120:150] = SEG["pleats"]
    sh = KP_FRONT[KP["left_shoulder"] if pallu_side == "left" else KP["right_shoulder"]]
    cv2.line(mask, (100, 190), (int(sh[0]), int(sh[1])), SEG["pallu"], 18)
    k = kps(0.95)
    pose = PoseResult(k, pose_conf, backend="test")
    seg = SegResult(True, mask, seg_conf, ambiguous_fraction=0.05, experimental=False)
    pl = PleatResult(True, "signal", [120, 175, 150, 360], count=pleat_count, count_conf=pleat_conf, spacing_cv=spacing_cv,
                     alignment_deg=2.0, tilt_deg=1.0, confidence=pleat_conf, experimental=False, fold_x=[125, 130])
    f, ex = frame_features(mask, k, pl.as_feature_input())
    return FrameState(W, H, pose, seg, pl, StageResult(True, "pleats_forming", 0.9, source="test", experimental=False), f, ex)


def engine(**kw):
    c = {**CFG["corrections"], **kw}
    return CorrectionEngine(Tutorial(ROOT / CFG["tutorial"]), TH, c, {"min_interval_s": 0, "repeat_suppress_s": 0})


def run(e, s, n=15):
    out = None
    for _ in range(n):
        out = e.evaluate(s)
    return out


def test_feature_vector_shape():
    f, _ = frame_features(state().seg.mask, kps(), None)
    assert f.shape == (FEATURE_DIM,) == (len(FEATURE_NAMES),)
    assert f[FEATURE_NAMES.index("pallu_left_shoulder")] > f[FEATURE_NAMES.index("pallu_right_shoulder")]


def test_low_pose_confidence_gives_positioning_not_correction():
    e = engine(); e.goto(3)
    g = run(e, state(pose_conf=0.2, pleat_count=2))
    assert g["status"] == "need_position" and g["corrections"] == []


def test_low_seg_and_pleat_confidence_never_corrects():
    e = engine(); e.goto(3)
    g = run(e, state(seg_conf=0.3, pleat_conf=0.2, pleat_count=2, spacing_cv=0.9))
    assert g["corrections"] == [] and g["status"] in ("uncertain", "ok")


def test_confident_failure_becomes_correction_after_persistence():
    e = engine(); e.goto(3)  # form_pleats
    g1 = e.evaluate(state(pleat_count=2))
    assert g1["corrections"] == []                     # one frame is not enough
    g = run(e, state(pleat_count=2))
    assert "pleats_too_few" in g["corrections"] and g["status"] == "correction"


def test_experimental_models_do_not_correct_by_default():
    e = engine(); e.goto(3)
    s = state(pleat_count=2); s.pleat.experimental = True
    g = run(e, s)
    assert g["corrections"] == []
    assert any(c["status"] == "unverified" for c in g["checks"])


def test_wrong_shoulder_detected():
    e = engine(); e.goto(5)
    g = run(e, state(pallu_side="right"))
    assert "pallu_wrong_shoulder" in g["corrections"]


def test_manual_confirm_and_navigation():
    e = engine(); e.goto(6)
    run(e, state())
    e.confirm_manual(None)
    g = run(e, state())
    assert g["status"] == "step_complete"
    e.restart(); assert e.step_index == 0


def test_heuristic_stage_is_capped_below_threshold():
    s = state()
    p = heuristic_stage_probs(s.feats)
    assert p.max() <= 0.55 < TH["stage"]


def test_stage_filter_without_model_or_seg():
    r = StageRecognizer({"min_status": "experimental"})
    assert r(None, False).available is False


def test_pleat_analyzer_counts_clean_folds_and_rejects_noise():
    img = np.full((200, 240, 3), 150, np.uint8)
    for i, x in enumerate(range(15, 225, 30)):          # 7 folds, 30 px each, alternating shade
        img[:, x:x + 30] = 95 if i % 2 else 205
    res = PleatSignalAnalyzer()(img, [0, 0, 240, 200])
    assert res.count == 7, res
    assert res.spacing_cv < 0.1 and res.confidence > 0.3, res
    noise = np.random.default_rng(0).integers(0, 255, (200, 240, 3)).astype(np.uint8)
    rn = PleatSignalAnalyzer()(noise, [0, 0, 240, 200])
    assert rn.confidence < res.confidence


def test_onnx_sessions_never_use_cloud_providers():
    import inspect
    from saree_ai.inference import segmentation, pleat, stage, pose
    for mod in (segmentation, pleat, stage, pose):
        src = inspect.getsource(mod)
        assert "Azure" not in src
        for line in src.splitlines():
            if "InferenceSession(" in line:
                assert "CPUExecutionProvider" in line, line


def test_orientation():
    assert estimate_orientation(kps())[0] == "front"
    back = kps(); back[[KP["left_shoulder"], KP["right_shoulder"]]] = back[[KP["right_shoulder"], KP["left_shoulder"]]]
    back[:3, 2] = 0.05
    assert estimate_orientation(back)[0] == "back"


@pytest.mark.parametrize("text,intent", [("next step", "next"), ("Previous step", "previous"), ("repeat instruction", "repeat"),
                                         ("explain pleat formation", "explain_pleats"), ("show pallu placement", "show_pallu"),
                                         ("start again", "restart"), ("am I doing this correctly?", "check"), ("banana", None)])
def test_intents(text, intent):
    assert parse(text)["intent"] == intent


def test_derive_status():
    from common import derive_status
    syn = {"train": {"real": 0}, "test": {"real": 0}}
    assert derive_status(syn, {}, {})[0] == "smoke_test_synthetic"
    real = {"train": {"real": 100}, "test": {"real": 20}}
    assert derive_status(real, {"miou": 0.5}, {"miou": 0.6})[0] == "experimental"
    assert derive_status(real, {"miou": 0.7}, {"miou": 0.6})[0] == "validated"


def test_split_never_puts_synthetic_in_test(tmp_path):
    sp = json.loads((ROOT / "data/splits/default.json").read_text()) if (ROOT / "data/splits/default.json").exists() else None
    if sp is None:
        pytest.skip("no split file yet")
    assert all(not p.startswith("synth") for p in sp["participants"]["test"])


def test_server_status_origin_guard_and_no_external_network(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.server import create_app, load_config
    real_connect = socket.socket.connect
    attempts = []

    def guard(self, addr):
        host = addr[0] if isinstance(addr, tuple) else str(addr)
        if host not in ("127.0.0.1", "::1", "localhost", "testserver"):
            attempts.append(host)
            raise AssertionError(f"outbound connection attempted to {host}")
        return real_connect(self, addr)
    monkeypatch.setattr(socket.socket, "connect", guard)
    app = create_app(load_config(ROOT / "config/app.yaml", False))
    c = TestClient(app)
    st = c.get("/api/status").json()
    assert "privacy" in st and "models" in st
    with pytest.raises(Exception):
        with c.websocket_connect("/ws/session", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_text()
    with c.websocket_connect("/ws/session", headers={"origin": "http://127.0.0.1:8765"}) as ws:
        assert json.loads(ws.receive_text())["type"] == "hello"
        img = cv2.imread(str(next((ROOT / "data/synthetic_v1/images").glob("*.jpg")))) if (ROOT / "data/synthetic_v1/images").exists() \
            else np.zeros((384, 256, 3), np.uint8)
        jpg = cv2.imencode(".jpg", img)[1].tobytes()
        hdr = json.dumps({"frame_id": 1, "ts": 0}).encode()
        ws.send_bytes(len(hdr).to_bytes(4, "little") + hdr + jpg)
        out = json.loads(ws.receive_text())
        assert out["type"] == "result" and out["frame_id"] == 1 and "guidance" in out
        ws.send_text(json.dumps({"type": "utterance", "text": "next step"}))
        v = json.loads(ws.receive_text())
        assert v["type"] == "voice" and v["intent"] == "next" and v["step"]["index"] == 1
    assert attempts == []


def test_body_outside_visible_screen_region_is_not_full_body():
    e = engine()
    s = state()
    s.extras["visible_px"] = [0, 60, 256, 384]   # screen crop hides the head
    g = e.evaluate(s)
    assert g["status"] == "need_position"
    s.extras["visible_px"] = [0, 0, 256, 384]
    assert e.evaluate(s)["status"] != "need_position"

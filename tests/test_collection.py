"""Data-collection tooling: protocol rotation, the session recorder (driven with video files
standing in for cameras - no real camera is opened) and the markers -> manifest import."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "training" / "data"))

from saree_ai.constants import CORRECTION_LABELS, DRAPE_STAGES  # noqa: E402
import record_session as rs  # noqa: E402
import label_video_steps as lv  # noqa: E402


def test_protocol_rotation_covers_every_label_and_slots_differ():
    p = rs.load_protocol()
    seen = set()
    for n in range(len(p["mistake_rotation"])):
        plan = rs.session_plan(p, f"P{n:03d}", "dim")
        assert [s["id"] for s in plan] == ["pose", "d1", "d2", "d3", "d4"]
        d2, d4 = plan[2], plan[4]
        assert d2["script"] == "mistakes_slot1" and d4["script"] == "mistakes_slot2"
        l2, l4 = [m["label"] for m in d2["mistakes"]], [m["label"] for m in d4["mistakes"]]
        assert l2 != l4
        # mistakes are presented in draping order
        order = [DRAPE_STAGES.index(m["during_stage"]) for m in d2["mistakes"]]
        assert order == sorted(order)
        assert plan[1]["mistakes"] == [] and plan[3]["mistakes"] == []
        seen |= set(l2) | set(l4)
    assert seen == set(CORRECTION_LABELS)


def _make_video(path: Path, seconds=3, fps=30, w=320, h=240, hue=0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(seconds * fps):
        f = np.full((h, w, 3), (40 + hue, 80, 120), np.uint8)
        cv2.putText(f, str(i), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        vw.write(f)
    vw.release()


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rec")
    srcs = [tmp / "cam_front.mp4", tmp / "cam_left.mp4"]
    _make_video(srcs[0]); _make_video(srcs[1], hue=60)
    p = rs.load_protocol()
    t0 = time.perf_counter()
    threads = {}
    for spec, src in zip(p["cameras"], srcs):
        th = rs.CameraThread(spec["name"], rs.open_camera(str(src)), t0)
        th.start(); threads[spec["name"]] = th
    deadline = time.time() + 5
    while any(t.latest is None for t in threads.values()) and time.time() < deadline:
        time.sleep(0.02)
    meta = {"participant_id": "P002", "consent_id": "C-2026-002", "session_id": "S-test", "operator": "test",
            "saree": {"A": "S01", "B": "S02"}, "l2_variant": "backlit", "protocol_version": p["protocol_version"],
            "cameras": [{**s, "source": str(x)} for s, x in zip(p["cameras"], srcs)]}
    plan = rs.session_plan(p, "P002", "backlit")
    rec = rs.Recorder(tmp / "out" / "P002" / "S-test", meta, plan, threads, t0)
    reports = []
    # pose clip
    rec.start(); time.sleep(0.4); rec.skip(); time.sleep(0.4); reports.append(rec.stop())
    # d1 (correct drape): just stage marks
    rec.start(); rec.mark_stage(0); time.sleep(0.4); rec.mark_stage(1); time.sleep(0.4); reports.append(rec.stop())
    # d2 (mistakes): stage marks, one held mistake, one skipped, a pause
    rec.start()
    rec.mark_stage(1); time.sleep(0.3)
    rec.mark_stage(2); time.sleep(0.3)
    first = rec.next_mistake()["label"]
    rec.toggle_mistake(); time.sleep(1.5); rec.toggle_mistake()
    second = rec.next_mistake()["label"]
    rec.skip()
    rec.toggle_pause(); time.sleep(0.4); rec.toggle_pause()
    rec.mark_stage(3); time.sleep(0.6)
    reports.append(rec.stop())
    for t in threads.values():
        t.stop_flag = True
    return {"tmp": tmp, "rec": rec, "reports": reports, "first": first, "second": second,
            "seg": tmp / "out" / "P002" / "S-test"}


def test_recorder_writes_consistent_files(recorded):
    seg = recorded["seg"]
    assert json.loads((seg / "session.json").read_text())["participant_id"] == "P002"
    for name in ("pose", "d1", "d2"):
        d = seg / name
        m = json.loads((d / "markers.json").read_text())
        for cam in ("front", "front_left"):
            ts = np.loadtxt(d / f"{cam}.ts.csv", delimiter=",", skiprows=1, ndmin=2)
            assert len(ts) == m["frames"][cam] > 10
            assert np.all(np.diff(ts[:, 1]) > 0), "timestamps must increase"
            cap = cv2.VideoCapture(str(d / f"{cam}.mp4"))
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(ts)   # every written frame has a timestamp
            cap.release()
    assert not (seg / "d3").exists()


def test_recorder_marks_and_sync_report(recorded):
    seg = recorded["seg"]
    pose = json.loads((seg / "pose" / "markers.json").read_text())
    assert len(pose["prompts"]) == 2
    m = json.loads((seg / "d2" / "markers.json").read_text())
    assert [x["stage"] for x in m["stage_marks"]] == ["wrapping_started", "waist_wrap_complete", "pleats_forming"]
    held = [x for x in m["mistakes"] if not x.get("skipped")]
    skipped = [x for x in m["mistakes"] if x.get("skipped")]
    assert held[0]["label"] == recorded["first"] and 1300 < held[0]["end_ms"] - held[0]["start_ms"] < 2500
    assert skipped[0]["label"] == recorded["second"]
    assert recorded["second"] in m["mistakes_not_recorded"]
    assert len(m["pauses"]) == 1 and m["pauses"][0]["end_ms"] > m["pauses"][0]["start_ms"]
    rep = m["sync_report"]
    for cam in ("front", "front_left"):
        assert 20 < rep["cameras"][cam]["fps_measured"] < 40
        assert rep["cameras"][cam]["dropped_frac"] < 0.05, rep   # pause gap must not count as dropped
    assert rep["nearest_frame_offset_ms"]["p95"] < 50
    # the skipped mistake AND the third one (never reached) are reported as not recorded
    planned = [x["label"] for x in recorded["rec"].plan[2]["mistakes"]]
    assert recorded["reports"][2]["mistakes_not_recorded"] == planned[1:]


def test_marks_import_to_manifest(recorded, tmp_path):
    import jsonschema
    seg = recorded["seg"]
    sarees = tmp_path / "sarees.csv"
    sarees.write_text("saree_id,primary_color,darkness(light|medium|dark),fabric(cotton|silk),pattern(plain|printed),"
                      "transparency(opaque|translucent|sheer),length_m,source(pool|participant)\n"
                      "S01,red,light,cotton,plain,opaque,5.5,pool\n", encoding="utf-8")
    info = lv.load_marks(seg / "d2" / "markers.json", sarees_csv=sarees)
    assert info["participant_id"] == "P002" and info["consent_id"] == "C-2026-002" and info["camera"] == "front"
    assert [s for _, s in info["stage_marks"]] == [1, 2, 3]
    assert len(info["mistakes"]) == 1 and info["mistakes"][0]["f1"] > info["mistakes"][0]["f0"]
    assert info["saree_id"] == "S01" and info["saree"]["fabric"] == "cotton"
    out = tmp_path / "ds"
    n = lv.save_samples(info["video"], info["stage_marks"], out, info["participant_id"], info["consent_id"],
                        video_id="P002-S-test-d2-front", sample_every_s=0.1, ts_ms=info["ts_ms"], mistakes=info["mistakes"],
                        camera_angle="front", lighting=lv.lighting_value(info), saree=info["saree"], notes="test")
    rows = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
    assert n == len(rows) > 5
    schema = json.loads((ROOT / "saree_ai/schema/sample.schema.json").read_text(encoding="utf-8"))
    for r in rows:
        jsonschema.validate(r, schema)
        assert (out / r["image"]).exists()
        assert r["capture"]["lighting"] == "backlit"   # d2 uses L2; today's L2 variant is backlit
    mk = info["mistakes"][0]
    inside = [r for r in rows if mk["f0"] <= r["frame_index"] <= mk["f1"]]
    assert inside and all(r["correction_labels"] == [mk["label"]] for r in inside)
    assert all(r["correction_labels"] == [] for r in rows if r not in inside)
    # nothing sampled within 0.5 s of the mistake's start/end
    assert all(min(abs(r["frame_index"] - mk["f0"]), abs(r["frame_index"] - mk["f1"])) >= 15 for r in rows)
    # frames near a stage boundary are flagged uncertain
    for r in rows:
        near = any(abs(r["frame_index"] - f) < 15 for f, _ in info["stage_marks"])
        assert r["step_label_certain"] is (not near)


def test_marks_import_cli_headless(recorded, tmp_path):
    import subprocess
    out = tmp_path / "cli_ds"
    r = subprocess.run([sys.executable, str(ROOT / "training/data/label_video_steps.py"), "--marks",
                        str(recorded["seg"] / "d1" / "markers.json"), "--out", str(out), "--no-gui"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
    assert rows and {x["step_label"] for x in rows} <= {"not_started", "wrapping_started"}
    assert all(x["participant_id"] == "P002" and x["source"]["consent_id"] == "C-2026-002" for x in rows)

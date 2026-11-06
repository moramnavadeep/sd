# Saree Mirror — on-device AI saree draping guide

A smart-mirror web app that watches you drape a saree (Nivi style) through your webcam, overlays
guidance on your mirrored reflection, and gives spoken feedback. **Everything runs on your
computer.** There are no external AI APIs, no API keys and no cloud inference. Camera frames go
only to a local server on 127.0.0.1, are processed in memory, and are never stored.

> **Current status: pipeline built, saree models NOT yet trained on real data.**
> The pose baseline (third-party, pretrained) and the local voice commands work. The saree
> segmentation, pleat and stage models have a complete training/evaluation/export pipeline, but
> no consented real dataset exists yet, so the app currently runs those parts in *uncertain*
> mode: it shows positional guides and asks you to confirm steps manually rather than inventing
> corrections. See [docs/STATUS_REPORT.md](docs/STATUS_REPORT.md).

## Quick start (Windows, PowerShell)

```bash
python -m venv .venv
```
```bash
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
```bash
.venv\Scripts\python -m pip install -r requirements.txt
```
```bash
.venv\Scripts\python scripts/download_models.py
```
```bash
.venv\Scripts\python -m backend.server
```

Then open http://127.0.0.1:8765 in Chrome or Edge and allow the camera. After the one-time
download step the app works with the network disconnected.

To test without a camera, put a recording or image under `frontend/` and open
`http://127.0.0.1:8765/?demo=/static/<file>`; `scripts/run_on_video.py` runs the same pipeline
offline on a video and writes `results.jsonl` plus an annotated video.

Keys: **N** next · **P** back · **R** repeat · **C** confirm step · **Space** "am I right?" ·
**M** toggle voice. Voice commands: *next step, previous step, repeat instruction, explain pleat
formation, show pallu placement, start again, am I doing this correctly, confirm*.

## Layout

| Path | What |
|---|---|
| `saree_ai/` | shared package: label spaces, schema, features, model architectures, registry, inference, correction engine, voice intents |
| `backend/server.py` | FastAPI server (loopback only) |
| `frontend/` | mirror UI (no CDN; plain ES modules) |
| `config/` | thresholds, tutorial definition (`tutorials/nivi.yaml`) |
| `training/` | dataset tools, training, evaluation, ONNX export, configs |
| `models/` | weights + `registry.json` (status, provenance, metrics) |
| `docs/` | architecture, dataset & labelling standard, model cards, dependencies, status report |

## Training workflow

Data collection follows [docs/DATA_COLLECTION_PLAN.md](docs/DATA_COLLECTION_PLAN.md) and
`data_collection/`. Record each session with the local multi-camera recorder, which writes to
the encrypted drive and never uploads anything. Then import each segment's live stage and
mistake marks:

```bash
.venv\Scripts\python training/data/record_session.py --out E:/saree_rec --cams 0 1 --participant P001 --consent C-2026-001 --saree-a S01 --saree-b S02 --l2 dim
```
```bash
.venv\Scripts\python training/data/label_video_steps.py --marks E:/saree_rec/P001/<session>/d2/markers.json --out data/real_v1
```

Then annotate regions, landmarks and pleats in self-hosted CVAT, and validate, split and train:

```bash
.venv\Scripts\python training/data/validate_dataset.py data/real_v1
```
```bash
.venv\Scripts\python training/data/prepare_dataset.py split --dirs data/real_v1 data/synthetic_v1 --out data/splits/default.json
```
```bash
.venv\Scripts\python training/train_segmentation.py --config training/configs/segmentation.yaml
```
```bash
.venv\Scripts\python training/evaluate.py --run runs/segmentation/<stamp>
```
```bash
.venv\Scripts\python training/export_onnx.py --run runs/segmentation/<stamp>
```

Same for `train_pleat.py`, `train_stage.py`, `train_pose.py`. Training never starts
automatically. Registry status is derived from data provenance and test metrics; models trained
without real data are tagged `smoke_test_synthetic` and the app refuses to load them unless
started with `--allow-synthetic-models` (dev only).

Tests: `.venv\Scripts\python -m pytest tests` and `node --test tests/js/geometry.test.mjs`.

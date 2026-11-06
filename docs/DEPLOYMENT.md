# Deployment options

| | Local (default) | Free online demo (Render free) | Full online (paid, ≥ 2 GB RAM) |
|---|---|---|---|
| Start | `python -m backend.server` | `Dockerfile` (lite) / `render.yaml` | `Dockerfile.full` |
| Camera | stays on this PC; body tracking on | **stays on the visitor's device**; body tracking **off** | sent to the server; body tracking on (CPU, about 1 update/s) |
| English voice | recognised on this PC | recognised **in the browser** (vosk-browser WASM) | recognised on the server |
| Telugu voice | recognised on this PC | recognised **on the server** (the 2022 browser engine cannot decode the Telugu model) | recognised on the server |
| Telugu speech output | Piper on this PC | pre-recorded at build time (22 sentences) | Piper on the server |
| Server RAM (measured) | n/a | ≈ 312 MB with the Telugu model | ≈ 1.5–2 GB |
| Privacy label in the UI | ON-DEVICE | states that only command text (English) or audio (Telugu) reaches the server | SERVER PROCESSING |

## Free online demo on Render

1. Render dashboard → **New + → Web Service** → repo `moramnavadeep/sd`, branch `main`.
2. Language **Docker** (it uses `./Dockerfile`), instance type **Free**.
3. Advanced → Health Check Path `/healthz`. No environment variables are needed.
4. Deploy. The first build takes several minutes: it downloads the models and pre-records the Telugu audio.

Free-plan limits: the service sleeps after about 15 minutes without visitors, and the first visit
after that takes about a minute to wake it. At most 2 people can use Telugu voice at the same time
(`MAX_ASR_STREAMS`); English voice has no such limit because it runs in each visitor's browser.
Visitors download the English speech model (≈ 40 MB) once; it is cached afterwards.

## Measured during preparation (2026-11-06)

* In-browser English recognition, using the real app page with recorded audio as the mic:
  "next step" → tutorial step 1 → 2, with zero requests to other sites.
* In-browser Telugu with vosk-browser 0.0.8: **failed**. It returned empty results with the full
  command grammar, even with a grammar containing only the "next" phrases. Hence Telugu runs on the server.
* Server-side Telugu: 3% of one i9 core in real time and 10/10 test commands recognised, measured
  on this PC. On Render's 0.1 CPU this is an estimate, not yet measured on Render itself.

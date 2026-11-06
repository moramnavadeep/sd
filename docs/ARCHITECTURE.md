# Architecture

```
Browser (localhost page, no CDN)                 Local Python backend (127.0.0.1 only)
───────────────────────────────                  ──────────────────────────────────────────
getUserMedia camera ──JPEG frame + id──ws──▶     /ws/session  Session.process(frame)
                                                   ├─ Model A  pose        (keypointrcnn baseline | own ONNX)
                                                   ├─ Model B  segmentation (own ONNX)
                                                   ├─ Model C  pleats       (signal analyzer + own ONNX)
                                                   ├─ features (saree_ai/features.py, interpretable)
                                                   ├─ Model D  stage        (own ONNX MLP | heuristic) + HMM filter
                                                   └─ Model E  correction engine (rules, confidence-gated)
mirror renderer  ◀── JSON: keypoints, regions, ──  structured result
(geometry.js / overlay.js / smoothing.js)          guides, status, speech, confidences
microphone PCM16 ──────────────────────ws──▶     /ws/audio  Vosk (grammar-constrained) → intent
speechSynthesis (localService voices only)
```

## Why a local Python backend (not in-browser ONNX Runtime Web)

Measured target: i9-13900HX, 32 GB RAM, RTX 4060 Laptop 8 GB. Keypoint R-CNN needs a GPU for
real-time use and is not practical in the browser; the backend runs on the same machine over
loopback, so latency is a few ms of JPEG encode/decode. ONNX Runtime Web remains possible for
our small models (B/C/D) later, but would need the runtime vendored locally (no CDN) to keep the
app offline. The `lite` profile (no CUDA GPU) runs the same backend at lower resolution and FPS;
`saree_ai/hardware.py` picks the profile and the UI shows an error if minimums aren't met. There
is no cloud fallback anywhere.

## Coordinate systems (single source of truth)

* Backend emits coordinates normalised to [0,1] in the **un-mirrored camera frame**.
* `frontend/js/geometry.js` is the only place that applies `object-fit: cover` scaling, cropping
  and mirroring. Tests: `tests/js/geometry.test.mjs`.
* Resolution changes (`video` resize event, settings change) reset smoothing so overlays re-anchor.

## Temporal behaviour

* Frames are sent with ids; one frame in flight (back-pressure, no queue build-up).
* **Live mode**: live video + One-Euro-smoothed landmarks/guides (low lag, small misalignment
  during fast motion). **Aligned mode**: the exact analysed frame is drawn with its own results
  (exact alignment, ~1 inference of lag).
* Track loss: landmarks hold for 700 ms, fade over 400 ms, then the framing guide asks the user to
  step back in. On re-entry filters reset (no sliding across the screen).
* Stage: HMM forward filter with a tutorial-order transition prior.
* Corrections: must hold for ≥ 70% of the last 12 frames.

## Trust model (requirement 14)

| Input | Threshold (config/app.yaml) | If below |
|---|---|---|
| body landmarks | 0.55 | positioning instruction only, no draping feedback |
| single keypoint | 0.30 | not used for anchoring |
| segmentation | 0.65 mean max-softmax, ≤ 35% ambiguous pixels | dependent checks → "uncertain", manual confirm |
| pleats | 0.55 | pleat checks → "uncertain" |
| stage | 0.60 filtered belief | step completion not inferred; manual confirm |

Experimental models (registry status below `validated`) produce "unverified" observations, never
corrections, unless `allow_experimental_corrections` is set. Models trained only on synthetic
data are not loaded at all unless the server is started with `--allow-synthetic-models`.

## Model registry

`models/registry.json` — per task, every version with file, sha256, status (derived, never hand
set), dataset split summary and versions, test metrics, ONNX parity error, measured latency,
limitations, licence/origin for third-party weights.

# Status and test report — 2026-11-06

Legend: **Implemented** = code exists and runs · **Verified** = I checked it with a test or a
demonstration on this machine (named below) · **Experimentally validated** = measured on real,
consented, participant-held-out data · **Not implemented / not done**.

**Nothing in this project is experimentally validated yet.** There is no real dataset, so no
saree-specific model has been trained on real data.

## Test machine

i9-13900HX (24C/32T), 31.7 GB RAM, RTX 4060 Laptop 8 GB, Windows 11, Python 3.13.14,
torch 2.11.0+cu128, onnxruntime 1.31.1. Selected profile: `full`.

## Automated tests

* `pytest tests`: **24 passed.** These cover: confidence gating, persistence, the
  experimental→unverified rule, wrong-shoulder detection, manual confirm/navigation, the
  heuristic stage cap, the pleat analyzer (exact count on clean folds, lower confidence on
  noise), orientation, the 8 voice intents, status derivation, "no synthetic data in test",
  the on-screen visible-region check, ONNX sessions never using cloud providers, and a server
  end-to-end run. The end-to-end run loads the real pose baseline on the GPU, checks
  `/api/status`, rejects a foreign-origin WebSocket, processes a frame, executes a voice
  command, and asserts **zero outbound non-loopback connection attempts**.
* `node --test tests/js/geometry.test.mjs`: **3 passed.** These cover the cover/mirror
  transform and its inverse, re-anchoring after a resolution change, One-Euro jitter
  attenuation, and track loss with reset on re-entry.

## Pipeline smoke run (SYNTHETIC DATA ONLY)

Purpose: prove the toolchain works end to end. The pipeline runs synthetic generation →
validation → participant split → versioning → training → evaluation → ONNX export → parity
check → registration. Data: 1,680 synthetic frames (60 synthetic "participants"), dataset
version `synthetic-b9f7ac640a35`. The validator reported 0 errors and 0 warnings. The split
was train 51 / val 9 / **test 0** participants: the test split is empty by design because
there is no real data.

| Model | Epochs | Val metric (synthetic val, **not real-world**) | ONNX parity | ORT CPU latency (median) | Registry status |
|---|---|---|---|---|---|
| B segmentation (LR-ASPP MBv3, 384×256) | 6 | mIoU 0.770, saree-mIoU 0.683, macro-F1 0.863 | 7.8e-6 | 5.0 ms | `smoke_test_synthetic` |
| C PleatNet (128²) | 8 | quality macro-F1 0.441, count MAE 3.15, count ±1 acc 0.30, secured F1 0.73 | 3.8e-6 | 0.58 ms | `smoke_test_synthetic` |
| D stage MLP | 150 | acc 1.00 / macro-F1 1.00 (trivial: features are computed from ground truth on cartoon renders) | 3.6e-7 | 0.02 ms | `smoke_test_synthetic` |
| A pose baseline (third party) | – | detected the person in only 7/40 synthetic frames (it is a real-photo model) | – | ≈47 ms on GPU (FP16, long side 384) | `pretrained_baseline` |

The app **refuses** all three smoke-test models by default. This was confirmed in the UI
status panel: "status 'smoke_test_synthetic' is below required 'experimental'".

## Correction engine (Model E), oracle inputs, synthetic val (252 frames)

* **Low-confidence probe: PASS.** With every input forced below threshold, it emitted 0
  corrections across 252 frames.
* Micro-precision 0.54, recall 0.22, F1 0.31, false-positive rate 4.2%. An initial run gave
  FPR 13.7%; it exposed a logic bug ("pallu not on shoulder" was flagged while the user is
  still placing the pallu), which I fixed. The re-measurement used the **same** set, so 4.2% is
  optimistic.
* Open problem: `pleats_off_center` has a false-positive rate of 0.57 on this set and needs
  rework with real data.
* Not yet detected at all here: `hem_uneven`, `pleats_uneven_spacing`, `pleats_misaligned`
  and `pleats_too_many` have recall 0 with oracle masks. The pleat signal analyzer's confidence
  is usually below threshold on these renders, so the checks correctly return "uncertain"
  rather than guessing.

## Voice (local): English and Telugu

**Bug reported 2026-11-06: saying "next" did not move to the next step.** Root causes found and fixed:
1. The app connected to the tutorial session only after the camera started. When camera access
   failed (always, in the in-app browser pane), voice transcripts had nowhere to go. The session now
   connects first, and a camera failure is a message rather than a dead page.
2. When the microphone was refused, a dead audio socket was left open. The mic is now requested first.
3. The mic is muted while the app speaks. That mute relied on an event Chrome can drop, so it now
   has a hard deadline. The worklet is now also kept connected to the audio graph (the
   data-collection session spotted this one).
4. Telugu-specific: grammars were serialised with `json.dumps` defaults, which escape Telugu into
   `\u0c..` codes, so Vosk ignored every Telugu word. Now `ensure_ascii=False`.
5. Telugu-specific: in Telugu mode the parallel English recogniser misheard "మళ్లీ మొదలు పెట్టు"
   (start again) as "repeat do" and won. English results now count only on an exact phrase match,
   and are held 0.6 s so the Telugu result takes priority.

**Telugu support:** commands via Vosk `small-te-0.42` with a command grammar (47 of 48 candidate
words are in its vocabulary; the loan word "ప్లీట్స్" is not, so it is not used). English runs in
parallel. Telugu speech output comes from Piper `te_IN-maya` on the local server. All guidance,
check messages, tutorial steps and explanations are translated (`config/i18n/te.yaml`,
`config/tutorials/nivi.te.yaml`); a test fails if any message lacks a Telugu version. The
translations are a **draft and need native-speaker review**.

| Test (all local) | Result |
|---|---|
| Windows SAPI English audio → Vosk, English mode | 7/7 |
| Same English audio, Telugu mode (mixed-language use) | 7/7 |
| Piper Telugu audio → Vosk Telugu (`test_telugu_commands_end_to_end…`) | 7/7, after fix 5 |
| In-browser path (fake mic: WAV → AudioWorklet → `/ws/audio` → session), camera blocked | "next step" → step 2; Telugu "తరువాతి దశ" → step 2; Telugu UI and spoken instruction; 0 external requests |

**Not verified:** real human voices, accents and real microphones (all test audio was synthetic
TTS); background noise; Telugu translation quality. The mic is deliberately ignored while the app
is speaking, so commands spoken over its speech are dropped; the meter shows "paused while I
speak" when that happens.

## Acceptance tests (requirement 15)

| # | Requirement | State |
|---|---|---|
| 1 | Live mirrored reflection | **Implemented.** Live camera is **not verified**: the in-app browser blocks camera access. The mirrored rendering was verified with the `?demo=` file source. Please verify with your webcam in Chrome or Edge. |
| 2 | Guidance drawn over the reflection | **Verified** in the browser (skeleton, framing box, tuck/pleat zone guides) via the `?demo=` source, at 800×600 and 540×900. |
| 3 | Guidance follows body movement | **Implemented** (landmark-anchored guides, One-Euro smoothing, recovery). Unit-tested. **Not verified** with a moving person. |
| 4 | Complete a Nivi tutorial | **Implemented** (7 steps, navigation, manual confirm; tested). **Not done** by a real user. |
| 5 | Voice commands work locally | **Verified** in English and Telugu with synthesized audio, including through the browser audio path. **Not verified** with a real microphone or real speakers. |
| 6 | Identify saree regions with own trained model | **Not done.** Pipeline ready; requires real annotated data. |
| 7 | Distinguish draping stages | **Not done** on real data. Only the heuristic fallback runs, capped below the trust threshold. |
| 8 | Appropriate correction feedback | **Partially.** Rules implemented and measured on synthetic oracle inputs (above); not validated. |
| 9 | Uncertain detections do not generate false corrections | **Verified** by unit tests and the 252-frame low-confidence probe. Live behaviour shows "Not sure — check yourself" with no correction. |
| 10 | Camera frames never sent to external AI services | **Verified** at code and test level: loopback-only bind (refuses other hosts), WebSocket origin check, no CDN, Azure EP never used, in-process network guard during a session. |
| 11 | Works offline after install | **Implemented** (weights stored locally, no runtime downloads). The test asserts no outbound connections. **Not tested** with the network adapter physically disabled. |

## What is needed next (in order)

1. **Real, consented data collection** following `docs/DATASET.md` (≥ 40 participants), then
   annotation in self-hosted CVAT or Label Studio.
2. Train B/C/D on real data. Run `evaluate.py` on the held-out participants, then export; the
   registry status will be derived from the metrics.
3. Calibrate the pose baseline's keypoint confidence on our test set; recompute stage features
   from model predictions.
4. Rework `pleats_off_center` and the pleat checks using real error examples. Measure the
   correction false-positive rate on real data (target ≤ 5%).
5. Train our own pose model (`train_pose.py`) to replace the baseline.

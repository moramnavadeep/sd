# Data collection plan v1

Goal: a consented, participant-level dataset that lets us **train and honestly evaluate** our own
saree segmentation (B), pleat (C), draping-stage (D) and correction (E) models, and later our own
pose model (A′). Labelling rules are in [DATASET.md](DATASET.md); this document covers *how we get
the recordings*. Session mechanics live in [`data_collection/protocol.yaml`](../data_collection/protocol.yaml)
and are enforced by `training/data/record_session.py`.

> **Decisions you need to make before Phase 1** are listed in §11. Items marked *(assumption)*
> are my defaults and should be changed to match your situation.

## 1. Targets

| | Phase 1: pilot | Phase 2: v1 dataset |
|---|---|---|
| Participants | 8 | 40 in total (pilot participants included if the protocol stayed unchanged) |
| Held-out test participants | — (the pilot validates the protocol, not models) | 8 (20%), chosen by `prepare_dataset.py split` |
| Recorded drapes | 32 | 160 (4 per participant) + 40 pose clips |
| Raw video | ~6 h per camera | ~27 h per camera |
| Frames with stage labels | ~2k (1 fps sample) | ~10k |
| Frames with full masks | 300 (pilot, used to **measure** annotation speed) | ~6,000 (~150 per participant) |
| Pleat crops with fold lines + quality | 150 | ~2,500 |
| Instances of each deliberate mistake | ≥ 2 each | ≥ 11 each (`pleats_too_many`), ~23 for the others, held ~20 s each |

Why 40: at 8 test participants, one participant is 12.5% of the test set. Below that, a single
person's body shape or saree dominates the result. 40 is the smallest number where a
participant-level split still leaves enough training variety. Treat it as a floor, not a target.

## 2. Diversity matrix

Track coverage live in `data_collection/trackers/participants.csv` and `sessions.csv`. Aim for
these minimum shares per cell across the 160 drapes:

| Axis | Minimum coverage |
|---|---|
| Saree darkness | light ≥ 30%, dark ≥ 30% |
| Pattern | plain, printed, woven border, checked/striped: each ≥ 15% |
| Fabric | cotton, silk, chiffon/georgette (translucent), synthetic: each ≥ 15%; net/sheer ≥ 5% |
| Lighting | L1 normal 50%; L2 split evenly across dim, backlit and warm/coloured |
| Body | varied height, build and skin tone. Do not record weight or ethnicity, only what the participant volunteers on the info sheet |
| Experience | ≥ 30% of participants who rarely drape a saree. Their natural mistakes are the most valuable data |

**Saree pool** *(assumption)*: assemble about 24 sarees that fill the matrix, so you aren't
relying on what participants bring. Each participant drapes two: **A**, the lighter or plainer
one, and **B**, the darker or patterned one. Record each saree's attributes once in
`sarees.csv` with an ID such as `S07`.

## 3. Ethics, consent and legal

This is the highest-risk part of the project. Participants are filmed in a blouse and petticoat
while dressing.

* **Ethics approval first.** If this runs through a university, submit to its Institutional
  Ethics Committee (IEC) before recruiting anyone. Use the info sheet and consent form in
  `data_collection/` as attachments. Under India's DPDP Act 2023, consent must be free, specific,
  informed and withdrawable. I am not a lawyer; have the IEC or a legal adviser confirm the forms.
* **Adults only** (18+), confirmed on the consent form.
* **Separate consent tick-boxes:** (a) recording, (b) use for training and evaluating this
  project's models, (c) showing frames in reports or papers (face blurred), (d) keeping the data
  after the project ends. Each can be declined independently. A "no" to (c) or (d) must still
  allow participation.
* **Withdrawal at any time**, without giving a reason, up to the date of the next dataset
  version release (deadline stated on the form). Process: delete that participant's
  files, re-run `dataset_version.py`, and retrain any model whose registry entry lists the old
  version.
* **Recording environment:** a private room. A female operator by default, or the
  participant's choice of operator. The participant changes into blouse and petticoat
  **off-camera**, and recording starts only once they say so. They can pause or stop at any point.
* **Compensation** *(assumption)*: a fixed amount per session, paid even if the participant
  stops early.

## 4. Privacy and storage (same rule as the app: nothing goes to the cloud)

* Raw video goes only to an **encrypted external drive** (BitLocker) and the recording laptop. No
  cloud drives, no messaging apps, no hosted annotation platforms.
* Annotation runs in **self-hosted CVAT or Label Studio** on a local machine or the lab LAN.
  Turn off any "AI assist" feature that calls a hosted model. CVAT's SAM assistant may be used
  only if it runs locally (nuclio on the same machine).
* Store the **consent register** (names, signatures) separately from the data, on paper or in a
  separate encrypted file. Data files carry only the pseudonymous `participant_id` (`P001`…) and
  the `consent_id` (`C-2026-001`…).
* **Access list:** name the people who can open raw video. Annotators see sampled frames, not
  whole sessions.
* **Retention:** delete raw video after annotation plus N months *(decision)*; keep annotated
  frames only where consent (d) was given.
* **Faces:** stored unblurred in the restricted raw set, because pose models need the head. Any
  frame shown outside the team is face-blurred.

## 5. Recording setup

* **Room:** at least 3 m of clear depth, plain-ish wall, floor visible (the hem check needs the
  feet).
* **Cameras** (see `protocol.yaml`):
  * **front:** a typical 720p USB webcam at chest height, 2.2–2.6 m away. Use the same class of
    camera the app will use, so the training data matches deployment.
  * **front_left:** a 1080p camera at 30°, eye height.
  * Both connect to one laptop so `record_session.py` timestamps the streams together.
* **Lighting** L1/L2 as defined in `protocol.yaml`. Photograph the setup once per session for the
  record.
* **Floor marks:** tape a standing spot so framing is consistent. Before starting, the operator
  checks the live preview: head and feet are in frame for both cameras.

## 6. Session script (~75–90 min)

1. Welcome, info sheet, consent, questions (10 min). Enter the session in `sessions.csv`.
2. The participant changes off-camera. Run the camera check with `record_session.py --preview`.
3. **Pose clip** (2 min): the participant follows on-screen prompts (turns, arms up, bend, step
   out and back in).
4. **Drapes d1–d4**, following the order in `protocol.yaml`:
   * **correct drapes** (d1, d3): the participant drapes naturally. The operator presses the
     stage key (`1`–`6`) as each stage begins. This gives stage labels almost for free; they
     are reviewed later in `label_video_steps.py`.
   * **mistake drapes** (d2, d4): the recorder shows the next scripted mistake when its stage
     arrives. The participant performs and holds it for about 20 s. The operator presses `m`
     once the mistake is in place and `m` again when it has been corrected, which marks the
     interval. If a mistake isn't performed, `n` skips it and the save report lists it as not
     recorded.
   * Afterwards, `label_video_steps.py --marks <segment>/markers.json` preloads these stage
     marks and mistake intervals for review, then samples frames into the dataset with
     `step_label` and `correction_labels` filled in.
   * The participant re-drapes or partly undrapes between drapes. Untimed breaks.
5. Debrief, compensation, and a reminder of how to withdraw.

## 7. Annotation plan and effort

| Task | Tool | Volume (Phase 2) | Est. min/item | Person-hours |
|---|---|---|---|---|
| Stage boundaries, reviewing live marks | `label_video_steps.py --marks` | 160 drapes | 6 | 16 |
| 7-class masks | CVAT polygons/brush (local SAM if available) | 6,000 frames | 5 | 500 |
| Landmarks on mask frames | CVAT skeleton, pre-filled by the baseline then corrected | 6,000 | 1 | 100 |
| Pleat ROI + fold lines + quality + secured | CVAT box + polylines + attributes | 2,500 | 2 | 83 |
| Correction labels | seeded from mistake marks, reviewed | 6,000 | 0.3 | 30 |
| Second-annotator review (25%) | CVAT review mode | — | — | ~180 |
| **Total** | | | | **~900** |

These per-item times are **guesses**. The pilot measures them: annotators log time per batch
in `annotation_log.csv`. Re-plan Phase 2 from the measured numbers. If masks take more than 8
min/frame, cut the mask target to 4,000 frames and rely more on stage and pleat labels.

**Frame sampling for masks:** take about 150 frames per participant from the front camera,
stratified by stage, with about 30% from mistake intervals. Drop near-duplicates where the frame
difference is below a threshold. Add about 20% from the front_left camera for angle diversity.

**Quality control:** 5% of frames are annotated twice. Report inter-annotator mIoU in the dataset
version notes; below 0.80, retrain the annotators on `DATASET.md`. Run `validate_dataset.py
--strict` before any `dataset_version.py`.

## 8. Timeline *(assumption: 2 operators, 3 annotators part-time)*

| Weeks | Work |
|---|---|
| 0–3 | Ethics submission and approval, buy the saree pool, rehearse the setup with a team member (not in the dataset) |
| 4–5 | **Phase 1 pilot:** 8 sessions, annotate 300 frames, measure times, fix the protocol |
| 6 | Go/no-go review (§10). Freeze `protocol.yaml` v1 |
| 7–12 | **Phase 2:** 32 more sessions (about 6 per week), annotation running in parallel |
| 13–16 | Finish annotation and review; release `real_v1-<hash>`; first real training run |

## 9. Budget lines *(fill in your local costs)*

Saree pool (about 24) · participant compensation (40 sessions) · 2 webcams + 2 tripods + 2
lamps + warm/coloured bulbs · encrypted 2 TB drive · annotator hours (about 900 h, §7) · room
booking.

## 10. Pilot go/no-go criteria

* Every pilot segment produced both camera files with per-frame timestamps, and the save report
  from `record_session.py` passed for each one:
  * dropped frames below 2% per camera. Gaps from an operator pause are not counted.
  * the 95th percentile of the offset between each front frame and the nearest front_left frame
    is below 50 ms.
  * the measured fps is close to the configured 30.

  In a test with video files standing in for cameras, the recorder measured 30.0 fps, 0%
  dropped and a 12.8 ms p95 offset. Real USB webcams still need checking during the pilot.
* In 95% or more of frames, both ankles and the head are visible in the front camera.
* Each scripted mistake was performed recognisably. The reviewer agrees with the mark in at
  least 90% of cases.
* Measured annotation times are within 1.5× of §7, or the plan has been re-scoped.
* No participant reported discomfort with the process. Revise the process if anyone did.

## 11. Decisions needed from you

1. **Institution and ethics committee:** which IEC will review this, or is this an independent
   project needing another route?
2. **Who records:** operators, and the default operator gender.
3. **Compensation amount** and **budget ceiling.**
4. **Retention period** for raw video.
5. **Whether frames may ever be published**, which decides whether consent box (c) is needed.
6. **Saree pool:** buy, borrow, or participant-supplied only.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Too few participants volunteer | Recruit through women's groups and saree-draping classes; allow participants to bring a friend; keep sessions short |
| Annotation takes far longer than planned | Pilot measurement (§7); local SAM-assisted masks; fall back to fewer mask frames |
| The dataset is too homogeneous (one body type, few fabrics) | Diversity matrix tracked weekly; targeted recruitment |
| Consent is withdrawn after training | Per-version dataset hashes plus registry links make the retrain traceable |
| Raw data leaks | Encrypted drive, access list, no cloud tools, pseudonymous IDs |

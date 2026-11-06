# Dataset: acquisition, schema, labelling standard, splits, versioning

## 1. Legal and ethical acquisition (mandatory)

* **Only consented recordings.** Every real participant signs a written consent covering:
  recording, annotation, use for training and evaluation of *this* project's models, retention
  period, and withdrawal. Store the signed forms **outside** the dataset; the dataset holds only a
  pseudonymous `participant_id` and a `consent_id` reference.
* **No scraping.** Publicly visible photos (Instagram, Pinterest, shopping sites, YouTube) are
  **not** usable without explicit permission from the rights holder *and* the person shown.
* Third-party datasets may be used only if their licence permits model training (record it in
  `source.license`). COCO-Keypoints annotations are CC BY 4.0; COCO images keep individual Flickr
  licences.
* `validate_dataset.py` **fails** any real sample without a `consent_id`.
* Withdrawal: delete that participant's rows + files, re-version the dataset, retrain.

## 2. Coverage targets (what to collect)

| Axis | Minimum coverage for a first real release |
|---|---|
| Participants | ≥ 40 distinct people (≥ 8 held out for test), varied body shapes and skin tones |
| Sarees | ≥ 60 sarees: light / medium / dark; plain, printed, woven-border, checked, embroidered |
| Fabrics | cotton, silk, chiffon/georgette (translucent), synthetic, net (sheer) |
| Lighting | bright daylight, normal indoor, dim, backlit (window behind), coloured/warm bulbs |
| Camera | front, front-left/right (~30°), side; chest and eye height; 480p–1080p webcams and phones |
| Body position | standing, bending to pleat, turning, arms raised (pallu) |
| Stages | every stage in `constants.DRAPE_STAGES`, recorded as **continuous videos** of full drapes |
| Errors | deliberately performed mistakes for every `CORRECTION_LABELS` entry (uneven/slanted pleats, off-centre, untucked, pallu on wrong shoulder, uneven hem, loose wrap), ≥ 150 frames each |

Recommended protocol per participant: 3 complete Nivi drapes (one correct, two with scripted
mistakes) × 2 sarees × 2 lighting setups, filmed with 2 cameras simultaneously.

Estimated annotation volume for v1: ~6,000 frames with segmentation masks (sampled ~1 fps from
videos, near-duplicates dropped), ~2,500 pleat crops with fold lines/quality, all frames with
step labels (cheap: done per video segment with `label_video_steps.py`).

## 3. Schema

One JSON object per line in `<dataset>/manifest.jsonl`, defined formally in
[`saree_ai/schema/sample.schema.json`](../saree_ai/schema/sample.schema.json). Key fields:

| Field | Content |
|---|---|
| `image`, `width`, `height` | relative image path and size |
| `participant_id`, `video_id`, `frame_index` | grouping used for splits and temporal models |
| `source.type` | `real` or `synthetic` (never mixed within a participant) |
| `source.consent_id` | required for real data |
| `capture.*`, `saree.*` | lighting, camera angle, colour darkness, fabric, pattern, transparency |
| `mask` | single-channel PNG; value = class index, 255 = ignore |
| `regions` | alternative polygon form (rasterised by `prepare_dataset.py rasterize`) |
| `landmarks` | 17 COCO keypoints `[x, y, v]` |
| `pleats` | `visible`, `roi`, `fold_lines`, `count`, `count_certain`, `quality`, `secured` |
| `step_label`, `step_label_certain` | draping stage |
| `correction_labels` | zero or more correction ids |

## 4. Labelling standard

**Segmentation classes** (draw order: later classes overwrite earlier ones):
`person` (skin, hair) → `other_clothing` (blouse, petticoat, anything not the saree) →
`saree_body` → `waist_wrap` → `pleats` → `pallu`.

* `waist_wrap`: the horizontal band of saree at the waist, from top edge (tucked line) down to
  ~1 hand-width.
* `pleats`: the whole front pleat stack, from where it is held/tucked to the hem. One region, not
  one per pleat (individual folds are captured as `fold_lines`).
* `pallu`: saree fabric from where it leaves the second waist wrap up across the torso and over
  the shoulder, including what hangs over the shoulder.
* `saree_body`: all other saree fabric (skirt part, partial wraps).
* **Hands** covering fabric are labelled `person` (we model occlusion honestly); tick
  `occluded_by_hand` on the occluded region.
* **Sheer fabric**: if the saree is visible as fabric (colour/texture), label it as saree even if
  the blouse/skin shows through; if you genuinely cannot tell, paint `255` (ignore).
* Boundaries: within ~3 px is acceptable; paint uncertain bands 255 rather than guessing.

**Landmarks**: COCO definitions. `v=2` visible, `v=1` occluded but position confidently
inferable, `v=0` not labelled. Do not guess hidden ankles under a floor-length saree — use `v=1`
only if the shape makes the position clear.

**Pleats**: `roi` = tight box around the pleat stack. `fold_lines` = one polyline per visible fold
edge, top to bottom, left to right. `count` = number of folds (pleats) visible from the camera;
`count_certain=false` when they overlap too much to count. `quality`: `neat` | `uneven_spacing`
(width variation clearly > ~30%) | `misaligned` (folds slanted / top edges not aligned) |
`loose_or_unformed`. `secured`: tucked at waist (true) vs held/loose (false).

**Step labels**: the stage the drape has *reached* in that frame (see `nivi.yaml` comments).
Frames within 0.5 s of a stage change get `step_label_certain=false` and are excluded from
stage training/eval.

**Correction labels**: tag only errors that are visible in that frame from that camera. A
reviewer must agree before `review_status=reviewed`. Disagreements → `disputed` (excluded from
evaluation, reported).

Every image is annotated by one annotator and reviewed by a second; report inter-annotator IoU
on a 5% overlap set in the dataset version notes.

## 5. Splits

`training/data/prepare_dataset.py split` assigns **participants**, not frames, to train / val /
test. Synthetic participants are only placed in train (or val when there is no real val data);
the script refuses to put synthetic data in test and leaves test **empty** rather than testing
on synthetic data when there are fewer than 5 real participants.

## 6. Versioning

`training/data/dataset_version.py <dir> --name <name>` hashes the manifest and every referenced
file into `data/versions/<name>-<hash>.json`. Put that id into the training config
(`data.versions`); `export_onnx.py` copies it into the model registry entry, so every registered
model traces back to the exact bytes it was trained on.

## 7. Synthetic data

`training/data/synth_generate.py` renders flat procedural figures with textured "fabric" layers
for every stage, with matching masks, landmarks, pleat folds, step labels and correction labels.
It exists to exercise the pipeline and for optional pre-training. It looks nothing like real
fabric and **no metric computed on it is evidence of real-world accuracy.**

## 8. Annotation tools (all self-hostable / local)

* Regions, landmarks, pleat boxes and fold polylines: **CVAT** (MIT, self-hosted via Docker) or
  **Label Studio** (Apache-2.0, local). Export COCO → `convert_coco.py`.
* Stage labels for whole videos: `training/data/label_video_steps.py` (local OpenCV tool).

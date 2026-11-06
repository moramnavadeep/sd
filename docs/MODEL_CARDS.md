# Model cards

Live numbers (dataset sizes, metrics, latency, status) for each trained version are in
`models/registry.json` and `runs/<task>/<stamp>/eval_test.md`. This page documents design and
requirements. **No model in this repository has yet been trained on real data.** See
`docs/STATUS_REPORT.md`.

| | A: pose (baseline) | A′: pose (own) | B: saree segmentation | C: pleat recognition | D: draping stage | E: correction engine |
|---|---|---|---|---|---|---|
| Ours? | **No** – pretrained third party | Yes | Yes | Yes | Yes | Yes (rules) |
| From scratch / fine-tuned | pretrained | ImageNet backbone → fine-tuned | `lraspp_mobilenet_v3`: ImageNet backbone, head from scratch; `tiny_unet`: from scratch | from scratch | from scratch | n/a (rules); learned replacement later |
| Architecture | Keypoint R-CNN R50-FPN (torchvision) | SimpleBaseline heatmap head on MobileNetV3-L, 256×192 | LR-ASPP MobileNetV3-L 384×256 (full) / TinyUNet 256×160 (lite) | 7-layer CNN, column pooling, 4 heads (count, quality, secured, visible) on 128² crop + signal analyzer | MLP 23→64→64→7 on interpretable features + HMM filter | per-check rules + persistence + confidence gates |
| Data needed | – | ≥ 5k annotated person instances (own + COCO) | ~6k masked frames, ≥ 40 participants | ~2.5k pleat crops w/ count, quality, secured, fold lines | step-labelled frames from ≥ 40 full drape videos | frames with reviewed `correction_labels` |
| Annotation | – | 17 COCO keypoints | 7-class masks | roi, fold_lines, count, quality, secured | step_label | correction_labels |
| Loss / training | – | weighted heatmap MSE, AdamW, cosine | weighted CE (1/√freq), AdamW, OneCycle, AMP | BCE(visible)+0.2·SmoothL1(count)+CE(quality)+BCE(secured) | class-weighted CE + feature noise | – |
| Evaluation | PCK@0.2 on our test participants | PCK@0.2 | per-class P/R/F1/IoU, mIoU, saree mIoU | quality P/R/F1, count MAE & ±1 acc, secured P/R/F1 | accuracy, macro P/R/F1, confusion | per-label P/R/F1, **false-positive correction rate**, low-confidence probe |
| Acceptance targets | – | PCK ≥ 0.90 | saree mIoU ≥ 0.60, macro F1 ≥ 0.70 | quality F1 ≥ 0.70, count±1 ≥ 0.80, secured F1 ≥ 0.85 | macro F1 ≥ 0.80, acc ≥ 0.85 | FPR ≤ 5%, low-confidence probe = 0 corrections |
| Hardware (train) | – | GPU ≥ 6 GB, ~6–10 h | GPU ≥ 6 GB, ~1–2 h for 6k images | CPU or GPU, minutes | CPU, seconds | – |
| Deployment | PyTorch (FP16 on CUDA) | ONNX / ORT | ONNX / ORT | ONNX / ORT + NumPy | ONNX / ORT | Python |

## Notes per model

**A – pose baseline.** Origin: `torchvision.models.detection.keypointrcnn_resnet50_fpn`,
`COCO_V1` weights from download.pytorch.org (sha256 prefix fc266e95). torchvision code is
BSD-3-Clause; the weights were trained by the PyTorch team on COCO 2017. We do not claim
ownership. Keypoint scores are heatmap logits; we squash them with `sigmoid(s − 1)` – an
**uncalibrated** mapping. Calibrate on our own annotated test set (Platt scaling) before relying on
the 0.55 threshold.

**A′ – own pose.** The pathway to independence: (1) collect landmark annotations on our own
consented videos, (2) optionally add COCO-Keypoints, (3) optionally pseudo-label our own videos
with A and have a human review a sample (record this as distillation in the registry), (4) train
with `training/train_pose.py`, (5) once PCK on held-out participants is within 2 points of A,
set `pose.backend: own`.

**B – segmentation.** Distinguishes saree vs background vs other clothing vs skin, and the
saree's parts. Ambiguity (sheer fabric, hand occlusion) is surfaced via the per-pixel max-softmax:
when > 35% of saree pixels are below 0.5, dependent checks become "uncertain" and the user is
asked to confirm manually.

**C – pleats.** The signal analyzer is interpretable but fragile on printed fabric; it reports
low confidence when the column-gradient profile isn't periodic. The CNN learns count/quality
from data. When they disagree on count, confidence is reduced so the engine stays quiet.

**D – stage.** Uses the same feature function at training (on ground truth) and inference (on
predictions); expect a domain gap until features are recomputed from model predictions. Without a
trained model, a capped heuristic (max probability 0.55, below the 0.6 threshold) keeps the UI
informative but never auto-completes a step.

**E – correction engine.** Rule set in `saree_ai/correction/checks.py`, tutorial in
`config/tutorials/nivi.yaml`. Measured by `training/evaluate_corrections.py`.

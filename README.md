# FrameDiff

**Learning When to Recompute for Efficient Video Object Detection**

FrameDiff is a small CNN "gate" that learns when a video object detector actually needs to re-run versus when it's safe to reuse the previous frame's bounding boxes. Trained and evaluated on the [UA-DETRAC](https://detrac-db.rit.albany.edu/) traffic dataset with YOLOv8m, the gate cuts average inference latency by **~2.2x** relative to running YOLO on every frame, while holding onto far more detection quality than naive frame-skipping.

This was built as the final project for CAP4410 (Computer Vision) at Florida Polytechnic University. The full write-up is in [`CAP4410_Final_Project.pdf`](./CAP4410_Final_Project.pdf).

Authors: [Wayne Hall](mailto:whall3063@floridapoly.edu), Shriraj Mandulapalli — instructor Dr. Muhammad Abid.

## The idea

Running a heavy detector (YOLO) on every single frame of a video is wasteful — in a lot of footage, nothing meaningfully changes from one frame to the next. But skipping frames on a fixed schedule (e.g. "run YOLO every 10 frames") is a blunt instrument: it stays cheap during motion and occlusion, when boxes actually go stale, and it stays needlessly careful during static scenes.

Instead of a fixed schedule, FrameDiff frames this as a learned binary decision per frame pair: **recompute** (run YOLO again) or **keep** (reuse the last known boxes). The gate never sees bounding boxes — only two cheap, pixel-level motion cues:

- **Optical flow magnitude** (Farneback) between the two frames
- **Gaussian-blurred absolute pixel difference** between the two frames

These are stacked into a 2-channel tensor and fed to a compact CNN that outputs a single logit: recompute or keep.

### How the training labels are generated (no manual annotation)

1. Run YOLOv8m on every frame of every video to get pseudo-ground-truth vehicle boxes (car, motorcycle, bus, truck).
2. Form frame pairs at several temporal offsets (Δ = 1, 3, 5, 7, 11, 13 — primes, to avoid periodic aliasing).
3. Match boxes between the two frames of a pair one-to-one via the Hungarian algorithm on pairwise IoU.
4. Label the pair `keep` if mean matched IoU ≥ τ, else `recompute`. τ = 0.70 was chosen as the best balance between over-triggering `keep` at small Δ and never `keep`ing at large Δ.
5. Split by **video**, not by row, into train/val/test so no frame pair leaks across splits.

The gate is then trained (BCE-with-logits, Adam, step-decayed LR, 15 epochs, best checkpoint by validation loss) to reproduce that recompute/keep decision from the motion cues alone.

## Results

| Method | Avg latency (ms) | Mean IoU |
|---|---|---|
| Full YOLO (every frame) | 19.94 | 1.000 |
| Fixed skip (k = 10) | 1.92 | 0.723 |
| **FrameDiff (learned gate)** | **8.98** | **0.831** |

The learned gate is ~2.2x faster than running YOLO on every frame, and keeps meaningfully better box quality than naive fixed-interval skipping.

| Metric | Test | Validation |
|---|---|---|
| AUC | 0.9446 | 0.9640 |
| Precision | 0.8385 | 0.8313 |
| Recall | 0.8907 | 0.9271 |
| F1 | 0.8639 | 0.8766 |

Gate accuracy is highest at small (Δ=1) and large (Δ=13) temporal gaps, and dips in the middle — the decision is easiest when almost nothing has changed or when a lot clearly has, and hardest in between. See the full paper for the per-Δ breakdown and discussion.

## Pipeline

The project is organized as a sequence of numbered stages under [`src/`](./src), each reading the previous stage's output:

| Stage | Script | Purpose |
|---|---|---|
| 1 | [`stage1_boxes.py`](src/stage1_boxes.py) | Run YOLOv8m on a directory of frames → per-frame YOLO label files + per-video JSON manifests. |
| 2 | [`stage2_pair.py`](src/stage2_pair.py) | Build temporally-offset frame pairs, Hungarian-match detections between them, compute IoU-based stability metrics. Writes a checkpointed Parquet manifest. |
| 3 | [`stage3_labels.py`](src/stage3_labels.py) | Threshold mean IoU into `recompute`/`keep` labels; split into train/val/test by video (stratified). |
| 4 | [`stage4_dataloader.py`](src/stage4_dataloader.py) | PyTorch `Dataset`/`DataLoader`s that turn a labeled pair into a 2-channel (optical flow + blurred diff) tensor. |
| 5 | [`stage5_model.py`](src/stage5_model.py) | `FrameDiffModel` — the 4-block CNN gate architecture. |
| 6 | [`stage6_train.py`](src/stage6_train.py) | Training loop; saves the best checkpoint by validation loss. |
| 7 | [`stage7_validation.py`](src/stage7_validation.py) | Detailed evaluation: per-Δ accuracy, ROC/AUC, precision/recall/F1, on val and test splits. |
| 8 | [`stage8_latency.py`](src/stage8_latency.py) | Benchmarks Full-YOLO vs. fixed-skip vs. learned-gate latency and mean IoU on the test videos. |

Supporting pieces:

- [`demo/`](./demo) — a standalone demo pipeline that runs all three strategies (full YOLO, fixed skip, learned gate) on a single video and renders annotated comparison videos.
- [`notebooks/`](./notebooks) — exploratory notebooks (optical flow / blur visualization, YOLO + Parquet evaluation, demo video generation).
- [`configs/`](./configs) — one YAML config per stage (paths, thresholds, hyperparameters).
- [`checkpoints/best_model/`](./checkpoints/best_model) — YOLOv8m fine-tuning artifacts (training curves, confusion matrix, sample predictions).

## Setup

Requires Python 3.12+. Dependencies are managed with [`uv`](https://docs.astral.sh/uv/) via `pyproject.toml`.

```bash
uv sync
```

Or with plain `pip`:

```bash
pip install flask jupyter pandas pyarrow scikit-learn torch ultralytics opencv-python scipy
```

You'll also need `ffmpeg` on your `PATH` if you want the demo scripts to stitch frames into video.

## Usage

Each stage reads its paths from a config in `configs/`. Point the config at your own UA-DETRAC frames directory before running.

```bash
# 1. Detect vehicles in every frame with YOLO
python src/stage1_boxes.py --frames-dir data/frames --weights models/yolov8m.pt --out-dir data/gt_dataset

# 2. Build frame pairs + IoU stability metrics
python src/stage2_pair.py

# 3. Threshold into recompute/keep labels + train/val/test split
python src/stage3_labels.py

# 4-6. Train the gate (dataloaders are built internally from configs/stage_6_config.yaml)
python src/stage6_train.py

# 7. Evaluate the trained gate
python src/stage7_validation.py

# 8. Benchmark latency/IoU: full YOLO vs. fixed-skip vs. learned gate
python src/stage8_latency.py
```

To render a side-by-side comparison video for one clip, configure `configs/demo_config.yaml` and run:

```bash
python demo/demo.py
python demo/make_vid.py
```

## Limitations & future work

- Labels are derived entirely from YOLO's own predictions — if YOLO is systematically wrong on a frame, the gate can learn to reproduce that failure rather than reason about true motion.
- Only trained on vehicle classes (car, motorcycle, bus, truck) from traffic-camera footage; it doesn't see pedestrians or other object types.
- Natural extensions: other object domains (people, aircraft, ships, trains), and motion-aware sampling of training pairs to better balance easy vs. hard motion regimes.

See Section V of the [paper](./CAP4410_Final_Project.pdf) for full discussion.

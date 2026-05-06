import cv2
import yaml
import torch
import time
import glob
import os
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import numpy as np
import shutil
import sys
import subprocess

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open("configs/demo_config.yaml", "r") as f:
    demo_config = yaml.safe_load(f)

YOLO_MODEL_PATH      = demo_config["model"]["yolo_model_path"]
FRAMEDIFF_MODEL_PATH = demo_config["model"]["framediff_model_path"]
DEMO_VIDEO_NAME      = demo_config["data"]["demo_video_name"]
FRAMES_PATH          = demo_config["data"]["frames_folder"]
SRC_PATH             = demo_config["modules"]["src_path"]
SKIP_K_FRAMES        = demo_config["output"]["skip_k_frames"]
LEARNED_SKIP_FRAMES  = demo_config["output"]["learned_skip_frames"]
FULL_COMPUTE_FRAMES  = demo_config["output"]["full_compute_frames"]
VIDEO_OUTPUT         = demo_config["output"]["video_output"]

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
def _reset_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

for _d in (SKIP_K_FRAMES, LEARNED_SKIP_FRAMES, FULL_COMPUTE_FRAMES, VIDEO_OUTPUT):
    _reset_dir(_d)

sys.path.append(SRC_PATH)
from stage5_model import FrameDiffModel

# ---------------------------------------------------------------------------
# Running mean (ignores first `warmup` values)
# ---------------------------------------------------------------------------
class RunningMean:
    def __init__(self, warmup: int = 3):
        self.warmup = warmup
        self.n = 0
        self.total = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        if self.n > self.warmup:
            self.total += value

    @property
    def mean(self) -> float:
        count = max(self.n - self.warmup, 0)
        return self.total / count if count > 0 else 0.0
    
class RunningIoU:
    def __init__(self, warmup: int = 3):
        self.warmup = warmup
        self.n = 0
        self.total_iou = 0.0

    def update(self, iou: float) -> None:
        self.n += 1
        if self.n > self.warmup:
            self.total_iou += iou

    @property
    def mean(self) -> float:
        count = max(self.n - self.warmup, 0)
        return self.total_iou / count if count > 0 else 0.0

# ---------------------------------------------------------------------------
# Device / model setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = device.type == "cuda"

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

yolo = YOLO(YOLO_MODEL_PATH).to(device)

framediff = FrameDiffModel()
framediff.load_state_dict(torch.load(FRAMEDIFF_MODEL_PATH, map_location=device))
framediff.to(device).eval()

# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------
def load_video_frames(video_name: str) -> list[str]:
    return sorted(glob.glob(os.path.join(FRAMES_PATH, f"{video_name}_img*.jpg")))

def frame_to_gpu(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC → normalised RGB NCHW tensor on device."""
    return (
        torch.from_numpy(frame_bgr)
        .to(device, non_blocking=True)
        .flip(-1)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div_(255.0)
    )

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_boxes(frame: np.ndarray, boxes: np.ndarray, is_fresh: bool) -> None:
    if len(boxes) == 0:
        return
    color     = (0, 255, 0) if is_fresh else (0, 140, 0)
    thickness = 2            if is_fresh else 1
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

def add_freshness_dot(frame: np.ndarray, is_fresh: bool) -> None:
    h, w   = frame.shape[:2]
    center = (w - 25, 25)
    color  = (0, 255, 0) if is_fresh else (60, 60, 60)
    cv2.circle(frame, center, 10, color, thickness=-1)

def add_bottom_strip(frame: np.ndarray, lines: list[str], strip_height: int = 90) -> np.ndarray:
    h, w  = frame.shape[:2]
    strip = np.zeros((strip_height, w, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        y = 25 + i * 25
        cv2.putText(strip, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([frame, strip])

def compute_iou(box_a, box_b):
    """IoU between two boxes in corner format [x1, y1, x2, y2]."""
    if len(box_a) < 4 or len(box_b) < 4:
        print(f"Error: Box A {box_a} or Box B {box_b} does not have 4 coordinates.")
        return 0.0

    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def match_detections(detections_a, detections_b, iou_threshold=0.5, min_iou=1e-6):
    if len(detections_a) == 0 or len(detections_b) == 0:
        return 0, 0, 0.0, 0.0  # No matches if either list is empty

    iou_matrix = np.zeros((len(detections_a), len(detections_b)))
    for i, det_a in enumerate(detections_a):
        for j, det_b in enumerate(detections_b):
            iou_matrix[i, j] = compute_iou(det_a, det_b)

    # linear_sum_assignment on -iou_matrix maximizes the sum of IoUs over the optimal one-to-one assignment
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_ious = iou_matrix[row_ind, col_ind]
    real = matched_ious > min_iou
    matched_ious = matched_ious[real]

    n_matched = int(real.sum())
    n_matched_well = int((matched_ious >= iou_threshold).sum())
    mean_iou = float(matched_ious.mean()) if n_matched > 0 else 0.0
    match_quality = n_matched_well / n_matched if n_matched > 0 else 0.0

    return n_matched, n_matched_well, mean_iou, match_quality

# ---------------------------------------------------------------------------
# Timing helper (unified for CPU and CUDA)
# ---------------------------------------------------------------------------
def _sync() -> None:
    if USE_CUDA:
        torch.cuda.synchronize()

# ---------------------------------------------------------------------------
# Optical-flow / blur-diff helpers
# ---------------------------------------------------------------------------
def _calc_flow_magnitude(gray_a: np.ndarray, gray_b: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag.astype(np.float32)

def _blur_difference(gray_a: np.ndarray, gray_b: np.ndarray,
                     ksize: int = 5, sigma: float = 1) -> np.ndarray:
    ba = cv2.GaussianBlur(gray_a, (ksize, ksize), sigma)
    bb = cv2.GaussianBlur(gray_b, (ksize, ksize), sigma)
    return cv2.absdiff(ba, bb).astype(np.float32)

# ---------------------------------------------------------------------------
# Core inference: run YOLO and return filtered boxes + elapsed ms
# ---------------------------------------------------------------------------
def _run_yolo(frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Run a single YOLO forward pass and return (vehicle_boxes_xyxy, ms)."""
    _sync()
    t0 = time.perf_counter()
    results = yolo(frame_bgr, conf=0.25, verbose=False)[0]
    _sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    boxes = results.boxes.xyxy.cpu().numpy()
    mask  = np.isin(results.boxes.cls.cpu().numpy(), list(VEHICLE_CLASSES.keys()))
    return boxes[mask], elapsed_ms

# ---------------------------------------------------------------------------
# Annotate + save a frame
# ---------------------------------------------------------------------------
def _annotate_and_save(
    frame_bgr: np.ndarray,
    boxes: np.ndarray,
    is_fresh: bool,
    label_lines: list[str],
    save_path: str,
) -> None:
    out = frame_bgr.copy()
    draw_boxes(out, boxes, is_fresh=is_fresh)
    add_freshness_dot(out, is_fresh=is_fresh)
    out = add_bottom_strip(out, label_lines)
    cv2.imwrite(save_path, out)

# ---------------------------------------------------------------------------
# Strategy 1 — always full YOLO ("baseline")
# ---------------------------------------------------------------------------
global gt_boxes
gt_boxes = None  # For demo annotation
def run_full_yolo(
    frame_bgr: np.ndarray,
    frame_idx: int,
    latency: RunningMean,
    save_dir: str = FULL_COMPUTE_FRAMES,
) -> tuple[np.ndarray, float]:
    boxes, t_ms = _run_yolo(frame_bgr)
    latency.update(t_ms)
    _annotate_and_save(
        frame_bgr, boxes, is_fresh=True,
        label_lines=[
            "Always YOLO",
            f"Avg latency: {latency.mean:.2f} ms",
            f"This frame: {t_ms:.2f} ms",
        ],
        save_path=os.path.join(save_dir, f"full_{frame_idx:05d}.jpg"),
    )
    global gt_boxes
    gt_boxes = boxes
    return boxes, latency.mean

# ---------------------------------------------------------------------------
# Strategy 2 — skip every K frames
# ---------------------------------------------------------------------------
class SkipKState:
    """Mutable state for the skip-K strategy."""
    __slots__ = ("last_frame_bgr", "last_boxes")

    def __init__(self) -> None:
        self.last_frame_bgr: np.ndarray | None = None
        self.last_boxes: np.ndarray | None = None


def run_skip_k(
    frame_bgr: np.ndarray,
    frame_idx: int,
    latency: RunningMean,
    IoU: RunningIoU,
    state: SkipKState,
    k: int = 10,
) -> tuple[np.ndarray, float]:
    is_keyframe = (state.last_frame_bgr is None) or (frame_idx % k == 0)

    if is_keyframe:
        boxes, t_ms = _run_yolo(frame_bgr)
        latency.update(t_ms)
        state.last_frame_bgr = frame_bgr
        state.last_boxes = boxes
        IoU.update(1) # Keyframes are "perfect" matches to themselves, so IoU=1.
        _annotate_and_save(
            frame_bgr, boxes, is_fresh=True,
            label_lines=[
                "Skip-K",
                f"Avg latency: {latency.mean:.2f} ms",
                f"This frame: {t_ms:.2f} ms",
            ],
            save_path=os.path.join(SKIP_K_FRAMES, f"skipk_{frame_idx:05d}.jpg"),
        )
        return boxes, latency.mean

    # --- skipped frame ---------------------------------------------------------
    global gt_boxes
    latency.update(0.0)  # Count this frame as 0 ms since we skip YOLO.
    cur_iou = match_detections(state.last_boxes, gt_boxes)[2]  # IoU of last boxes vs GT.
    IoU.update(cur_iou)  # Update IoU for skipped frames.
    avg_iou = IoU.mean
    avg = latency.mean
    _annotate_and_save(
        frame_bgr, state.last_boxes, is_fresh=False,
        label_lines=[
            "Skip-K",
            f"Avg latency: {avg:.2f} ms, Avg IoU: {avg_iou:.3f}",
            f"This frame: 0 ms, IoU: {cur_iou:.3f} (skipped)",
        ],
        save_path=os.path.join(SKIP_K_FRAMES, f"skipk_{frame_idx:05d}.jpg"),
    )
    return state.last_boxes, avg

# ---------------------------------------------------------------------------
# Strategy 3 — learned skip via FrameDiff
# ---------------------------------------------------------------------------
class LearnedSkipState:
    """Mutable state for the learned-skip strategy."""
    __slots__ = ("last_frame_bgr", "last_gray", "last_boxes")

    def __init__(self) -> None:
        self.last_frame_bgr: np.ndarray | None = None
        # PERF FIX: cache the greyscale of the previous keyframe so we don't
        # recompute it on every frame.
        self.last_gray: np.ndarray | None = None
        self.last_boxes: np.ndarray | None = None

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

global skips
skips = 0

def run_learned_skip(
    frame_bgr: np.ndarray,
    frame_idx: int,
    latency: RunningMean,
    IoU: RunningIoU,
    state: LearnedSkipState,
) -> tuple[np.ndarray, float]:
    global skips
    global gt_boxes
    current_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Bootstrap: no prior frame yet — always run YOLO.
    if state.last_frame_bgr is None:
        cur_iou = 1.0
        IoU.update(cur_iou)  # First frame is a "perfect" match to itself, so IoU=1.
        boxes, t_ms = _run_yolo(frame_bgr)
        latency.update(t_ms)
        state.last_frame_bgr = frame_bgr
        state.last_gray      = current_gray
        state.last_boxes     = boxes
        _annotate_and_save(
            frame_bgr, boxes, is_fresh=True,
            label_lines=[
                f"Learned Skip Skip Rate: {skips}/{frame_idx+1} ({skips/(frame_idx+1)*100:.1f}%)",
                f"Avg latency: {latency.mean:.2f} ms, Avg IoU: {IoU.mean:.3f}",
                f"This frame: {t_ms:.2f} ms, IoU: {cur_iou:.3f}",
            ],
            save_path=os.path.join(LEARNED_SKIP_FRAMES, f"learned_skip_{frame_idx:05d}.jpg"),
        )
        return boxes, latency.mean

    # --- Build framediff input (uses cached last_gray — no redundant cvtColor) --
    w, h = (128, 128)
    img_a = cv2.resize(current_gray, (w, h), interpolation=cv2.INTER_AREA)
    img_b = cv2.resize(state.last_gray, (w, h), interpolation=cv2.INTER_AREA)
    flow_mag   = _calc_flow_magnitude(state.last_gray, current_gray)
    blur_diff  = _blur_difference(state.last_gray, current_gray)

    mag_norm = np.clip(flow_mag / 30, 0.0, 1.0)
    diff_norm = np.clip(blur_diff / 255.0, 0.0, 1.0)

    x = np.stack([mag_norm, diff_norm], axis=0)
    x_t = torch.from_numpy(np.ascontiguousarray(x)).float().to(device).unsqueeze(0)

    # BUG FIX: synchronize AFTER building the tensor but BEFORE timing,
    # not before the framediff forward pass which breaks CPU timing.
    _sync()
    t0 = time.perf_counter()
    skip_pred = framediff(x_t).item()
    _sync()
    fd_ms = (time.perf_counter() - t0) * 1000

    if sigmoid(skip_pred) > 0.5:
        skips += 1
        # Skipped: count only the framediff cost, not a 0-latency entry.
        cur_iou = match_detections(state.last_boxes, gt_boxes)[2]  # IoU of last boxes vs GT.
        IoU.update(cur_iou)
        latency.update(fd_ms)
        _annotate_and_save(
            frame_bgr, state.last_boxes, is_fresh=False,
            label_lines=[
                f"Learned Skip Skip Rate: {skips}/{frame_idx+1} ({skips/(frame_idx+1)*100:.1f}%)",
                f"Avg latency: {latency.mean:.2f} ms, Avg IoU: {IoU.mean:.3f}",
                f"This frame: {fd_ms:.2f} ms, IoU: {cur_iou:.3f} (skip decision only)"
            ],
            save_path=os.path.join(LEARNED_SKIP_FRAMES,
                                   f"learned_skip_{frame_idx:05d}.jpg"),
        )
        return state.last_boxes, latency.mean

    # Not skipped: run full YOLO (timing restarts to include full pipeline).
    _sync()
    t0 = time.perf_counter()
    boxes, _ = _run_yolo(frame_bgr)
    _sync()
    total_ms = (time.perf_counter() - t0) * 1000 + fd_ms
    latency.update(total_ms)
    cur_iou = 1.0
    IoU.update(cur_iou)  # Update IoU for this frame.
    state.last_frame_bgr = frame_bgr
    state.last_gray      = current_gray
    state.last_boxes     = boxes
    _annotate_and_save(
        frame_bgr, boxes, is_fresh=True,
        label_lines=[
            f"Learned Skip Skip Rate: {skips}/{frame_idx+1} ({skips/(frame_idx+1)*100:.1f}%)",
            f"Avg latency: {latency.mean:.2f} ms, Avg IoU: {IoU.mean:.3f}",
            f"This frame: {total_ms:.2f} ms, IoU: {cur_iou:.3f}",
        ],
        save_path=os.path.join(LEARNED_SKIP_FRAMES,
                               f"learned_skip_{frame_idx:05d}.jpg"),
    )
    return boxes, latency.mean

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
# FIX: rename to avoid shadowing the frame-list variable at the module level.
all_frame_paths = load_video_frames(DEMO_VIDEO_NAME)
print(f"Loaded {len(all_frame_paths)} frames for video '{DEMO_VIDEO_NAME}'")

full_latencies  = RunningMean(warmup=3)
skipk_latencies = RunningMean(warmup=3)
learn_latencies = RunningMean(warmup=3)

skipk_iou = RunningIoU(warmup=3)
learn_iou = RunningIoU(warmup=3)

skipk_state = SkipKState()
learn_state = LearnedSkipState()

for frame_idx, frame_path in enumerate(all_frame_paths):
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Warning: could not read {frame_path}")
        continue

    run_full_yolo(frame, frame_idx, full_latencies)
    run_skip_k(frame, frame_idx, skipk_latencies, skipk_iou, skipk_state)
    run_learned_skip(frame, frame_idx, learn_latencies, learn_iou, learn_state)

# ---------------------------------------------------------------------------
# Compile videos
# FIX: use explicit sorted glob with numeric sort key; rename loop variable.
# ---------------------------------------------------------------------------
output_dirs = {
    FULL_COMPUTE_FRAMES:  ("full",          full_latencies),
    SKIP_K_FRAMES:        ("skipk",         skipk_latencies),
    LEARNED_SKIP_FRAMES:  ("learned_skip",  learn_latencies),
}

for folder, (prefix, lat) in output_dirs.items():
    # Sort by the numeric frame index embedded in the filename.
    jpg_files = sorted(
        glob.glob(os.path.join(folder, "*.jpg")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].rsplit("_", 1)[-1]),
    )
    if not jpg_files:
        print(f"No frames in {folder}, skipping.")
        continue

    out_path = os.path.join(VIDEO_OUTPUT, f"{prefix}_output.mp4")

    # Write a sorted file list for ffmpeg to avoid shell-glob ordering issues.
    list_file = os.path.join(folder, "_filelist.txt")
    with open(list_file, "w") as fh:
        for p in jpg_files:
            fh.write(f"file '{os.path.abspath(p)}'\n")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-framerate", "30",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            out_path,
        ],
        check=True,
    )
    os.remove(list_file)
    print(
        f"Saved {len(jpg_files)} frames → {out_path}  "
        f"(avg latency: {lat.mean:.2f} ms)"
    )

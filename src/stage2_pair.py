"""
Stage 2: Pairing frames based on variable offsets
Input: A directory of .txt files containing YOLO detections for each frame, and a directory of the corresponding frames.
Output: A parquet manifest containing pairs of frames and their detections, where the second frame in each pair is offset from the first by a variable number of frames (e.g., 1, 3, 5, 7, 11, 13).
The manifest will have the following columns:
- video_name: The name of the video (derived from the frame filenames).
- frame_num_a: The frame number of the first frame in the pair.
- frame_num_b: The frame number of the second frame in the pair.
- offset: The number of frames between frame_num_a and frame_num_b.
- num_detections_a: The number of YOLO detections for the first frame, parsed from the corresponding .txt file.
- num_detections_b: The number of YOLO detections for the second frame, parsed from the corresponding .txt file.
- n_matched: The number of matched detections between the two frames, determined by a simple IoU-based matching algorithm (e.g., using the Hungarian algorithm for optimal assignment).
- n_matched_well: The number of matched detections that have an IoU above a certain threshold (e.g., 0.5), indicating a strong match.
- mean_iou: The average IoU of the matched detections between the two frames.
- match_quality: n_matched_well / n_matched, representing the quality of the matches between the two frames.
- label
This stage will enable us to analyze how object detections change over
"""
import os
from scipy.optimize import linear_sum_assignment
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse
import re
from collections import defaultdict
import random
import yaml
import cv2

with open('configs/stage_2_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

FRAMES_DIR = Path(config['data']['frames_folder'])
DETECTIONS_DIR = Path(config['data']['detections_folder'])
PAIRS_DIR = Path(config['data']['pairs_folder'])
FUTURE_OFFSETS = config['data']['future_frames']
PARQUET_CHECKPOINT_DIR = Path(config['data']['parquet_checkpoint_folder'])


FRAME_RE = re.compile(r"^(?P<video>.+)_img(?P<num>\d+)\.(?P<ext>[^.]+)$")

IMAGE_EXTS = {"txt"}

def compute_iou(box_a, box_b):
    """IoU between two boxes in YOLO normalized center format [x_c, y_c, w, h].

    Coordinates are floats in [0, 1]; we convert to corner form (x1, y1, x2, y2)
    before intersecting. No pixel-style `+ 1` term, since the boxes are
    continuous, not integer pixel grids.
    """
    if len(box_a) < 4 or len(box_b) < 4:
        print(f"Error: Box A {box_a} or Box B {box_b} does not have 4 coordinates.")
        return 0.0

    ax1 = box_a[0] - box_a[2] / 2.0
    ay1 = box_a[1] - box_a[3] / 2.0
    ax2 = box_a[0] + box_a[2] / 2.0
    ay2 = box_a[1] + box_a[3] / 2.0

    bx1 = box_b[0] - box_b[2] / 2.0
    by1 = box_b[1] - box_b[3] / 2.0
    bx2 = box_b[0] + box_b[2] / 2.0
    by2 = box_b[1] + box_b[3] / 2.0

    xA = max(ax1, bx1)
    yA = max(ay1, by1)
    xB = min(ax2, bx2)
    yB = min(ay2, by2)

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union
    

def match_detections(detections_a, detections_b, iou_threshold=0.5):
    if len(detections_a) == 0 or len(detections_b) == 0:
        return 0, 0, 0.0, 0.0  # No matches if either list is empty
    
    iou_matrix = np.zeros((len(detections_a), len(detections_b)))
    
    for i, det_a in enumerate(detections_a):
        for j, det_b in enumerate(detections_b):
            iou_matrix[i, j] = compute_iou(det_a[1:], det_b[1:])
    
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Maximize IoU
    n_matched = len(row_ind)
    n_matched_well = sum(iou_matrix[row_ind[k], col_ind[k]] >= iou_threshold for k in range(n_matched))
    mean_iou = np.mean([iou_matrix[row_ind[k], col_ind[k]] for k in range(n_matched)]) if n_matched > 0 else 0.0
    match_quality = n_matched_well / n_matched if n_matched > 0 else 0.0
    
    return n_matched, n_matched_well, mean_iou, match_quality

def group_detections_by_video(detections_dir: Path):
    """Walk detections_dir and return {video_name: [(frame_num, path), ...]} sorted by frame_num."""
    groups = defaultdict(list)
    skipped = 0

    for d in detections_dir.iterdir():
        if not d.is_file():
            continue
        m = FRAME_RE.match(d.name)
        if not m or m.group("ext").lower() not in IMAGE_EXTS:
            skipped += 1
            continue
        groups[m.group("video")].append((int(m.group("num")), d))

    for video in groups:
        groups[video].sort(key=lambda t: t[0])  # numeric sort, not lex

    if skipped:
        print(f"[warn] skipped {skipped} files that didn't match the naming pattern")
    return groups

def parse_detections(detection_path: Path):
    """Parse a YOLO .txt file and return a list of detections with absolute box coordinates."""
    detections = np.loadtxt(detection_path, dtype=np.float32)
    # Check if the file is empty (no detections)
    if detections.size == 0:
        return None
    return detections if detections.ndim == 2 else [detections]  # Handle single detection case

def build_pairs(detections_by_video, future_offsets, checkpoint_interval=10000):
    columns = ["video_name", "frame_num_a", "frame_num_b", "offset", "num_detections_a", "num_detections_b", "n_matched", "n_matched_well", "mean_iou", "match_quality"]
    detection_df = pd.DataFrame(columns=columns)
    rows = []
    detection_count = 0
    for video, detections in detections_by_video.items():
        for i, (frame_num_a, path_a) in enumerate(detections):
            detections_a = parse_detections(path_a)
            for offset in future_offsets:
                frame_num_b = frame_num_a + offset
                path_b = detections[i + offset][1] if i + offset < len(detections) else None
                if path_b is None:
                    continue
                detections_b = parse_detections(path_b)
                n_matched, n_matched_well, mean_iou, match_quality = match_detections(detections_a, detections_b)
                rows.append({
                    "video_name": video,
                    "frame_num_a": frame_num_a,
                    "frame_num_b": frame_num_b,
                    "offset": offset,
                    "num_detections_a": len(detections_a),
                    "num_detections_b": len(detections_b),
                    "n_matched": n_matched,
                    "n_matched_well": n_matched_well,
                    "mean_iou": mean_iou,
                    "match_quality": match_quality
                })
                detection_count += 1
                if detection_count % checkpoint_interval == 0:
                    print(f"Creating Checkpoint at {detection_count} detections...")
                    checkpoint_path = PARQUET_CHECKPOINT_DIR / f"detection_pairs_checkpoint_{detection_count}.parquet"
                    pd.DataFrame(rows, columns=columns).to_parquet(checkpoint_path, compression='snappy')
        print(f"Processed video {video}, total pairs so far: {len(rows)}")
    detection_df = pd.concat([detection_df, pd.DataFrame(rows, columns=columns)], ignore_index=True)
    return detection_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections-dir", type=Path, default=DETECTIONS_DIR,
                    help="Directory containing {video}_img{N}.txt files")
    ap.add_argument("--out-dir", type=Path, default=PAIRS_DIR,
                    help="Output root;")
    ap.add_argument("--conf", type=float, default=0.3,
                    help="Confidence threshold (default 0.3)")
    args = ap.parse_args()

    if not os.path.exists(args.out_dir):
        args.out_dir.mkdir(parents=True, exist_ok=True)

    detections_by_video = group_detections_by_video(args.detections_dir)
    print(f"Found {len(detections_by_video)} videos, {sum(len(v) for v in detections_by_video.values())} detections total")
    pairs_df = build_pairs(detections_by_video, FUTURE_OFFSETS)
    pairs_path = args.out_dir / "detection_pairs.parquet"
    pairs_df.to_parquet(pairs_path)
    print(f"Saved pairs manifest to {pairs_path}")



if __name__ == "__main__":
    main()
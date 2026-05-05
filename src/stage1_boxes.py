"""
Stage 1: Run YOLO inference on a directory of video frames to produce
ground-truth bounding boxes for the frame-differential dataset.

Input:  a directory of frames named like `{videoname}_frame{NNNN}.{ext}`
Output: per-frame YOLO .txt label files + per-video JSON manifests

Label format (one line per detection):
    class_id  x_center  y_center  width  height  confidence
All coordinates are normalized to [0, 1] (standard YOLO format).
The 6th confidence column is non-standard but ignored by Ultralytics
loaders, and you'll want it for the recompute-decision task downstream.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


# Matches `{videoname}_img{number}.{ext}`, e.g. `MVI_20011_img00001.jpg`.
# The videoname itself can contain underscores (like `MVI_20011`); the
# greedy `.+` plus the literal `_img` anchor handles that correctly.
FRAME_RE = re.compile(r"^(?P<video>.+)_img(?P<num>\d+)\.(?P<ext>[^.]+)$")

IMAGE_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def group_frames_by_video(frames_dir: Path):
    """Walk frames_dir and return {video_name: [(frame_num, path), ...]} sorted by frame_num."""
    groups = defaultdict(list)
    skipped = 0

    for p in frames_dir.iterdir():
        if not p.is_file():
            continue
        m = FRAME_RE.match(p.name)
        if not m or m.group("ext").lower() not in IMAGE_EXTS:
            skipped += 1
            continue
        groups[m.group("video")].append((int(m.group("num")), p))

    for video in groups:
        groups[video].sort(key=lambda t: t[0])  # numeric sort, not lex

    if skipped:
        print(f"[warn] skipped {skipped} files that didn't match the naming pattern")
    return groups


def write_label_file(label_path: Path, result, conf_threshold: float):
    """Write one YOLO .txt file from an Ultralytics Results object."""
    boxes = result.boxes
    lines = []
    if boxes is not None and len(boxes) > 0:
        # xywhn = normalized center-x, center-y, width, height
        xywhn = boxes.xywhn.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        for c, (x, y, w, h), cf in zip(cls, xywhn, conf):
            if c not in VEHICLE_CLASSES:
                continue
            if cf < conf_threshold:
                continue
            lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {cf:.6f}")

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def process_video(model, video_name, frames, labels_dir, manifests_dir,
                  batch_size, conf_threshold, imgsz, device):
    """Run inference on one video's frames and write outputs."""
    paths = [p for _, p in frames]
    nums = [n for n, _ in frames]

    manifest_entries = []
    total_dets = 0

    # Ultralytics handles batching internally when you pass a list,
    # but we chunk to keep memory predictable for long videos.
    for i in range(0, len(paths), batch_size):
        chunk_paths = paths[i:i + batch_size]
        chunk_nums = nums[i:i + batch_size]

        results = model.predict(
            source=[str(p) for p in chunk_paths],
            conf=conf_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
            stream=False,
        )

        for frame_num, src_path, result in zip(chunk_nums, chunk_paths, results):
            label_path = labels_dir / f"{src_path.stem}.txt"
            n = write_label_file(label_path, result, conf_threshold)
            total_dets += n
            manifest_entries.append({
                "frame_num": frame_num,
                "image_path": str(src_path),
                "label_path": str(label_path),
                "num_detections": n,
                "image_height": int(result.orig_shape[0]),
                "image_width": int(result.orig_shape[1]),
            })

    manifest = {
        "video_name": video_name,
        "num_frames": len(frames),
        "total_detections": total_dets,
        "frames": manifest_entries,  # already in temporal order
    }
    manifest_path = manifests_dir / f"{video_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return total_dets, manifest_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True,
                    help="Directory containing {video}_frame{N}.jpg files")
    ap.add_argument("--weights", type=Path, required=True,
                    help="Path to your trained YOLO .pt weights")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output root; labels/ and manifests/ are created inside")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold (default 0.25)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None,
                    help="e.g. 'cuda:0' or 'cpu'; default lets Ultralytics decide")
    args = ap.parse_args()

    labels_dir = args.out_dir / "labels"
    manifests_dir = args.out_dir / "manifests"
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.weights}")
    model = YOLO(str(args.weights))

    groups = group_frames_by_video(args.frames_dir)
    print(f"Found {len(groups)} videos, {sum(len(v) for v in groups.values())} frames total")

    summary = []
    for video_name, frames in sorted(groups.items()):
        print(f"\n[{video_name}] {len(frames)} frames")
        n_dets, manifest_path = process_video(
            model, video_name, frames,
            labels_dir, manifests_dir,
            args.batch_size, args.conf, args.imgsz, args.device,
        )
        print(f"  -> {n_dets} detections, manifest: {manifest_path.name}")
        summary.append({"video": video_name, "frames": len(frames), "detections": n_dets})

    # Top-level index across all videos for convenience
    (args.out_dir / "index.json").write_text(json.dumps({
        "videos": summary,
        "labels_dir": str(labels_dir),
        "manifests_dir": str(manifests_dir),
    }, indent=2))
    print(f"\nDone. Wrote {len(summary)} manifests + index.json to {args.out_dir}")


if __name__ == "__main__":
    main()
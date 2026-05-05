"""
Stage 3: Labeling frame pairs based on computed metrics
Input: A parquet manifest containing pairs of frames and their computed metrics, where the second frame in each pair is offset from the first by a variable number of frames (e.g., 1, 3, 5, 7, 11, 13).
Output: A parquet manifest containing pairs of frames and their labels, where the second frame in each pair is offset from the first by a variable number of frames (e.g., 1, 3, 5, 7, 11, 13).
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
- label: 0 for "recompute bounding boxes" or 1 for "keep bounding boxes", determined by a simple heuristic based on the computed metrics.
This stage will enable us to provide labels for training a model to predict when bounding boxes need to be recomputed based on the changes in object detections between pairs of frames.
"""


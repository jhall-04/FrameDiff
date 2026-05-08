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

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

with open('configs/stage_3_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

THRESHOLDS = config['data']['thresholds']
PARQUET_FILE = config['data']['parquet_file']
LABELS_OUTPUT_DIR = Path(config['data']['labels_output_dir'])
LABELS_OUTPUT_DIR.mkdir(exist_ok=True)

def split_dataset(df):
    """Assign each video to train / val / test. Split is by video, never by row."""
    videos = df[["video_name"]].drop_duplicates()
    videos["video_group"] = videos["video_name"].str.extract(r"MVI_(\d{2})")[0]

    group_counts = videos["video_group"].value_counts()
    singles_mask = videos["video_group"].isin(group_counts[group_counts < 2].index)

    singles = videos[singles_mask]
    multiples = videos[~singles_mask]

    # If every group has fewer than 2 videos, stratify is impossible (empty `multiples`).
    if len(multiples) == 0:
        print(
            "[warn] No video_group has >=2 videos; using non-stratified random splits."
        )
        if len(videos) == 0:
            raise ValueError("No videos in dataframe")
        train_v, hold_v = train_test_split(
            videos, test_size=0.40, random_state=42, shuffle=True
        )
        if len(hold_v) >= 2:
            val_v, test_v = train_test_split(
                hold_v, test_size=0.50, random_state=42, shuffle=True
            )
        elif len(hold_v) == 1:
            val_v, test_v = hold_v.copy(), videos.iloc[0:0].copy()
        else:
            val_v, test_v = videos.iloc[0:0].copy(), videos.iloc[0:0].copy()
    else:
        # 60% train / 20% val / 20% test, stratified by video_group
        train_v, hold_v = train_test_split(
            multiples,
            test_size=0.40,
            stratify=multiples["video_group"],
            random_state=42,
        )
        # Ensure each set has at least two videos for stratification; if not, assign to test
        hold_group_counts = hold_v["video_group"].value_counts()
        hold_rare_mask = hold_v["video_group"].isin(
            hold_group_counts[hold_group_counts < 2].index
        )
        hold_rare = hold_v[hold_rare_mask]
        hold_common = hold_v[~hold_rare_mask]

        if len(hold_common) == 0:
            print(
                "[warn] Nothing left in hold_common for val/test; "
                "assigning empty val/test sets."
            )
            val_v, test_v = videos.iloc[0:0].copy(), videos.iloc[0:0].copy()
        elif len(hold_common) == 1:
            val_v, test_v = hold_common.copy(), videos.iloc[0:0].copy()
        else:
            val_v, test_v = train_test_split(
                hold_common,
                test_size=0.50,
                stratify=hold_common["video_group"],
                random_state=42,
            )

        train_v = pd.concat([train_v, singles, hold_rare])

    video_to_split = {
        **{v: "train" for v in train_v["video_name"]},
        **{v: "val" for v in val_v["video_name"]},
        **{v: "test" for v in test_v["video_name"]},
    }
    # Map each row to a split in the original dataframe to its split based on video_name
    df["split"] = df["video_name"].map(video_to_split)
    return df


def drop_empty_detection_pairs(df):
    """Drop pairs where either frame has no YOLO boxes (nothing to match / trivial)."""
    n0 = len(df)
    df = df[
        (df["num_detections_a"] > 0) & (df["num_detections_b"] > 0)
    ].reset_index(drop=True)
    dropped = n0 - len(df)
    if dropped:
        print(f"[info] Dropped {dropped} pairs with zero detections on one or both frames ({len(df)} left).")
    return df

def label_data(df, threshold):
    """Label each pair as 0 (recompute) or 1 (keep) based on heuristics."""
    def label_row(row):
        if row['mean_iou'] < threshold:
            return 0  # recompute
        else:
            return 1  # keep
    df['label'] = df.apply(label_row, axis=1)
    return df

def main():
    # Load pairs
    df = pd.read_parquet(PARQUET_FILE)
    # Drop pairs where either frame has no detections, as these are trivial cases that don't provide useful training signal for the model. This also helps to balance the dataset and focus on more challenging examples.
    df = drop_empty_detection_pairs(df)
    df = split_dataset(df)
    # For each threshold label the data and check class balance.
    for t in THRESHOLDS:
        labeled_df = label_data(df.copy(), t)
        # Check class balance
        balance = labeled_df['label'].value_counts(normalize=True)
        print(f"Threshold {t}: Class balance:\n{balance}\n")
        output_file = LABELS_OUTPUT_DIR / f"detection_pairs_labeled_{int(t*100)}.parquet"
        labeled_df.to_parquet(output_file)
        print(f"Saved labeled dataset with threshold {t} to {output_file.name}")


if __name__ == "__main__":
    main()

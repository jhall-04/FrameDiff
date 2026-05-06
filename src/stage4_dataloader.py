"""
Stage 4: PyTorch Dataset + DataLoaders for frame-pair training.

Reads rows from the labeled parquet (Stage 3), loads two frames, builds a 2-channel tensor:
  ch0 = optical flow magnitude (normalized)
  ch1 = Gaussian-blurred absolute difference (normalized)

No mkdir / folder creation here — we only read frames and parquet paths you already have.

For downsample=False you get full-res tensors (slow, big memory). For downsample=True (default)
images are resized before flow + diff so the CNN sees e.g. 256x256.
"""

from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader, Dataset

import cv2
import numpy as np
import pandas as pd
import numpy as np
import cv2
import torch
import yaml


def frame_path(frames_folder: Path, video_name: str, frame_num: int) -> Path:
    # must match Stage 1 frame filenames: {video}_img{NNNNN}.jpg
    return frames_folder / f"{video_name}_img{frame_num:05d}.jpg"


def _calc_flow_magnitude(img_a_gray: np.ndarray, img_b_gray: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        img_a_gray,
        img_b_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    magnitude, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude.astype(np.float32)


def _blur_difference(img_a_gray: np.ndarray, img_b_gray: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    blurred_a = cv2.GaussianBlur(img_a_gray, (ksize, ksize), sigma)
    blurred_b = cv2.GaussianBlur(img_b_gray, (ksize, ksize), sigma)
    return cv2.absdiff(blurred_a, blurred_b).astype(np.float32)


class PairDataset(Dataset):
    """One row = one (frame_a, frame_b) pair + offset k + binary label."""

    def __init__(self,
                 parquet_path: Path | str,
                 frames_folder: Path | str,
                 split: str,
                 *,
                 downsample: bool = True,
                 target_size: int | tuple[int, int] = 128,
                 flow_norm_scale: float = 30.0,
                 gaussian_ksize: int = 5,
                 gaussian_sigma: float = 1.0) -> None:

        self.frames_folder = Path(frames_folder)
        df = pd.read_parquet(parquet_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows for split={split!r} in {parquet_path}")

        self.df = df
        self.downsample = downsample
        if isinstance(target_size, int):
            self.target_hw = (target_size, target_size)
        else:
            self.target_hw = target_size  # (h, w)
        self.flow_norm_scale = float(flow_norm_scale)
        self.gaussian_ksize = int(gaussian_ksize)
        self.gaussian_sigma = float(gaussian_sigma)


    def __len__(self) -> int:
        return len(self.df)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        video = str(row["video_name"])
        fa = int(row["frame_num_a"])
        fb = int(row["frame_num_b"])
        label = float(row["label"])

        pa = frame_path(self.frames_folder, video, fa)
        pb = frame_path(self.frames_folder, video, fb)
        if not pa.is_file():
            raise FileNotFoundError(f"Missing frame: {pa}")
        if not pb.is_file():
            raise FileNotFoundError(f"Missing frame: {pb}")

        # grayscale — flow + diff do not need color
        img_a = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
        if img_a is None or img_b is None:
            raise RuntimeError(f"cv2 failed to read images: {pa} / {pb}")

        # optional resize (cheap features — 256 is enough for the CNN)
        if self.downsample:
            w, h = self.target_hw[1], self.target_hw[0]
            img_a = cv2.resize(img_a, (w, h), interpolation=cv2.INTER_AREA)
            img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_AREA)

        mag = _calc_flow_magnitude(img_a, img_b)
        diff = _blur_difference(img_a, img_b, ksize=self.gaussian_ksize, sigma=self.gaussian_sigma)

        # put both channels on a similar [0, 1] scale so neither dominates the convs
        mag_norm = np.clip(mag / self.flow_norm_scale, 0.0, 1.0)
        diff_norm = np.clip(diff / 255.0, 0.0, 1.0)

        x = np.stack([mag_norm, diff_norm], axis=0)
        x_t = torch.from_numpy(np.ascontiguousarray(x)).float()
        y_t = torch.tensor(label, dtype=torch.float32)
        return x_t, y_t


def make_dataloaders(config_path: str | Path = "configs/stage4_config.yaml",
                     *,
                     batch_size: int | None = None,
                     num_workers: int | None = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from a YAML config.
    Expected config keys under `data`: labeled_parquet, frames_folder,
    plus optional: downsample, target_hw, flow_norm_scale, batch_size, num_workers (defaults below if missing).
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config not found: {path}. Create it or pass paths via PairDataset directly."
        )
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = cfg["data"]

    common = dict(
        downsample=data.get("downsample", True),
        target_size=data.get("target_hw", 128),
        flow_norm_scale=data.get("flow_norm_scale", 30.0),
        gaussian_ksize=data.get("gaussian_ksize", 5),
        gaussian_sigma=data.get("gaussian_sigma", 1.0),
    )

    pq = data["labeled_parquet"]
    frames = data["frames_folder"]

    train_ds = PairDataset(pq, frames, "train", **common)
    val_ds = PairDataset(pq, frames, "val", **common)
    test_ds = PairDataset(pq, frames, "test", **common)

    bs = batch_size if batch_size is not None else int(data.get("batch_size", 32))
    nw = num_workers if num_workers is not None else int(data.get("num_workers", 0))

    # pin_memory helps GPU transfer; only matters if you train on CUDA
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )
    return train_loader, val_loader, test_loader


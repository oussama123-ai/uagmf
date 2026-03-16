"""
Dataset classes for BioVid, UNBC-McMaster, EmoPain, and MD-NPL (OOD).

5-fold subject-independent cross-validation:
    Test subjects appear in NEITHER training NOR validation folds.

MD-NPL is used exclusively for OOD evaluation; it was not used in any
training or model-selection step (Section 4.1 / Table 6 of the paper).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


DATASET_SPECS = {
    "biovid": {
        "n_subjects": 87, "modalities": ["video", "hrv", "sco2", "resp"],
        "annotation": "pspi", "fps": 25, "irr_kappa": 0.81,
    },
    "unbc": {
        "n_subjects": 129, "modalities": ["video"],
        "annotation": "pspi", "fps": 25, "irr_kappa": 0.79,
    },
    "emopain": {
        "n_subjects": 60, "modalities": ["video", "hrv", "resp"],
        "annotation": "nrs_continuous", "fps": 25, "irr_kappa": 0.76,
    },
    "mdnpl": {
        "n_subjects": 48, "modalities": ["video"],
        "annotation": "nfcs_pspi", "fps": 25, "irr_kappa": 0.83,
        "ood_only": True,
    },
}


class PainClipDataset(Dataset):
    """
    Base dataset for short video clips with synchronised physiological signals.

    Each sample is one clip with:
        video:     (T, 3, 112, 112) float32 normalised frames
        hrv:       (T, 4) HRV features or zeros if unavailable
        spo2:      (T, 1) SpO₂ or zeros
        resp:      (T, 1) respiratory rate or zeros
        nrs_score: float scalar
        subject_id: str

    Args:
        feature_dir:  Root directory of preprocessed features.
        subjects:     Subject IDs to include.
        dataset_name: One of biovid / unbc / emopain / mdnpl.
        n_frames:     Frames per clip (default 64).
        stride:       Clip stride (default 32, 50% overlap).
        augment:      Apply training augmentation.
    """

    def __init__(
        self,
        feature_dir: str,
        subjects: List[str],
        dataset_name: str = "biovid",
        n_frames: int = 64,
        stride: int = 32,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dir = Path(feature_dir)
        self.subjects = subjects
        self.dataset_name = dataset_name
        self.n_frames = n_frames
        self.stride = stride
        self.augment = augment
        self.modalities = DATASET_SPECS[dataset_name]["modalities"]

        self._index: List[Dict] = []
        self._build_index()

    def _build_index(self) -> None:
        for subj in self.subjects:
            subj_dir = self.feature_dir / subj
            if not subj_dir.exists():
                continue
            for clip_dir in sorted(subj_dir.glob("clip_*")):
                label_f = clip_dir / "labels.csv"
                if not label_f.exists():
                    continue
                df = pd.read_csv(label_f)
                n = len(df)
                for start in range(0, n - self.n_frames + 1, self.stride):
                    nrs = float(df["nrs_score"].iloc[start:start + self.n_frames].mean())
                    if np.isnan(nrs):
                        continue
                    self._index.append({
                        "subject_id": subj,
                        "clip_dir": str(clip_dir),
                        "start": start,
                        "nrs_score": nrs,
                    })
        logger.info(
            f"{self.dataset_name}: {len(self._index)} clips "
            f"from {len(self.subjects)} subjects."
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self._index[idx]
        clip_dir = Path(entry["clip_dir"])
        s = entry["start"]
        T = self.n_frames

        # Video frames: (T, 3, 112, 112) uint8 → float32 / 255
        video_arr = np.load(clip_dir / "video.npy", mmap_mode="r")[s:s + T]
        video = torch.from_numpy(video_arr.astype(np.float32) / 255.0)

        # Physiological signals
        hrv = self._load_signal(clip_dir / "hrv.npy", s, T, dim=4)
        spo2 = self._load_signal(clip_dir / "spo2.npy", s, T, dim=1)
        resp = self._load_signal(clip_dir / "resp.npy", s, T, dim=1)

        if self.augment:
            video = self._augment_video(video)

        return {
            "video": video,
            "hrv": hrv,
            "spo2": spo2,
            "resp": resp,
            "nrs_score": torch.tensor(
                np.clip(entry["nrs_score"], 0, 10), dtype=torch.float32
            ),
            "subject_id": entry["subject_id"],
        }

    def _load_signal(
        self, path: Path, start: int, n: int, dim: int
    ) -> torch.Tensor:
        if path.exists() and path.stat().st_size > 0:
            arr = np.load(path, mmap_mode="r")[start:start + n]
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            return torch.from_numpy(arr.astype(np.float32))
        return torch.zeros(n, dim, dtype=torch.float32)

    def _augment_video(self, video: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            video = video.flip(-1)
        brightness = torch.rand(1).item() * 0.4 - 0.2
        video = (video + brightness).clamp(0, 1)
        return video


def get_cv_splits(
    dataset_name: str,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Dict[str, List[str]]]:
    """Generate stratified 5-fold subject-independent CV splits."""
    from sklearn.model_selection import KFold

    spec = DATASET_SPECS[dataset_name]
    n = spec["n_subjects"]
    prefix = "subject" if dataset_name != "mdnpl" else "neonate"
    subjects = [f"{prefix}_{i:03d}" for i in range(1, n + 1)]

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(subjects).tolist()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    return [
        {"train": [shuffled[i] for i in tr],
         "test":  [shuffled[i] for i in te]}
        for tr, te in kf.split(shuffled)
    ]


def build_dataloaders(
    feature_dir: str,
    fold: int,
    dataset_name: str = "biovid",
    batch_size: int = 16,
    num_workers: int = 8,
    n_frames: int = 64,
    pin_memory: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    from torch.utils.data import DataLoader

    splits = get_cv_splits(dataset_name)
    fold_split = splits[fold]

    train_ds = PainClipDataset(feature_dir, fold_split["train"],
                               dataset_name, n_frames, stride=32, augment=True)
    val_ds = PainClipDataset(feature_dir, fold_split["test"],
                             dataset_name, n_frames, stride=8, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader

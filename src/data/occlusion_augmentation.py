"""
Synthetic Occlusion Augmentation.

Applies geometric occlusion masks to video frames for training
and controlled robustness evaluation (Table 8 / Fig 4).

IMPORTANT: All occlusion robustness results in the paper were evaluated
under SYNTHETICALLY APPLIED geometric masks. They do not constitute
validation under naturally occurring ICU occlusions.
See Section 5.2 (Scope) of the paper.

Four occlusion types:
    - none:    No occlusion (pass-through)
    - mask:    Medical face mask (lower face, r ≈ 0.35–0.50)
    - tube:    Breathing tube (perioral + nasal, r ≈ 0.45–0.65)
    - partial: Partial facial occlusion (random region, r ≈ 0.15–0.30)
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

OCCLUSION_TYPES = ["none", "mask", "tube", "partial"]


def apply_medical_mask(
    frame: np.ndarray, occlusion_ratio: float = 0.40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a medical face mask (lower face region).
    Returns (occluded_frame, binary_mask).
    """
    H, W = frame.shape[:2]
    mask = np.zeros((H, W), dtype=np.float32)
    # Lower 40-50% of face (nose to chin)
    top = int(H * 0.45)
    mask[top:, :] = 1.0
    occluded = frame.copy().astype(np.float32)
    occluded[mask == 1] = 0
    return occluded.astype(frame.dtype), mask


def apply_breathing_tube(
    frame: np.ndarray, occlusion_ratio: float = 0.55
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate breathing tube + tape (perioral + nasal region, heavier).
    """
    H, W = frame.shape[:2]
    mask = np.zeros((H, W), dtype=np.float32)
    # Tube strip (horizontal band across nose/mouth)
    top = int(H * 0.40)
    bot = int(H * 0.85)
    mask[top:bot, :] = 1.0
    # Additional vertical tape strip
    cx = W // 2
    tape_w = W // 8
    mask[:, cx - tape_w:cx + tape_w] = 1.0

    occluded = frame.copy().astype(np.float32)
    occluded[mask == 1] = 0
    return occluded.astype(frame.dtype), mask


def apply_partial_occlusion(
    frame: np.ndarray,
    target_ratio: float = 0.20,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate random partial occlusion (bandage, hand, equipment).
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    H, W = frame.shape[:2]
    mask = np.zeros((H, W), dtype=np.float32)

    # Random rectangle with target area ≈ target_ratio * H * W
    area = int(target_ratio * H * W)
    rect_h = rng.randint(int(H * 0.15), int(H * 0.45))
    rect_w = area // max(rect_h, 1)
    rect_w = min(rect_w, W)

    y0 = rng.randint(0, max(H - rect_h, 1))
    x0 = rng.randint(0, max(W - rect_w, 1))
    mask[y0:y0 + rect_h, x0:x0 + rect_w] = 1.0

    occluded = frame.copy().astype(np.float32)
    occluded[mask == 1] = 0
    return occluded.astype(frame.dtype), mask


def apply_occlusion(
    frame: np.ndarray,
    occlusion_type: str = "mask",
    occlusion_ratio: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply a specified occlusion type to a frame.

    Args:
        frame:           (H, W, 3) uint8 or float32 frame.
        occlusion_type:  "none", "mask", "tube", or "partial".
        occlusion_ratio: Target coverage ratio (uses type default if None).
        seed:            Random seed for partial occlusion.

    Returns:
        occluded_frame: (H, W, 3)
        binary_mask:    (H, W) float32 in {0, 1}
        actual_ratio:   float — actual fraction of masked pixels
    """
    if occlusion_type == "none":
        H, W = frame.shape[:2]
        return frame, np.zeros((H, W), dtype=np.float32), 0.0

    if occlusion_type == "mask":
        r = occlusion_ratio or 0.40
        occ, mask = apply_medical_mask(frame, r)
    elif occlusion_type == "tube":
        r = occlusion_ratio or 0.55
        occ, mask = apply_breathing_tube(frame, r)
    elif occlusion_type == "partial":
        r = occlusion_ratio or 0.20
        occ, mask = apply_partial_occlusion(frame, r, seed)
    else:
        raise ValueError(f"Unknown occlusion type: {occlusion_type}")

    actual_ratio = float(mask.mean())
    return occ, mask, actual_ratio


class OcclusionAugmentor:
    """
    Dataset-level occlusion augmentor for training and evaluation.

    Training mode: randomly applies occlusion types with given probabilities.
    Evaluation mode: applies a fixed occlusion type at a specified rate.
    """

    def __init__(
        self,
        train_probs: dict = None,
        eval_occlusion_type: str = "none",
        eval_occlusion_ratio: float = 0.0,
    ) -> None:
        self.train_probs = train_probs or {
            "none": 0.40,
            "mask": 0.25,
            "tube": 0.20,
            "partial": 0.15,
        }
        self.eval_occlusion_type = eval_occlusion_type
        self.eval_occlusion_ratio = eval_occlusion_ratio

    def augment_clip(
        self,
        frames: np.ndarray,
        training: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, str, float]:
        """
        Apply consistent occlusion across all frames of a clip.

        Args:
            frames:   (T, H, W, 3) uint8.
            training: If True, sample occlusion type from train_probs.

        Returns:
            occ_frames: (T, H, W, 3)
            masks:      (T, H, W)
            occ_type:   str
            avg_ratio:  float
        """
        if training:
            occ_type = random.choices(
                list(self.train_probs.keys()),
                weights=list(self.train_probs.values()),
            )[0]
            ratio = None
        else:
            occ_type = self.eval_occlusion_type
            ratio = self.eval_occlusion_ratio if self.eval_occlusion_ratio > 0 else None

        occ_frames = []
        masks = []
        actual_ratios = []

        for frame in frames:
            occ, mask, actual_ratio = apply_occlusion(frame, occ_type, ratio)
            occ_frames.append(occ)
            masks.append(mask)
            actual_ratios.append(actual_ratio)

        return (
            np.stack(occ_frames),
            np.stack(masks),
            occ_type,
            float(np.mean(actual_ratios)),
        )

    def augment_tensor(
        self,
        video: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, str, float]:
        """Tensor-friendly wrapper (input/output: B,T,C,H,W or T,C,H,W)."""
        squeeze = video.dim() == 4
        if squeeze:
            video = video.unsqueeze(0)

        B, T, C, H, W = video.shape
        results_v, results_m = [], []

        for b in range(B):
            frames = (
                video[b].permute(0, 2, 3, 1).cpu().numpy() * 255
            ).astype(np.uint8)
            occ_frames, masks, occ_type, ratio = self.augment_clip(frames, training)
            results_v.append(
                torch.from_numpy(occ_frames.astype(np.float32) / 255.0)
                .permute(0, 3, 1, 2)
            )
            results_m.append(torch.from_numpy(masks))

        occ_video = torch.stack(results_v)
        occ_masks = torch.stack(results_m)

        if squeeze:
            occ_video = occ_video.squeeze(0)
            occ_masks = occ_masks.squeeze(0)

        return occ_video, occ_masks, occ_type, ratio

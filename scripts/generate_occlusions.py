#!/usr/bin/env python3
"""
Synthetic occlusion data generator.

Applies geometric occlusion masks to held-out test frames at specified
coverage ratios (0%, 20%, 40%, 60%, 80%) for Table 8 evaluation.

IMPORTANT: These are SYNTHETICALLY APPLIED masks for controlled
robustness evaluation only. See Section 5.2 of the paper.

Usage:
    python scripts/generate_occlusions.py \\
        --data_root data/features \\
        --output_dir data/occluded \\
        --occlusion_type mask \\
        --ratios 0.0 0.2 0.4 0.6 0.8
"""

import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src.data.occlusion_augmentation import apply_occlusion, OCCLUSION_TYPES
from src.utils.logging_utils import setup_logging

logger = logging.getLogger("uagmf.gen_occlusions")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--occlusion_type", default="mask", choices=OCCLUSION_TYPES)
    p.add_argument("--ratios", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8])
    p.add_argument("--split", default="test", choices=["train", "test"])
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.output_dir, run_name="gen_occlusions")
    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Generating {args.occlusion_type} occlusions at ratios {args.ratios}. "
        "All are SYNTHETIC geometric masks (Section 5.2)."
    )

    video_files = list(data_root.rglob("video.npy"))
    logger.info(f"Found {len(video_files)} video files.")

    for ratio in args.ratios:
        ratio_dir = out_root / f"{args.occlusion_type}_r{int(ratio*100):03d}"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        total_frames = 0

        for vf in tqdm(video_files, desc=f"r={ratio:.0%}"):
            frames = np.load(vf)   # (T, C, H, W) or (T, H, W, C)
            # Convert to (T, H, W, C) uint8 for occlusion application
            if frames.shape[1] == 3:   # (T, 3, H, W)
                frames = frames.transpose(0, 2, 3, 1)
            if frames.dtype != np.uint8:
                frames = (frames * 255).clip(0, 255).astype(np.uint8)

            occ_frames, masks, actual_ratios = [], [], []
            for frame in frames:
                occ, mask, actual_r = apply_occlusion(
                    frame, args.occlusion_type, ratio if ratio > 0 else None
                )
                occ_frames.append(occ)
                masks.append(mask)
                actual_ratios.append(actual_r)

            occ_arr = np.stack(occ_frames).transpose(0, 3, 1, 2)   # (T,3,H,W)
            mask_arr = np.stack(masks)[:, np.newaxis]               # (T,1,H,W)

            # Mirror directory structure
            rel = vf.relative_to(data_root)
            out_clip = ratio_dir / rel.parent
            out_clip.mkdir(parents=True, exist_ok=True)
            np.save(out_clip / "video.npy", occ_arr.astype(np.float32) / 255.0)
            np.save(out_clip / "mask.npy", mask_arr)
            total_frames += len(frames)

        logger.info(f"r={ratio:.0%}: {total_frames} frames saved to {ratio_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
LOSO cross-validation for UAG-MF.

Usage:
    python scripts/train_loso.py --config configs/loso.yaml --dataset biovid
    python scripts/train_loso.py --config configs/loso.yaml --dataset unbc
    python scripts/train_loso.py --config configs/loso.yaml --dataset emopain
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets import get_dataset  # type: ignore
from src.models.uagmf import UAGMF           # type: ignore
from src.training.loso import loso_cross_validate  # type: ignore
from src.utils.logging_utils import get_logger  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="LOSO cross-validation for UAG-MF")
    p.add_argument("--config", required=True, help="Path to loso.yaml")
    p.add_argument(
        "--dataset",
        required=True,
        choices=["biovid", "unbc", "emopain"],
        help="Dataset to run LOSO on",
    )
    p.add_argument("--data_root", default=None, help="Override data root")
    p.add_argument("--output_dir", default="results/loso", help="Output directory")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load config ──────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Inject dataset-specific settings
    dataset_cfg = next(
        (d for d in config["datasets"] if d["name"] == args.dataset), None
    )
    if dataset_cfg is None:
        raise ValueError(f"Dataset '{args.dataset}' not found in {args.config}")

    config["dataset"] = dataset_cfg
    if args.data_root is not None:
        config["dataset"]["root"] = args.data_root

    # ── Logger ───────────────────────────────────────────────
    logger = get_logger("train_loso")
    logger.info(f"LOSO on {args.dataset} — {config['dataset']['num_subjects']} subjects")

    # ── Dataset ──────────────────────────────────────────────
    dataset = get_dataset(
        name=args.dataset,
        root=config["dataset"]["root"],
        split="all",
    )
    logger.info(f"Loaded {len(dataset)} samples from {args.dataset}")

    # ── Run LOSO ─────────────────────────────────────────────
    summary = loso_cross_validate(
        config=config,
        model_fn=lambda: UAGMF(config),
        dataset=dataset,
        device=args.device,
        output_dir=args.output_dir,
        logger=logger,
    )

    # ── Print summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"LOSO SUMMARY — {args.dataset.upper()}")
    print("=" * 60)
    print(f"Subjects:  {summary['n_subjects']}")
    print(f"MSE:       {summary['mse_mean']:.4f} ± {summary['mse_std']:.4f}")
    print(f"PCC:       {summary['pcc_mean']:.4f} ± {summary['pcc_std']:.4f}")
    print(f"MAE:       {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
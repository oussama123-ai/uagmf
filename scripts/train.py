#!/usr/bin/env python3
"""
Main training script for UAG-MF.

Usage:
    # Single fold
    python scripts/train.py --config configs/biovid.yaml \\
        --data_root data/features --fold 0 --output_dir experiments/biovid_f0

    # All 5 folds
    for fold in 0 1 2 3 4; do
        python scripts/train.py --config configs/default.yaml \\
            --fold $fold --data_root data/features --output_dir experiments/fold${fold}
    done

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 scripts/train.py \\
        --config configs/default.yaml --data_root data/features
"""

import argparse, logging, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, yaml
from omegaconf import OmegaConf

from src.data.datasets import build_dataloaders
from src.models.uagmf import UAGMF
from src.training.trainer import Trainer
from src.utils.logging_utils import setup_logging


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", default="experiments/uagmf_run")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--dataset", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    if args.dataset:
        cfg.data.dataset = args.dataset

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + args.fold)

    if local_rank == 0:
        setup_logging(os.path.join(args.output_dir, "logs"), run_name=f"fold{args.fold}")

    train_loader, val_loader = build_dataloaders(
        feature_dir=args.data_root, fold=args.fold,
        dataset_name=cfg.data.dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        n_frames=cfg.data.clip_frames,
    )

    model = UAGMF(
        d_model=cfg.model.fusion_d_model,
        n_mc_samples=cfg.model.mc_dropout_samples,
        n_ensemble=cfg.model.ensemble_size,
        alert_threshold=cfg.model.alert_threshold,
        rules_path=cfg.model.rules_path,
    ).to(device)

    if local_rank == 0:
        logging.getLogger("uagmf").info(
            f"Parameters: {model.count_parameters()/1e6:.1f}M total, "
            f"{model.count_parameters(inference_only=True)/1e6:.1f}M inference"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    total_steps = cfg.training.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    fold_dir = os.path.join(args.output_dir, f"fold{args.fold}")
    trainer = Trainer(
        model=model, output_dir=fold_dir, device=str(device),
        local_rank=local_rank, use_amp=cfg.training.amp,
        grad_clip=cfg.training.grad_clip_max_norm,
        patience=cfg.training.early_stopping_patience,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(train_loader, val_loader, optimizer, scheduler,
                  n_epochs=cfg.training.epochs)

    if local_rank == 0:
        logging.getLogger("uagmf").info(
            f"Training complete. Best val MSE = {trainer.best_val_mse:.4f}"
        )


if __name__ == "__main__":
    main()

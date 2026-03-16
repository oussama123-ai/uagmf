#!/usr/bin/env python3
"""
Evaluation script for UAG-MF — computes all metrics from Tables 7–10.

Usage:
    python scripts/evaluate.py \\
        --checkpoint experiments/fold0/best_model.pth \\
        --data_root data/features --dataset biovid --fold 0 \\
        --output_dir results/fold0

    # All folds + OOD
    python scripts/evaluate.py --all_folds \\
        --checkpoint experiments/fold0/best_model.pth \\
        --ood_eval --output_dir results/
"""

import argparse, json, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import build_dataloaders, get_cv_splits, PainClipDataset
from src.evaluation.metrics import compute_all_metrics, format_results_table
from src.models.uagmf import UAGMF
from src.utils.logging_utils import setup_logging

logger = logging.getLogger("uagmf.evaluate")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset", default="biovid")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--all_folds", action="store_true")
    p.add_argument("--ood_eval", action="store_true", help="Also evaluate on MD-NPL")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    preds, targets, sigmas, alerts = [], [], [], []
    for batch in loader:
        video = batch["video"].to(device)
        target = batch["nrs_score"]
        out = model(video,
                    hrv=batch.get("hrv", torch.zeros(video.size(0), video.size(1), 4)).to(device))
        preds.append(out["mu"].cpu().numpy())
        targets.append(target.numpy())
        sigmas.append(out["sigma"].cpu().numpy())
        alerts.append(out["alert"].cpu().numpy())
    return (np.concatenate(preds), np.concatenate(targets),
            np.concatenate(sigmas), np.concatenate(alerts))


def evaluate_fold(model, args, fold, device):
    _, val_loader = build_dataloaders(
        args.data_root, fold, args.dataset, args.batch_size, num_workers=4
    )
    pred, target, sigma, alert = run_inference(model, val_loader, device)
    return compute_all_metrics(pred, target, pred_std=sigma, n_bootstrap=1000)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(out_dir / "logs"), run_name="evaluate")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.checkpoint, map_location=device)
    model = UAGMF()
    state = {k: v for k, v in ck["model_state_dict"].items()
             if not k.startswith("generative_recon.discriminator")}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    if args.all_folds:
        fold_metrics = []
        for fold in range(5):
            m = evaluate_fold(model, args, fold, device)
            fold_metrics.append(m)
            logger.info(f"Fold {fold}: MSE={m['mse']:.4f} PCC={m['pcc']:.4f}")

        summary = {
            "mse_mean": float(np.mean([m["mse"] for m in fold_metrics])),
            "mse_std":  float(np.std( [m["mse"] for m in fold_metrics])),
            "pcc_mean": float(np.mean([m["pcc"] for m in fold_metrics])),
            "icc_mean": float(np.mean([m["icc"] for m in fold_metrics])),
            "qwk_mean": float(np.mean([m["qwk"] for m in fold_metrics])),
            "ece_mean": float(np.mean([m.get("ece", 0) for m in fold_metrics])),
            "per_fold": fold_metrics,
        }
        logger.info(
            f"5-fold summary: MSE={summary['mse_mean']:.4f}±{summary['mse_std']:.4f} "
            f"PCC={summary['pcc_mean']:.4f}"
        )
        with open(out_dir / "cv_summary.json", "w") as f:
            json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v
                       for k, v in summary.items()}, f, indent=2)
    else:
        m = evaluate_fold(model, args, args.fold, device)
        logger.info(f"Fold {args.fold}: MSE={m['mse']:.4f} PCC={m['pcc']:.4f} "
                    f"ICC={m['icc']:.4f} QWK={m['qwk']:.4f} ECE={m.get('ece',0):.4f}")
        with open(out_dir / f"fold{args.fold}_metrics.json", "w") as f:
            json.dump({k: float(v) for k, v in m.items()}, f, indent=2)

    if args.ood_eval:
        logger.info("Running OOD evaluation on MD-NPL (no fine-tuning)...")
        splits = get_cv_splits("mdnpl", n_folds=1)
        ood_ds = PainClipDataset(args.data_root, splits[0]["test"], "mdnpl", augment=False)
        ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, num_workers=4)
        pred, target, sigma, _ = run_inference(model, ood_loader, device)
        ood_m = compute_all_metrics(pred, target, pred_std=sigma, n_bootstrap=500)
        logger.info(f"OOD MD-NPL: {ood_m}")
        with open(out_dir / "ood_mdnpl_metrics.json", "w") as f:
            json.dump({k: float(v) for k, v in ood_m.items()}, f, indent=2)


if __name__ == "__main__":
    main()

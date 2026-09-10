"""
Leave-One-Subject-Out (LOSO) cross-validation.

This module implements the gold-standard LOSO protocol requested by Reviewer 2.
Each subject serves as the sole test subject exactly once, with all other
subjects used for training.

Compatible with the existing UAG-MF codebase at github.com/oussama123-ai/uagmf.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


def loso_cross_validate(
    config: dict,
    model_fn: Callable,
    dataset,
    device: str = "cuda",
    output_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Run Leave-One-Subject-Out cross-validation.

    Args:
        config: Configuration dict (must contain 'training' section).
        model_fn: Callable returning a fresh model instance.
        dataset: Full dataset with `.subject_ids` attribute and `.samples`.
        device: torch device string.
        output_dir: Where to save per-fold results (JSON).
        logger: Optional logger.

    Returns:
        dict with keys:
            - mse_mean, mse_std
            - pcc_mean, pcc_std
            - per_subject: list of per-fold results
    """
    log = logger or logging.getLogger(__name__)

    # ── Get unique subject IDs ───────────────────────────────
    subject_ids = sorted(set(getattr(dataset, "subject_ids", [])))
    if not subject_ids:
        # Fallback: derive from samples
        subject_ids = sorted(set(s["subject_id"] for s in dataset.samples))

    log.info(f"LOSO: {len(subject_ids)} subjects to iterate")

    per_subject_results: List[Dict] = []

    for i, sid in enumerate(subject_ids):
        log.info(f"[{i + 1}/{len(subject_ids)}] Holding out subject {sid}")

        # ── Build train / test indices ───────────────────────
        train_idx, test_idx = [], []
        for idx, sample in enumerate(dataset.samples):
            if sample["subject_id"] == sid:
                test_idx.append(idx)
            else:
                train_idx.append(idx)

        if len(test_idx) == 0:
            log.warning(f"Subject {sid} has no samples — skipping")
            continue

        train_ds = Subset(dataset, train_idx)
        test_ds = Subset(dataset, test_idx)

        train_loader = DataLoader(
            train_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # ── Train model on all-but-one subject ───────────────
        model = model_fn().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
        )

        best_mse = float("inf")
        for epoch in range(config["training"]["epochs"]):
            model.train()
            for batch in train_loader:
                batch = _to_device(batch, device)
                out = model(batch)
                loss = _compute_loss(out, batch, config)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["gradient_clip"],
                )
                optimizer.step()

            # Quick eval on test subject
            mse = _quick_eval(model, test_loader, device)
            if mse < best_mse:
                best_mse = mse

        # ── Final evaluation on held-out subject ─────────────
        preds, labels, variances = _full_eval(model, test_loader, device)

        mse = float(np.mean((preds - labels) ** 2))
        pcc = float(np.corrcoef(preds, labels)[0, 1]) if len(preds) > 1 else 0.0
        mae = float(np.mean(np.abs(preds - labels)))

        per_subject_results.append({
            "subject_id": int(sid),
            "n_samples": len(test_idx),
            "mse": mse,
            "pcc": pcc,
            "mae": mae,
        })

        log.info(f"  Subject {sid}: MSE={mse:.4f}  PCC={pcc:.4f}")

    # ── Aggregate ────────────────────────────────────────────
    mses = [r["mse"] for r in per_subject_results]
    pccs = [r["pcc"] for r in per_subject_results]
    maes = [r["mae"] for r in per_subject_results]

    summary = {
        "n_subjects": len(per_subject_results),
        "mse_mean": float(np.mean(mses)),
        "mse_std": float(np.std(mses)),
        "pcc_mean": float(np.mean(pccs)),
        "pcc_std": float(np.std(pccs)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "per_subject": per_subject_results,
    }

    log.info(
        f"LOSO complete: MSE = {summary['mse_mean']:.4f} ± "
        f"{summary['mse_std']:.4f} | "
        f"PCC = {summary['pcc_mean']:.4f} ± {summary['pcc_std']:.4f}"
    )

    if output_dir is not None:
        out_path = Path(output_dir) / "loso_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Saved LOSO results to {out_path}")

    return summary


# ── Internal helpers ─────────────────────────────────────────

def _to_device(batch, device):
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def _compute_loss(out, batch, config):
    pred = out["mu"].squeeze()
    target = batch["label"]
    huber = torch.nn.functional.huber_loss(pred, target)
    var = out["var"].squeeze() + 1e-6
    nll = 0.5 * (torch.log(var) + (target - pred) ** 2 / var).mean()
    return huber + config["training"].get("lambda_uq", 0.2) * nll


@torch.no_grad()
def _quick_eval(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch)
        preds.append(out["mu"].squeeze().cpu())
        labels.append(batch["label"].cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return float(np.mean((preds - labels) ** 2))


@torch.no_grad()
def _full_eval(model, loader, device):
    model.eval()
    preds, labels, variances = [], [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch)
        preds.append(out["mu"].squeeze().cpu())
        labels.append(batch["label"].cpu())
        variances.append(out["var"].squeeze().cpu())
    return (
        torch.cat(preds).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(variances).numpy(),
    )
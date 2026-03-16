"""
Main training loop for UAG-MF.

Supports:
  - Single-GPU and multi-GPU (DDP)
  - 5-fold subject-independent cross-validation
  - AMP + gradient clipping (ℓ₂ ≤ 1.0)
  - Early stopping (patience = 15 epochs)
  - TensorBoard logging
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import UAGMFLoss

logger = logging.getLogger("uagmf.trainer")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        output_dir: str,
        device: str = "cuda",
        local_rank: int = 0,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        patience: int = 15,
        save_every: int = 10,
        log_interval: int = 50,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.local_rank = local_rank
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.patience = patience
        self.save_every = save_every
        self.log_interval = log_interval

        self.loss_fn = UAGMFLoss(
            recon_weight=0.5, uq_nll_weight=0.2,
            l1_weight=10.0, perc_weight=1.0, vae_beta=0.5,
        )
        self.scaler = GradScaler(enabled=use_amp)
        self.best_val_mse = float("inf")
        self.epochs_no_improve = 0
        self.global_step = 0

        self._writer = None
        if local_rank == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(str(self.output_dir / "tensorboard"))
            except ImportError:
                pass

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        n_epochs: int = 100,
    ) -> None:
        self.model.to(self.device)
        for epoch in range(1, n_epochs + 1):
            tr_metrics = self._train_epoch(train_loader, optimizer, scheduler, epoch)
            val_metrics = self._val_epoch(val_loader, epoch)

            if self.local_rank == 0:
                logger.info(
                    f"Epoch {epoch}/{n_epochs} | "
                    f"Train MSE: {tr_metrics['mse']:.4f} | "
                    f"Val MSE: {val_metrics['mse']:.4f}"
                )
                if val_metrics["mse"] < self.best_val_mse:
                    self.best_val_mse = val_metrics["mse"]
                    self.epochs_no_improve = 0
                    self._save("best_model.pth", epoch, val_metrics)
                    logger.info(f"  ↳ New best val MSE: {self.best_val_mse:.4f}")
                else:
                    self.epochs_no_improve += 1

                if epoch % self.save_every == 0:
                    self._save(f"epoch_{epoch}.pth", epoch, val_metrics)

            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    def _train_epoch(self, loader, optimizer, scheduler, epoch) -> Dict:
        self.model.train()
        total_mse, n_batches = 0.0, 0
        for batch in tqdm(loader, desc=f"Train E{epoch}", disable=self.local_rank != 0):
            video = batch["video"].to(self.device)
            target = batch["nrs_score"].to(self.device)
            hrv = batch.get("hrv"); spo2 = batch.get("spo2"); resp = batch.get("resp")
            if hrv is not None: hrv = hrv.to(self.device)
            if spo2 is not None: spo2 = spo2.to(self.device)
            if resp is not None: resp = resp.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                out = self.model(video, hrv=hrv, spo2=spo2, resp=resp,
                                 return_intermediates=True, training_recon=True)
                recon_out = out.get("recon_out", {})
                loss, loss_dict = self.loss_fn(
                    pred_mu=out["mu"], pred_var=out["var"], target=target,
                    x_v=batch.get("mid_frame", video[:, video.size(1)//2]).to(self.device),
                    x_hat=recon_out.get("x_hat"),
                    vae_mu=recon_out.get("mu"), vae_logvar=recon_out.get("logvar"),
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(optimizer); self.scaler.update()
            scheduler.step()

            mse_val = float(((out["mu"].detach() - target) ** 2).mean())
            total_mse += mse_val; n_batches += 1; self.global_step += 1

            if self.global_step % self.log_interval == 0 and self._writer:
                self._writer.add_scalar("train/mse", mse_val, self.global_step)
                for k, v in loss_dict.items():
                    self._writer.add_scalar(f"train/{k}", v, self.global_step)

        return {"mse": total_mse / max(n_batches, 1)}

    @torch.no_grad()
    def _val_epoch(self, loader, epoch) -> Dict:
        self.model.eval()
        all_pred, all_target = [], []
        for batch in loader:
            video = batch["video"].to(self.device)
            target = batch["nrs_score"].to(self.device)
            out = self.model(video)
            all_pred.append(out["mu"].cpu())
            all_target.append(target.cpu())
        pred = torch.cat(all_pred); target = torch.cat(all_target)
        mse_val = float(((pred - target) ** 2).mean())
        if self._writer:
            self._writer.add_scalar("val/mse", mse_val, epoch)
        return {"mse": mse_val}

    def _save(self, filename: str, epoch: int, metrics: Dict) -> None:
        torch.save({
            "epoch": epoch, "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "best_val_mse": self.best_val_mse, "metrics": metrics,
        }, self.output_dir / filename)

    def load_checkpoint(self, path: str) -> int:
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self.global_step = ck.get("global_step", 0)
        self.best_val_mse = ck.get("best_val_mse", float("inf"))
        return ck.get("epoch", 0)

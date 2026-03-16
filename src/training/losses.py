"""
Loss functions for UAG-MF.

Composite objective (Eq. 1 from paper):
    L = L_reg + λ₁ L_recon + λ₂ L_UQ
    λ₁ = 0.5,  λ₂ = 0.2

Components:
    L_reg:   Huber regression loss (robust to outliers)
    L_recon: cGAN + VAE reconstruction losses
    L_UQ:    Gaussian negative log-likelihood for uncertainty calibration
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberRegressionLoss(nn.Module):
    """Huber loss for NRS regression (robust to outlier annotations)."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta)


class GaussianNLLLoss(nn.Module):
    """Gaussian negative log-likelihood for uncertainty calibration (L_UQ)."""

    def forward(
        self, mu: torch.Tensor, var: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        var = var.clamp(min=1e-6)
        return (0.5 * torch.log(var) + 0.5 * (target - mu).pow(2) / var).mean()


class UAGMFLoss(nn.Module):
    """
    Combined UAG-MF training loss (Eq. 1).

    L = L_huber + λ₁ L_recon + λ₂ L_UQ_NLL

    Args:
        recon_weight:   λ₁ = 0.5
        uq_nll_weight:  λ₂ = 0.2
        l1_weight:      λ_ℓ1 = 10 for cGAN L1 term
        perc_weight:    λ_perc = 1 for perceptual loss
        vae_beta:       β = 0.5 for VAE KL term
    """

    def __init__(
        self,
        recon_weight: float = 0.5,
        uq_nll_weight: float = 0.2,
        l1_weight: float = 10.0,
        perc_weight: float = 1.0,
        vae_beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.recon_weight = recon_weight
        self.uq_nll_weight = uq_nll_weight
        self.l1_weight = l1_weight
        self.perc_weight = perc_weight
        self.vae_beta = vae_beta

        self.huber = HuberRegressionLoss(delta=1.0)
        self.nll = GaussianNLLLoss()

    def forward(
        self,
        pred_mu: torch.Tensor,
        pred_var: torch.Tensor,
        target: torch.Tensor,
        x_v: Optional[torch.Tensor] = None,
        x_hat: Optional[torch.Tensor] = None,
        vae_mu: Optional[torch.Tensor] = None,
        vae_logvar: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_mu:   (B,) NRS mean predictions.
            pred_var:  (B,) predictive variances.
            target:    (B,) NRS ground truth.
            x_v:       (B, 3, H, W) original frame (for reconstruction loss).
            x_hat:     (B, 3, H, W) reconstructed frame.
            vae_mu:    VAE latent mean (for KL loss).
            vae_logvar: VAE latent log-variance.
        """
        # Primary regression loss
        l_huber = self.huber(pred_mu, target)

        # Reconstruction loss
        l_recon = torch.tensor(0.0, device=pred_mu.device)
        if x_v is not None and x_hat is not None:
            l_l1 = self.l1_weight * F.l1_loss(x_hat, x_v)
            l_perc = self.perc_weight * F.mse_loss(x_hat, x_v)
            l_recon = l_l1 + l_perc

            # VAE KL term if applicable
            if vae_mu is not None and vae_logvar is not None:
                l_kl = -0.5 * (1 + vae_logvar - vae_mu.pow(2) - vae_logvar.exp()).sum(1).mean()
                l_recon = l_recon + self.vae_beta * l_kl

        # UQ calibration loss
        l_uq = self.nll(pred_mu, pred_var, target)

        total = l_huber + self.recon_weight * l_recon + self.uq_nll_weight * l_uq

        return total, {
            "loss_total": total.item(),
            "loss_huber": l_huber.item(),
            "loss_recon": l_recon.item(),
            "loss_uq_nll": l_uq.item(),
        }

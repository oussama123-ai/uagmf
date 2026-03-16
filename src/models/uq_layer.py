"""
Stage ⑥: Dual Uncertainty Quantification Layer.

Implements Algorithm 3 from the paper:
    MC Dropout (S=50 samples) + Deep Ensemble (K=5 members)
    Combined: µ = ½(µ_MC + µ_ens), σ² = ½(σ²_MC + σ²_ens)

Residual inflation (Step 10):
    σ² ← σ² + γρ,  γ = 0.05
    where ρ is the reconstruction residual from Stage ③.

Partial-modality elevation (Step 11):
    σ² ← σ² + δᵤ
    where δᵤ = 0.08 × |Mabs| from Stage ④ Algorithm 2.

Alert threshold (Step 12):
    τ* = 0.35  (Youden-optimal on dev set; TPR=0.91, FPR=0.12)
    Same threshold used by Tier-3 symbolic escalation (Algorithm 4).

The alert criterion σ² > τ* = 0.35 is equivalent to σ > √τ* ≈ 0.59.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_ALERT_THRESHOLD = 0.35   # τ* (variance units)
DEFAULT_GAMMA = 0.05             # residual inflation coefficient
DEFAULT_DELTA_U = 0.08           # per-modality uncertainty penalty


class MCDropoutHead(nn.Module):
    """
    Regression head with dropout active during inference (MC Dropout).

    Produces S stochastic forward passes to estimate predictive variance.
    Dropout probability matches training (0.3 per Table 2).
    """

    def __init__(
        self,
        d_model: int = 256,
        dropout: float = 0.3,
        nrs_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.nrs_max = nrs_max
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B,) NRS prediction ∈ [0, nrs_max]."""
        return torch.sigmoid(self.net(x)).squeeze(-1) * self.nrs_max

    def mc_sample(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run n_samples stochastic forward passes with dropout active.

        Returns:
            mu_mc:    (B,) mean over samples
            var_mc:   (B,) variance over samples
        """
        # Ensure dropout is active even in eval mode
        self.train()
        samples = torch.stack(
            [self.forward(x) for _ in range(n_samples)], dim=1
        )  # (B, S)
        self.eval()
        return samples.mean(1), samples.var(1)


class EnsembleMember(nn.Module):
    """Single member of the Deep Ensemble (dropout OFF at inference)."""

    def __init__(self, d_model: int = 256, nrs_max: float = 10.0) -> None:
        super().__init__()
        self.nrs_max = nrs_max
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.var_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),   # variance must be positive
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu:  (B,) mean prediction ∈ [0, nrs_max]
            var: (B,) predicted variance > 0
        """
        mu = torch.sigmoid(self.mu_head(x)).squeeze(-1) * self.nrs_max
        var = self.var_head(x).squeeze(-1) + 1e-6
        return mu, var


class DualUQLayer(nn.Module):
    """
    Dual UQ Layer combining MC Dropout + Deep Ensemble.

    Implements Algorithm 3 from the paper.

    Args:
        d_model:         Input feature dimension.
        n_mc_samples:    S — number of MC Dropout samples (default 50).
        n_ensemble:      K — number of ensemble members (default 5).
        alert_threshold: τ* — variance threshold for human-in-loop alert (default 0.35).
        gamma:           Residual inflation coefficient γ (default 0.05).
        nrs_max:         Maximum NRS value (default 10.0).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_mc_samples: int = 50,
        n_ensemble: int = 5,
        alert_threshold: float = DEFAULT_ALERT_THRESHOLD,
        gamma: float = DEFAULT_GAMMA,
        nrs_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.n_mc_samples = n_mc_samples
        self.n_ensemble = n_ensemble
        self.alert_threshold = alert_threshold
        self.gamma = gamma
        self.nrs_max = nrs_max

        # MC Dropout branch
        self.mc_head = MCDropoutHead(d_model, dropout=0.3, nrs_max=nrs_max)

        # Deep Ensemble branch (K independent members)
        self.ensemble = nn.ModuleList([
            EnsembleMember(d_model, nrs_max=nrs_max)
            for _ in range(n_ensemble)
        ])

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        delta_u: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Implements Algorithm 3.

        Args:
            x:        (B, d_model) temporal features from Stage ⑤.
            residual: (B,) reconstruction residual ρ from Stage ③ (or None).
            delta_u:  Uncertainty elevation from Stage ④ Algorithm 2.

        Returns:
            dict with:
                "mu":     (B,) combined mean NRS estimate
                "sigma":  (B,) combined standard deviation
                "var":    (B,) combined variance σ²
                "mu_mc":  (B,) MC Dropout mean
                "var_mc": (B,) MC Dropout variance
                "mu_ens": (B,) Ensemble mean
                "var_ens":(B,) Ensemble variance
                "alert":  (B,) bool — True if σ² > τ*
        """
        # ── MC Dropout branch (Steps 1–4) ────────────────────────────────
        mu_mc, var_mc = self.mc_head.mc_sample(x, self.n_mc_samples)

        # ── Deep Ensemble branch (Steps 5–8) ─────────────────────────────
        mu_list, var_list = [], []
        for member in self.ensemble:
            mu_k, var_k = member(x)
            mu_list.append(mu_k)
            var_list.append(var_k)

        mu_ens = torch.stack(mu_list, dim=1).mean(1)             # (B,)
        # Law of total variance: Var = E[σ²] + Var[µ]
        var_ens = (
            torch.stack(var_list, dim=1).mean(1)
            + torch.stack(mu_list, dim=1).var(1)
        )

        # ── Combine (Step 9) ─────────────────────────────────────────────
        mu = 0.5 * (mu_mc + mu_ens)
        var = 0.5 * (var_mc + var_ens)

        # ── Residual inflation (Step 10) ─────────────────────────────────
        if residual is not None:
            var = var + self.gamma * residual

        # ── Partial-modality elevation (Step 11) ──────────────────────────
        var = var + delta_u

        # ── Alert (Steps 12–16) ───────────────────────────────────────────
        alert = var > self.alert_threshold

        return {
            "mu": mu.clamp(0, self.nrs_max),
            "sigma": var.clamp(min=0).sqrt(),
            "var": var,
            "mu_mc": mu_mc,
            "var_mc": var_mc,
            "mu_ens": mu_ens,
            "var_ens": var_ens,
            "alert": alert,
        }

    def gaussian_nll_loss(
        self,
        pred_mu: torch.Tensor,
        pred_var: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gaussian negative log-likelihood loss for UQ training (L_UQ in Eq. 1).
        """
        return (
            0.5 * torch.log(pred_var.clamp(min=1e-6))
            + 0.5 * (target - pred_mu).pow(2) / pred_var.clamp(min=1e-6)
        ).mean()

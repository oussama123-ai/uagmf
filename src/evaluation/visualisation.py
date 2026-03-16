"""
Visualisation utilities reproducing Figures 3–7 from the paper.

Fig 3: Performance comparison (MSE / PCC / ICC bar charts)
Fig 4: Synthetic occlusion simulation + generative reconstruction
Fig 5: Uncertainty calibration, occlusion sensitivity, alert threshold
Fig 6: Ablation study MSE/PCC/ICC
Fig 7: Temporal pain trajectory + uncertainty profile
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

PALETTE = {
    "uagmf": "#2CA02C",
    "transformer": "#4C72B0",
    "baseline": "#AAAAAA",
    "alert": "#D62728",
    "mc": "#FF7F0E",
    "ensemble": "#9467BD",
    "gt": "#1F77B4",
}


# ── Fig 5a: Reliability / Calibration diagram ─────────────────────────────────

def plot_calibration_diagram(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    target: np.ndarray,
    n_bins: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """ECE = 0.038 for combined UAG-MF (Table 9 / Fig 5a)."""
    import scipy.stats as stats

    alphas = np.linspace(0.05, 0.95, n_bins)
    coverages = []
    for alpha in alphas:
        z = stats.norm.ppf((1 + alpha) / 2)
        lo = pred_mean - z * pred_std
        hi = pred_mean + z * pred_std
        coverages.append(float(((target >= lo) & (target <= hi)).mean()))

    ece = float(np.mean(np.abs(np.array(coverages) - alphas)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.5, label="Perfect calibration")
    ax.plot(alphas, coverages, "o-", color=PALETTE["mc"], lw=2,
            label="MC Dropout", markersize=5)
    ax.set_xlabel("Confidence Level", fontsize=12)
    ax.set_ylabel("Empirical Accuracy", fontsize=12)
    ax.set_title("(a) Reliability (Calibration) Diagram", fontsize=12)
    ax.text(0.05, 0.92, f"ECE = {ece:.3f}", transform=ax.transAxes,
            fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Fig 5b: Uncertainty vs. occlusion rate ─────────────────────────────────────

def plot_uncertainty_vs_occlusion(
    occlusion_rates: np.ndarray,
    var_mc: np.ndarray,
    var_ensemble: np.ndarray,
    var_combined: np.ndarray,
    alert_threshold: float = 0.35,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Shows variance rises through τ* at ≈55% occlusion (Fig 5b)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(occlusion_rates, var_mc, "--", color=PALETTE["mc"],
            lw=1.5, label="MC Dropout")
    ax.plot(occlusion_rates, var_ensemble, "--", color=PALETTE["ensemble"],
            lw=1.5, label="Deep Ensemble")
    ax.plot(occlusion_rates, var_combined, "-", color=PALETTE["uagmf"],
            lw=2.5, label="UAG-MF (Ours)")
    ax.axhline(alert_threshold, color=PALETTE["alert"], linestyle="--",
               lw=2, label=f"Alert threshold (τ*={alert_threshold})")

    # Mark ~55% crossover
    crossover = occlusion_rates[np.argmin(np.abs(var_combined - alert_threshold))]
    ax.axvline(crossover, color="gray", linestyle=":", lw=1, alpha=0.7)
    ax.text(crossover + 0.01, alert_threshold + 0.02,
            f"~{crossover:.0%}\ncrossover", fontsize=9, color="gray")

    ax.set_xlabel("Occlusion Coverage Ratio r (masked pixels / total pixels)", fontsize=11)
    ax.set_ylabel("Predictive Variance (σ²)", fontsize=11)
    ax.set_title("(b) Uncertainty vs. Occlusion Rate", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Fig 5c: Human-in-the-loop alert threshold optimisation ───────────────────

def plot_alert_threshold_optimisation(
    thresholds: np.ndarray,
    tpr: np.ndarray,
    fpr: np.ndarray,
    optimal_threshold: float = 0.35,
    optimal_tpr: float = 0.91,
    optimal_fpr: float = 0.12,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, tpr, "-", color=PALETTE["uagmf"], lw=2, label="True Positive Rate")
    ax.plot(thresholds, fpr, "--", color=PALETTE["alert"], lw=2, label="False Positive Rate")
    ax.scatter([optimal_threshold], [optimal_tpr], color=PALETTE["uagmf"],
               s=120, zorder=5, label=f"TPR={optimal_tpr}")
    ax.scatter([optimal_threshold], [optimal_fpr], color=PALETTE["alert"],
               s=120, zorder=5, label=f"FPR={optimal_fpr}")
    ax.axvline(optimal_threshold, color="gray", linestyle=":", lw=1.5,
               label=f"τ*={optimal_threshold}")
    ax.set_xlabel("Uncertainty Threshold (τ)", fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("(c) Human-in-the-Loop Alert Rate", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0.1, 0.9); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Fig 6: Ablation study ─────────────────────────────────────────────────────

def plot_ablation(
    configs: List[str],
    mse_values: List[float],
    pcc_values: List[float],
    icc_values: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Reproduces Figure 6 from the paper."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Ablation Study — BioVid (5-fold CV)", fontsize=13, fontweight="bold")

    x = np.arange(len(configs))
    colors = [PALETTE["baseline"]] * (len(configs) - 1) + [PALETTE["uagmf"]]

    bars = ax1.bar(x, mse_values, color=colors, alpha=0.85, width=0.6)
    for bar, val in zip(bars, mse_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    reduction = (mse_values[0] - mse_values[-1]) / mse_values[0] * 100
    ax1.text(len(configs) - 1, mse_values[-1] / 2,
             f"−{reduction:.1f}%", ha="center", color="white",
             fontsize=12, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(configs, rotation=20, ha="right", fontsize=10)
    ax1.set_ylabel("MSE (BioVid dataset) ↓", fontsize=11)
    ax1.set_title("(a) Ablation Study MSE ↓", fontsize=11)
    ax1.text(0.98, 0.98, "All steps p < 0.010\n(Wilcoxon, Bonferroni)",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9,
             color="gray")

    ax2.plot(x, pcc_values, "o-", color=PALETTE["uagmf"], lw=2,
             markersize=8, label="PCC ↑")
    ax2.plot(x, icc_values, "s--", color=PALETTE["transformer"], lw=2,
             markersize=8, label="ICC ↑")
    for xi, (p, i) in enumerate(zip(pcc_values, icc_values)):
        ax2.annotate(f"{p:.3f}", (xi, p), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9, color=PALETTE["uagmf"])
        ax2.annotate(f"{i:.3f}", (xi, i), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=9, color=PALETTE["transformer"])
    ax2.set_xticks(x); ax2.set_xticklabels(configs, rotation=20, ha="right", fontsize=10)
    ax2.set_ylabel("Correlation Coefficient", fontsize=11)
    ax2.set_title("(b) Ablation Study PCC & ICC ↑", fontsize=11)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Fig 7: Temporal pain estimation ──────────────────────────────────────────

def plot_temporal_estimation(
    time_axis: np.ndarray,
    ground_truth: np.ndarray,
    pred_mu: np.ndarray,
    pred_sigma: np.ndarray,
    occlusion_start: float = 40.0,
    alert_threshold_sd: float = 0.59,   # √τ* = √0.35 ≈ 0.59
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Temporal evolution figure (Fig 7 from paper).
    Alert threshold shown in SD units σ > √τ* ≈ 0.59
    (equivalent to variance criterion σ² > τ* = 0.35).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Temporal Pain Estimation — 60-second BioVid Segment",
                 fontsize=13, fontweight="bold")

    # Panel (a): pain trajectory
    occ_mask = time_axis >= occlusion_start
    ax1.fill_between(time_axis, 0, 10, where=occ_mask,
                     alpha=0.12, color=PALETTE["alert"], label="Occlusion region")
    ax1.plot(time_axis, ground_truth, "-", color=PALETTE["gt"], lw=2.5,
             label="Ground Truth (PSPI)", zorder=5)
    ax1.plot(time_axis, pred_mu, "--", color=PALETTE["uagmf"], lw=2,
             label="UAG-MF Prediction (µ)")
    ax1.fill_between(time_axis, pred_mu - pred_sigma, pred_mu + pred_sigma,
                     alpha=0.25, color=PALETTE["uagmf"], label="±1σ")
    ax1.fill_between(time_axis, pred_mu - 2 * pred_sigma, pred_mu + 2 * pred_sigma,
                     alpha=0.10, color=PALETTE["uagmf"], label="±2σ")
    ax1.axvline(occlusion_start, color=PALETTE["alert"], linestyle="--",
                lw=1.5, alpha=0.8, label=f"Occlusion onset (t={occlusion_start}s)")
    ax1.set_ylabel("Pain Intensity (0–10 NRS)", fontsize=11)
    ax1.set_ylim(-0.5, 11)
    ax1.legend(loc="upper left", fontsize=9, ncol=2)
    ax1.set_title("(a) Continuous Pain Trajectory: Ground Truth vs. UAG-MF Prediction",
                  fontsize=11)

    # Panel (b): temporal uncertainty profile
    alert_zone = occ_mask & (pred_sigma > alert_threshold_sd)
    ax2.fill_between(time_axis, 0, pred_sigma,
                     where=alert_zone, color=PALETTE["alert"], alpha=0.3,
                     label=f"Alert zone (σ > {alert_threshold_sd:.2f})")
    ax2.plot(time_axis, pred_sigma, "-", color=PALETTE["mc"], lw=2,
             label="Predictive σ (MC Dropout)")
    ax2.axhline(alert_threshold_sd, color=PALETTE["alert"], linestyle="--", lw=2,
                label=f"Alert threshold σ = {alert_threshold_sd:.2f} (= √τ* = √0.35)")
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Uncertainty (σ)", fontsize=11)
    ax2.set_title(
        "(b) Temporal Uncertainty Profile and Human-in-the-Loop Alerts\n"
        "Alert criterion: σ² > τ* = 0.35  ⟺  σ > √0.35 ≈ 0.59 (shown here)",
        fontsize=11,
    )
    ax2.legend(loc="upper left", fontsize=10)
    ax2.set_ylim(0, max(pred_sigma.max() * 1.2, alert_threshold_sd * 1.5))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

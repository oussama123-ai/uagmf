"""
Evaluation metrics for UAG-MF.

Primary:   MSE
Secondary: RMSE, MAE, PCC, ICC(3,1), QWK, ECE

QWK thresholds (clinically grounded NRS bands per Farrar et al. 2001):
    No pain:  ŷ < 1.5
    Mild:     1.5 ≤ ŷ < 3.5
    Moderate: 3.5 ≤ ŷ < 6.5
    Severe:   ŷ ≥ 6.5

Statistical tests: Wilcoxon signed-rank, Bonferroni correction (α=0.05).
Bootstrap CI: B=1000.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats


# ── Point metrics ─────────────────────────────────────────────────────────────

def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(pred - target).mean())


def pearson_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    if np.std(pred) < 1e-9 or np.std(target) < 1e-9:
        return 0.0
    r, _ = stats.pearsonr(pred, target)
    return float(r)


def intraclass_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """ICC(3,1) — two-way mixed, single measures, absolute agreement."""
    try:
        import pingouin as pg
        import pandas as pd
        n = len(pred)
        df = pd.DataFrame({
            "rater": ["system"] * n + ["ground_truth"] * n,
            "subject": list(range(n)) * 2,
            "score": list(pred) + list(target),
        })
        res = pg.intraclass_corr(data=df, targets="subject",
                                 raters="rater", ratings="score")
        icc31 = res.loc[res["Type"] == "ICC3", "ICC"].values
        return float(icc31[0]) if len(icc31) > 0 else _icc_fallback(pred, target)
    except Exception:
        return _icc_fallback(pred, target)


def _icc_fallback(pred: np.ndarray, target: np.ndarray) -> float:
    n = len(pred)
    grand = np.concatenate([pred, target]).mean()
    ss_b = np.sum(((pred + target) / 2 - grand) ** 2) * 2 / (n - 1)
    ss_w = np.sum((pred - target) ** 2) / (2 * n)
    if ss_b + ss_w < 1e-12:
        return 0.0
    return float((ss_b - ss_w) / (ss_b + ss_w))


def quadratic_weighted_kappa(
    pred: np.ndarray,
    target: np.ndarray,
    thresholds: Tuple[float, float, float] = (1.5, 3.5, 6.5),
) -> float:
    """
    QWK with 4 discrete pain levels (clinically grounded NRS bands).
    Thresholds: < 1.5 (no pain), 1.5–3.5 (mild), 3.5–6.5 (moderate), ≥ 6.5 (severe).
    """
    from sklearn.metrics import cohen_kappa_score

    def to_class(arr):
        c = np.zeros(len(arr), dtype=int)
        c[(arr >= thresholds[0]) & (arr < thresholds[1])] = 1
        c[(arr >= thresholds[1]) & (arr < thresholds[2])] = 2
        c[arr >= thresholds[2]] = 3
        return c

    try:
        return float(cohen_kappa_score(to_class(target), to_class(pred), weights="quadratic"))
    except Exception:
        return 0.0


def expected_calibration_error(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    target: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    ECE for regression via confidence interval coverage.

    For each confidence level α ∈ [0,1], compute the fraction of
    targets falling within the predicted α-interval. ECE = mean |coverage - α|.
    Paper reports ECE = 0.038 (combined UAG-MF).
    """
    alphas = np.linspace(0.05, 0.95, n_bins)
    ece = 0.0
    for alpha in alphas:
        z = stats.norm.ppf((1 + alpha) / 2)
        lower = pred_mean - z * pred_std
        upper = pred_mean + z * pred_std
        coverage = float(((target >= lower) & (target <= upper)).mean())
        ece += abs(coverage - alpha)
    return float(ece / n_bins)


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    pred: np.ndarray,
    target: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(pred)
    scores = [metric_fn(pred[idx := rng.randint(0, n, n)], target[idx])
              for _ in range(n_bootstrap)]
    a = (1 - confidence) / 2
    return (metric_fn(pred, target),
            float(np.percentile(scores, 100 * a)),
            float(np.percentile(scores, 100 * (1 - a))))


# ── Statistical tests ─────────────────────────────────────────────────────────

def wilcoxon_bonferroni(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_comparisons: int = 6,
) -> Tuple[float, float]:
    """Wilcoxon signed-rank test with Bonferroni correction."""
    stat, p = stats.wilcoxon(errors_a, errors_b, alternative="two-sided")
    return float(stat), float(min(p * n_comparisons, 1.0))


# ── Combined metric computation ───────────────────────────────────────────────

def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    pred_std: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)

    results: Dict[str, float] = {
        "mse": mse(pred, target),
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "pcc": pearson_correlation(pred, target),
        "icc": intraclass_correlation(pred, target),
        "qwk": quadratic_weighted_kappa(pred, target),
    }

    if pred_std is not None:
        results["ece"] = expected_calibration_error(pred, pred_std, target)

    if n_bootstrap > 0:
        _, lo, hi = bootstrap_ci(pred, target, mse, n_bootstrap)
        results["mse_ci_lower"] = lo
        results["mse_ci_upper"] = hi

    return results


def format_results_table(
    method_results: Dict[str, Dict[str, float]],
    primary: str = "UAG-MF (Ours)",
) -> str:
    header = f"{'Method':<35} {'MSE':>7} {'PCC':>7} {'ICC':>7} {'QWK':>7} {'ECE':>7}"
    sep = "-" * len(header)
    lines = [header, sep]
    primary_mse = method_results.get(primary, {}).get("mse", None)
    for name, m in method_results.items():
        improvement = ""
        if primary_mse and name != primary:
            gain = (m.get("mse", 0) - primary_mse) / m.get("mse", 1) * 100
            improvement = f"  (+{gain:.1f}%)"
        lines.append(
            f"{name:<35} "
            f"{m.get('mse', float('nan')):>7.3f} "
            f"{m.get('pcc', float('nan')):>7.3f} "
            f"{m.get('icc', float('nan')):>7.3f} "
            f"{m.get('qwk', float('nan')):>7.3f} "
            f"{m.get('ece', float('nan')):>7.3f}"
            f"{improvement}"
        )
    return "\n".join(lines)

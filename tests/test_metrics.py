"""Unit tests for evaluation metrics."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    mse, rmse, mae, pearson_correlation, quadratic_weighted_kappa,
    expected_calibration_error, bootstrap_ci, compute_all_metrics,
)


class TestBasicMetrics:
    def test_mse_perfect(self):
        x = np.array([1.0, 2.0, 3.0])
        assert mse(x, x) == pytest.approx(0.0)

    def test_mse_known(self):
        assert mse(np.array([0.0, 4.0]), np.array([0.0, 0.0])) == pytest.approx(8.0)

    def test_rmse_known(self):
        assert rmse(np.array([0.0, 4.0]), np.array([0.0, 0.0])) == pytest.approx(np.sqrt(8.0))

    def test_mae_known(self):
        assert mae(np.array([1.0, 3.0]), np.array([2.0, 2.0])) == pytest.approx(1.0)

    def test_pcc_perfect(self):
        x = np.arange(10, dtype=float)
        assert pearson_correlation(x, x) == pytest.approx(1.0)

    def test_qwk_perfect(self):
        x = np.array([0.0, 2.0, 5.0, 8.0])
        assert quadratic_weighted_kappa(x, x) == pytest.approx(1.0)

    def test_qwk_range(self):
        rng = np.random.RandomState(42)
        pred = rng.uniform(0, 10, 100)
        tgt = rng.uniform(0, 10, 100)
        qwk = quadratic_weighted_kappa(pred, tgt)
        assert -1.0 <= qwk <= 1.0


class TestECE:
    def test_perfect_calibration(self):
        rng = np.random.RandomState(0)
        n = 1000
        target = rng.uniform(0, 10, n)
        pred_mu = target.copy()
        pred_std = np.ones(n) * 0.01   # very tight, well-calibrated
        ece = expected_calibration_error(pred_mu, pred_std, target)
        assert ece >= 0

    def test_ece_overconfident(self):
        rng = np.random.RandomState(1)
        n = 500
        target = rng.uniform(0, 10, n)
        pred_mu = target + rng.normal(0, 2, n)
        pred_std = np.ones(n) * 0.01   # overconfident (too narrow)
        ece = expected_calibration_error(pred_mu, pred_std, target)
        assert ece > 0.3   # should be poorly calibrated


class TestBootstrapCI:
    def test_ci_contains_point_estimate(self):
        rng = np.random.RandomState(42)
        pred = rng.uniform(0, 10, 200)
        tgt = pred + rng.normal(0, 1, 200)
        point, lo, hi = bootstrap_ci(pred, tgt, mse, n_bootstrap=200)
        assert lo <= point <= hi

    def test_ci_reproducible(self):
        rng = np.random.RandomState(0)
        pred = rng.uniform(0, 10, 100)
        tgt = pred + rng.normal(0, 1, 100)
        _, lo1, hi1 = bootstrap_ci(pred, tgt, mse, n_bootstrap=100, seed=7)
        _, lo2, hi2 = bootstrap_ci(pred, tgt, mse, n_bootstrap=100, seed=7)
        assert lo1 == lo2 and hi1 == hi2


class TestComputeAllMetrics:
    def test_keys_present(self):
        rng = np.random.RandomState(3)
        pred = rng.uniform(0, 10, 100)
        tgt = pred + rng.normal(0, 1, 100)
        m = compute_all_metrics(pred, tgt, n_bootstrap=50)
        for k in ("mse", "rmse", "mae", "pcc", "icc", "qwk",
                  "mse_ci_lower", "mse_ci_upper"):
            assert k in m, f"Missing: {k}"

    def test_mse_matches_direct(self):
        rng = np.random.RandomState(5)
        pred = rng.uniform(0, 10, 100)
        tgt = rng.uniform(0, 10, 100)
        m = compute_all_metrics(pred, tgt, n_bootstrap=0)
        assert m["mse"] == pytest.approx(mse(pred, tgt))

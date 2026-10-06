"""Unit checks for Z-score, KMO/MSA, and Bartlett sphericity."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from src.factor_analysis.suitability import (
    bartlett_sphericity,
    evaluate_suitability,
    kaiser_meyer_olkin,
    load_questionnaire_matrix,
    pearson_correlation,
    plot_correlation_heatmap,
    run_suitability_pipeline,
    zscore_standardize,
)


class SuitabilityMathTests(unittest.TestCase):
    """Algebraic identities that do not depend on the contest CSV."""

    def test_zscore_mean_std(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(loc=4.0, scale=2.0, size=(80, 6))
        z = zscore_standardize(x)
        np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1.0e-12)
        np.testing.assert_allclose(z.std(axis=0, ddof=0), 1.0, atol=1.0e-12)

    def test_identity_correlation_bartlett_near_null(self) -> None:
        corr = np.eye(4)
        chi2_stat, df, p_value, log_det = bartlett_sphericity(corr, n_obs=40)
        self.assertEqual(df, 6)
        self.assertAlmostEqual(log_det, 0.0, places=12)
        self.assertAlmostEqual(chi2_stat, 0.0, places=12)
        self.assertGreater(p_value, 0.99)

    def test_identity_kmo_undefined_or_zero_partials(self) -> None:
        kmo, msa = kaiser_meyer_olkin(np.eye(5))
        self.assertTrue(np.isnan(kmo))
        self.assertTrue(np.all(np.isnan(msa)))

    def test_one_factor_rejects_sphericity(self) -> None:
        rng = np.random.default_rng(7)
        n, p = 120, 8
        f = rng.normal(size=(n, 1))
        x = 1.6 * f + 0.35 * rng.normal(size=(n, p))
        report = evaluate_suitability(
            x, tuple(f"v{i}" for i in range(p))
        )
        self.assertGreater(report.kmo_overall, 0.70)
        self.assertLess(report.bartlett_p, 1.0e-3)
        self.assertEqual(report.bartlett_df, p * (p - 1) // 2)
        np.testing.assert_allclose(np.diag(report.corr), 1.0, atol=1.0e-12)

    def test_msa_in_unit_interval_when_defined(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(60, 5))
        x[:, 1] = 0.7 * x[:, 0] + 0.3 * x[:, 1]
        corr = pearson_correlation(zscore_standardize(x))
        kmo, msa = kaiser_meyer_olkin(corr)
        self.assertTrue(np.isfinite(kmo))
        self.assertTrue(0.0 <= kmo <= 1.0)
        self.assertTrue(np.all(np.isfinite(msa)))
        self.assertTrue(np.all((msa >= 0.0) & (msa <= 1.0)))


class SuitabilityIoTests(unittest.TestCase):
    """CSV contract and heatmap write."""

    def test_pack_csv_is_50_by_15(self) -> None:
        x, names = load_questionnaire_matrix()
        self.assertEqual(x.shape, (50, 15))
        self.assertEqual(len(names), 15)

    def test_heatmap_png_written(self) -> None:
        corr = np.eye(3)
        corr[0, 1] = corr[1, 0] = 0.4
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corr.png"
            plot_correlation_heatmap(corr, ("a", "b", "c"), path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 100)

    def test_pipeline_writes_pack_figure(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with tempfile.TemporaryDirectory() as tmp:
                fig = Path(tmp) / "fig_correlation_matrix.png"
                report = run_suitability_pipeline(
                    figure_path=fig,
                    assert_bartlett=False,
                )
                self.assertTrue(fig.is_file())
                self.assertGreater(fig.stat().st_size, 100)
                self.assertEqual(report.n_obs, 50)
                self.assertEqual(report.n_var, 15)
                self.assertEqual(report.corr.shape, (15, 15))


if __name__ == "__main__":
    unittest.main()

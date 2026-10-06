"""Algebraic checks for PC-EFA, Varimax, and Thompson scores."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.factor_analysis.efa_engine import (
    CONTEST_N_FACTORS,
    eigendecompose_correlation,
    extract_efa,
    kaiser_and_cumulative_rank,
    loadings_frame,
    plot_scree,
    run_efa_pipeline,
    thompson_factor_scores,
    thompson_score_weights,
    unrotated_pc_loadings,
    varimax_criterion,
    varimax_rotate,
)
from src.factor_analysis.suitability import pearson_correlation, zscore_standardize


class SpectralTests(unittest.TestCase):
    """Eigenvalues of a correlation matrix."""

    def test_identity_all_ones(self) -> None:
        eig, vec = eigendecompose_correlation(np.eye(5))
        np.testing.assert_allclose(eig, np.ones(5), atol=1.0e-12)
        np.testing.assert_allclose(vec.T @ vec, np.eye(5), atol=1.0e-12)
        n_kaiser, n_cum, cum = kaiser_and_cumulative_rank(eig)
        self.assertEqual(n_kaiser, 0)
        self.assertEqual(n_cum, 4)
        np.testing.assert_allclose(cum[-1], 1.0, atol=1.0e-12)

    def test_loadings_reproduce_rank_m_gram(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(80, 6))
        x[:, 1] += 0.8 * x[:, 0]
        x[:, 3] += 0.8 * x[:, 2]
        corr = pearson_correlation(zscore_standardize(x))
        eig, vec = eigendecompose_correlation(corr)
        lam = unrotated_pc_loadings(eig, vec, n_factors=2)
        recon = lam @ lam.T
        spectral = (vec[:, :2] * eig[:2]) @ vec[:, :2].T
        np.testing.assert_allclose(recon, spectral, atol=1.0e-10)


class VarimaxTests(unittest.TestCase):
    """Orthogonal rotation properties."""

    def test_t_orthogonal_and_criterion_nondecreasing(self) -> None:
        rng = np.random.default_rng(2)
        n, p = 200, 9
        factors = rng.normal(size=(n, 3))
        true_load = np.zeros((p, 3))
        true_load[0:3, 0] = 0.9
        true_load[3:6, 1] = 0.9
        true_load[6:9, 2] = 0.9
        x = factors @ true_load.T + 0.15 * rng.normal(size=(n, p))
        z = zscore_standardize(x)
        corr = pearson_correlation(z)
        eig, vec = eigendecompose_correlation(corr)
        unrot = unrotated_pc_loadings(eig, vec, n_factors=3)
        rotated, t_mat = varimax_rotate(unrot)
        np.testing.assert_allclose(t_mat.T @ t_mat, np.eye(3), atol=1.0e-8)
        np.testing.assert_allclose(rotated, unrot @ t_mat, atol=1.0e-10)
        self.assertGreaterEqual(
            varimax_criterion(rotated) + 1.0e-10,
            varimax_criterion(unrot),
        )
        primary = np.argmax(np.abs(rotated), axis=1)
        for block, expected in ((slice(0, 3), primary[0]), (slice(3, 6), primary[3]), (slice(6, 9), primary[6])):
            self.assertTrue(np.all(primary[block] == expected))

    def test_single_factor_is_identity_rotation(self) -> None:
        loadings = np.array([[0.8], [0.6], [0.4]])
        rotated, t_mat = varimax_rotate(loadings)
        np.testing.assert_allclose(rotated, loadings)
        np.testing.assert_allclose(t_mat, np.eye(1))


class ThompsonTests(unittest.TestCase):
    """W = R^{-1} Lambda^* and F = Z W."""

    def test_weights_and_scores_shapes(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(size=(50, 8))
        z = zscore_standardize(x)
        corr = pearson_correlation(z)
        eig, vec = eigendecompose_correlation(corr)
        unrot = unrotated_pc_loadings(eig, vec, n_factors=3)
        rotated, _ = varimax_rotate(unrot)
        weights = thompson_score_weights(corr, rotated)
        scores = thompson_factor_scores(z, weights)
        self.assertEqual(weights.shape, (8, 3))
        self.assertEqual(scores.shape, (50, 3))
        np.testing.assert_allclose(scores, z @ np.linalg.inv(corr) @ rotated, atol=1.0e-10)


class PackPipelineTests(unittest.TestCase):
    """Contest CSV contract: N=50, p=15, m=3."""

    def test_extract_on_pack_questionnaire(self) -> None:
        from src.factor_analysis.suitability import load_questionnaire_matrix, evaluate_suitability

        x, names = load_questionnaire_matrix()
        suit = evaluate_suitability(x, names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = extract_efa(suit.z, suit.corr, names, n_factors=CONTEST_N_FACTORS)
        self.assertEqual(result.n_obs, 50)
        self.assertEqual(result.n_var, 15)
        self.assertEqual(result.n_factors, 3)
        self.assertEqual(result.rotated_loadings.shape, (15, 3))
        self.assertEqual(result.factor_scores.shape, (50, 3))
        self.assertEqual(result.eigenvalues.shape, (15,))
        np.testing.assert_allclose(
            np.sum(result.eigenvalues),
            15.0,
            atol=1.0e-8,
        )
        frame = loadings_frame(result)
        self.assertEqual(len(frame), 15)
        self.assertTrue(set(frame["primary_factor"].tolist()) <= {1, 2, 3})

    def test_pipeline_writes_artifacts(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                loadings = root / "factor_loadings.csv"
                scores = root / "student_factor_scores.csv"
                scree = root / "fig_scree_plot.png"
                result = run_efa_pipeline(
                    loadings_path=loadings,
                    scores_path=scores,
                    scree_path=scree,
                )
                self.assertTrue(loadings.is_file())
                self.assertTrue(scores.is_file())
                self.assertTrue(scree.is_file())
                self.assertGreater(scree.stat().st_size, 100)
                load_df = pd.read_csv(loadings)
                score_df = pd.read_csv(scores)
                self.assertEqual(len(load_df), 15)
                self.assertEqual(len(score_df), 50)
                self.assertEqual(result.factor_scores.shape, (50, 3))
                for col in ("Factor_1", "Factor_2", "Factor_3", "primary_factor"):
                    self.assertIn(col, load_df.columns)
                for col in ("student_id", "Factor_1", "Factor_2", "Factor_3"):
                    self.assertIn(col, score_df.columns)

    def test_scree_png_written(self) -> None:
        eig = np.array([4.2, 2.1, 1.3, 0.8, 0.6, 0.4], dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scree.png"
            plot_scree(eig, n_factors=3, out_path=path, n_kaiser=3)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()

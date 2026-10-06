"""Unit checks for min-max shift, entropy identities, and job utility polarity."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.entropy_weight.entropy_engine import (
    SHIFT_EPS,
    column_shares,
    compute_entropy_weights,
    information_entropy,
    job_utility,
    load_job_factor_matrix,
    load_student_factor_scores,
    run_entropy_pipeline,
    shift_minmax,
    signed_utility_weights,
)


class ShiftAndShareTests(unittest.TestCase):
    def test_shift_range(self) -> None:
        rng = np.random.default_rng(0)
        f = rng.normal(size=(40, 3))
        y, col_min, col_max = shift_minmax(f)
        np.testing.assert_allclose(col_min, f.min(axis=0))
        np.testing.assert_allclose(col_max, f.max(axis=0))
        self.assertTrue(np.all(y >= SHIFT_EPS - 1.0e-15))
        self.assertTrue(np.all(y <= 1.0 + SHIFT_EPS + 1.0e-12))

    def test_degenerate_column_has_unit_entropy(self) -> None:
        f = np.column_stack(
            [
                np.linspace(-1.0, 1.0, 20),
                np.ones(20),
                np.linspace(0.0, 3.0, 20),
            ]
        )
        y, _, _ = shift_minmax(f)
        p = column_shares(y)
        np.testing.assert_allclose(p.sum(axis=0), np.ones(3), atol=1.0e-12)
        entropy = information_entropy(p)
        self.assertAlmostEqual(float(entropy[1]), 1.0, places=10)

    def test_uniform_shares_entropy_one(self) -> None:
        p = np.full((10, 1), 0.1)
        e = information_entropy(p)
        self.assertAlmostEqual(float(e[0]), 1.0, places=12)


class WeightAndUtilityTests(unittest.TestCase):
    def test_weights_sum_to_one_on_pack_students(self) -> None:
        scores = load_student_factor_scores()
        self.assertEqual(scores.shape, (50, 3))
        _y, p, entropy, divergence, weights, _mm = compute_entropy_weights(scores)
        np.testing.assert_allclose(p.sum(axis=0), np.ones(3), atol=1.0e-12)
        self.assertTrue(np.all((entropy >= 0.0) & (entropy <= 1.0)))
        np.testing.assert_allclose(divergence, 1.0 - entropy)
        np.testing.assert_allclose(float(weights.sum()), 1.0, atol=1.0e-12)
        self.assertTrue(np.all(weights >= 0.0))

    def test_signed_third_is_cost(self) -> None:
        w = np.array([0.2, 0.3, 0.5])
        s = signed_utility_weights(w)
        np.testing.assert_allclose(s, [0.2, 0.3, -0.5])

    def test_utility_matches_formula(self) -> None:
        jobs = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
        signed = np.array([0.2, 0.3, -0.5])
        u = job_utility(jobs, signed)
        np.testing.assert_allclose(u, [0.2 * 1 + 0.3 * 2 - 0.5 * 3, -0.5])


class PipelineIoTests(unittest.TestCase):
    def test_eight_jobs_on_disk(self) -> None:
        ids, factors, titles, names = load_job_factor_matrix()
        self.assertEqual(ids.tolist(), list(range(8)))
        self.assertEqual(factors.shape, (8, 3))
        self.assertEqual(len(titles), 8)
        self.assertEqual(len(names), 3)

    def test_pipeline_writes_csv_and_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "entropy_weights.csv"
            utility = root / "job_entropy_utility.csv"
            fig = root / "fig_entropy_weights.png"
            result = run_entropy_pipeline(
                weights_path=weights,
                utility_path=utility,
                figure_path=fig,
            )
            self.assertTrue(weights.is_file())
            self.assertTrue(utility.is_file())
            self.assertTrue(fig.is_file())
            self.assertGreater(fig.stat().st_size, 100)
            w_df = pd.read_csv(weights)
            u_df = pd.read_csv(utility)
            self.assertEqual(len(w_df), 3)
            self.assertEqual(len(u_df), 8)
            self.assertAlmostEqual(float(w_df["weight_w"].sum()), 1.0, places=10)
            self.assertEqual(w_df.loc[w_df["factor"] == "Factor_3", "direction"].iloc[0], "cost")
            np.testing.assert_allclose(result.n_obs, 50)
            pet = u_df.loc[u_df["title_en"] == "Pet Sitter", "utility"].iloc[0]
            guard = u_df.loc[u_df["title_en"] == "Lifeguard", "utility"].iloc[0]
            self.assertGreater(float(pet), float(guard))


if __name__ == "__main__":
    unittest.main()

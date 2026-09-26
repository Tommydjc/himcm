"""AHP 特征值法：一致矩阵精确权重，以及高 CR 时的警告。"""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcda.ahp import compute_ahp_weights


def _consistent_reciprocal(weights: np.ndarray) -> np.ndarray:
    """由正权重构造完全一致互反矩阵 ``a_ij = w_i / w_j``，此时 ``λ_max = n``。"""
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w <= 0.0):
        raise ValueError("seed weights must be positive")
    return np.outer(w, 1.0 / w)


class TestAHPConsistencyRatio(unittest.TestCase):
    """3×3 与 6×6 标准一致矩阵；不一致矩阵触发 UserWarning 且 CR≥0.1。"""

    def test_3x3_consistent_rank_one_weights_sum_to_one(self) -> None:
        true_w = np.array([0.2, 0.3, 0.5], dtype=np.float64)
        matrix = _consistent_reciprocal(true_w)
        weights, cr, ok = compute_ahp_weights(matrix)
        self.assertEqual(weights.shape, (3,))
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(weights, true_w, atol=1.0e-10)
        self.assertTrue(ok)
        self.assertLess(cr, 0.10)
        self.assertAlmostEqual(cr, 0.0, places=8)

    def test_6x6_consistent_rank_one_matches_seed_weights(self) -> None:
        true_w = np.array([0.30, 0.20, 0.15, 0.15, 0.12, 0.08], dtype=np.float64)
        true_w = true_w / np.sum(true_w)
        matrix = _consistent_reciprocal(true_w)
        self.assertEqual(matrix.shape, (6, 6))
        weights, cr, ok = compute_ahp_weights(matrix)
        self.assertEqual(weights.shape, (6,))
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(weights, true_w, atol=1.0e-9)
        self.assertTrue(ok)
        self.assertAlmostEqual(cr, 0.0, places=6)

    def test_inconsistent_matrix_warns_and_fails_cr(self) -> None:
        inconsistent = np.array(
            [
                [1.0, 9.0, 9.0],
                [1.0 / 9.0, 1.0, 9.0],
                [1.0 / 9.0, 1.0 / 9.0, 1.0],
            ],
            dtype=np.float64,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            weights, cr, ok = compute_ahp_weights(inconsistent)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=12)
        self.assertFalse(ok)
        self.assertGreaterEqual(cr, 0.10)
        self.assertTrue(any(issubclass(item.category, UserWarning) for item in caught))
        self.assertTrue(any("consistency failed" in str(item.message) for item in caught))

    def test_rejects_non_reciprocal(self) -> None:
        bad = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            compute_ahp_weights(bad)


if __name__ == "__main__":
    unittest.main()

"""AHP 与 ML 权重：L1 归一、余弦、Spearman。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.importance import cosine_similarity, l1_normalize, spearman_rho


class TestWeightAgreement(unittest.TestCase):
    """归一化和相似度的确定性检查。"""

    def test_l1_sums_to_one(self) -> None:
        w = l1_normalize(np.array([2.0, -1.0, 1.0], dtype=np.float64))
        np.testing.assert_allclose(float(np.sum(w)), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(w, np.array([0.5, 0.25, 0.25]))

    def test_cosine_identical_is_one(self) -> None:
        v = np.array([0.22, 0.18, 0.16], dtype=np.float64)
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=12)

    def test_spearman_monotone(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(spearman_rho(a, b), 1.0, places=8)

    def test_six_criterion_weights_drop_inertia(self) -> None:
        from src.ml.dataset import FEATURE_COLS
        from src.ml.importance import logistic_criterion_weights

        coef = np.arange(len(FEATURE_COLS), dtype=np.float64)
        w, beta_six = logistic_criterion_weights(coef, FEATURE_COLS)
        self.assertEqual(int(beta_six.size), 6)
        np.testing.assert_allclose(float(np.sum(w)), 1.0, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()

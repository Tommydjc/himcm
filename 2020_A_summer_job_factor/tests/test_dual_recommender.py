"""Softmax probabilities, MLP parameter count, and pack LOOCV I/O."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.classification.dual_recommender import (
    N_CLASSES,
    N_FEATURES,
    TwoLayerRecommender,
    align_features_and_labels,
    fit_linear_softmax,
    run_dual_recommender_pipeline,
    stable_softmax,
    softmax_logits,
    top_k_hit,
)


class SoftmaxMathTests(unittest.TestCase):
    def test_probs_sum_to_one(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(12, 3))
        w = rng.normal(size=(8, 3))
        b = rng.normal(size=(8,))
        p = stable_softmax(softmax_logits(x, w, b))
        self.assertEqual(p.shape, (12, 8))
        np.testing.assert_allclose(p.sum(axis=1), np.ones(12), atol=1.0e-12)

    def test_softmax_fits_separable_toy(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(40, 3))
        y = np.argmax(x, axis=1) % 8
        w, b = fit_linear_softmax(x, y, epochs=250, step_size=0.4, seed=1)
        pred = np.argmax(stable_softmax(softmax_logits(x, w, b)), axis=1)
        self.assertGreater(float(np.mean(pred == y)), 0.45)

    def test_top3_hit(self) -> None:
        proba = np.array([0.05, 0.4, 0.3, 0.1, 0.05, 0.04, 0.04, 0.02])
        self.assertEqual(top_k_hit(proba, 1, 1), 1.0)
        self.assertEqual(top_k_hit(proba, 2, 3), 1.0)
        self.assertEqual(top_k_hit(proba, 7, 3), 0.0)

    def test_mlp_param_count(self) -> None:
        n_params = TwoLayerRecommender().n_parameters()
        self.assertEqual(n_params, 3 * 16 + 16 + 16 * 8 + 8)


class PackDataTests(unittest.TestCase):
    def test_aligned_50x3_not_15(self) -> None:
        x, y, sid_s, sid_c = align_features_and_labels()
        self.assertEqual(x.shape, (50, N_FEATURES))
        self.assertEqual(y.shape, (50,))
        self.assertTrue(set(y.tolist()).issubset(set(range(N_CLASSES))))
        self.assertEqual(sid_s[0], 1)
        self.assertEqual(sid_c[0], 0)

    def test_pipeline_writes_metrics_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_dual_recommender_pipeline(
                metrics_path=root / "model_comparison_metrics.csv",
                heatmap_path=root / "heat.png",
                curve_path=root / "curve.png",
            )
            self.assertEqual(result.x.shape[1], 3)
            self.assertTrue((root / "model_comparison_metrics.csv").is_file())
            self.assertTrue((root / "heat.png").is_file())
            self.assertGreater((root / "heat.png").stat().st_size, 1000)
            frame = pd.read_csv(root / "model_comparison_metrics.csv")
            self.assertGreaterEqual(len(frame), 2)
            self.assertIn("top1_accuracy", frame.columns)
            self.assertTrue(np.all((frame["top1_accuracy"].dropna() >= 0.0) & (frame["top1_accuracy"].dropna() <= 1.0)))


if __name__ == "__main__":
    unittest.main()

"""无泄漏数据集：标签、惯性置零、折内 StandardScaler。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.dataset import (
    FEATURE_COLS,
    INERTIA_COLS,
    build_inclusion_dataset,
    extract_inertia_table,
    get_loocv_folds,
)


class TestMlDataset(unittest.TestCase):
    """维度、NaN、候选惯性、折内 scaler。"""

    def test_shapes_and_no_nan(self) -> None:
        data = build_inclusion_dataset()
        self.assertEqual(data.X.shape[1], len(FEATURE_COLS))
        self.assertEqual(data.X_cand.shape[1], data.X.shape[1])
        self.assertEqual(data.X.shape[0], data.y.shape[0])
        self.assertFalse(np.isnan(data.X).any())
        self.assertFalse(np.isnan(data.X_cand).any())
        self.assertEqual(set(data.y.tolist()), {0, 1})
        self.assertIn("ATH", data.codes)
        self.assertIn("KTE", data.codes)
        self.assertNotIn("CKT", data.codes)

    def test_candidate_inertia_is_zero(self) -> None:
        data = build_inclusion_dataset()
        inertia = data.cand_frame.loc[:, list(INERTIA_COLS)].to_numpy(dtype=float)
        np.testing.assert_allclose(inertia, 0.0)

    def test_xlsx_baseball_inertia_finite(self) -> None:
        inertia = extract_inertia_table()
        baseball = inertia.loc[inertia["Code"].isin(["BSB", "SBL"])]
        self.assertEqual(baseball.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(baseball.loc[:, list(INERTIA_COLS)].to_numpy(dtype=float))))

    def test_scaler_matches_training_fold_mean(self) -> None:
        data = build_inclusion_dataset()
        n = 0
        for fold in get_loocv_folds(data.X, data.y):
            n += 1
            raw_train = data.X[fold.train_index]
            np.testing.assert_allclose(fold.scaler.mean_, np.mean(raw_train, axis=0), atol=1.0e-12)
            self.assertEqual(fold.X_val.shape, (1, data.X.shape[1]))
        self.assertEqual(n, data.y.size)


if __name__ == "__main__":
    unittest.main()

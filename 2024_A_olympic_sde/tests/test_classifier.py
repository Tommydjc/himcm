"""小样本入席分类：LOOCV、无 80/20、概率落在 [0,1]。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.classifier import (
    _new_logit,
    inclusion_label,
    labeled_training_frame,
    load_candidates,
    load_matrix,
    loocv_scores,
    predict_candidate_probability,
    fit_inclusion_model,
)


class TestInclusionClassifier(unittest.TestCase):
    """标签、留一法与候选接口。"""

    def test_backtest_labels_have_both_classes(self) -> None:
        matrix = load_matrix()
        _raw, y = labeled_training_frame(matrix)
        self.assertEqual(int(y.size), 6)
        self.assertEqual(int(np.sum(y)), 4)
        self.assertIn(0, set(y.tolist()))
        self.assertIn(1, set(y.tolist()))
        full_y = inclusion_label(matrix)
        ckt = int(full_y.loc[matrix["Code"].astype(str) == "CKT"].iloc[0])
        self.assertEqual(ckt, 0)

    def test_loocv_not_holdout_split(self) -> None:
        source = Path(__file__).resolve().parents[1] / "src" / "ml" / "classifier.py"
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("train_test_split", text)
        self.assertIn("get_loocv_folds", text)

    def test_2032_table_schema_and_ci_order(self) -> None:
        from src.ml.classifier import (
            bootstrap_candidate_probs,
            build_2032_table,
            run_loocv_dataset,
        )
        from src.ml.dataset import build_inclusion_dataset

        data = build_inclusion_dataset()
        report = run_loocv_dataset(data.X, data.y)
        self.assertEqual(report.confusion_logit.shape, (2, 2))
        self.assertTrue(0.0 <= report.acc_logit <= 1.0)
        self.assertTrue(0.0 <= report.brier_logit <= 1.0)
        logit_draws, rf_draws = bootstrap_candidate_probs(
            data.X, data.y, data.X_cand, n_boot=40, seed=42
        )
        table = build_2032_table(data.cand_names, logit_draws, rf_draws)
        self.assertEqual(
            list(table.columns),
            ["sde_name", "prob_logistic", "prob_rf", "ci_lower", "ci_upper", "ml_tier"],
        )
        self.assertEqual(table.shape[0], 3)
        self.assertTrue(np.all(table["ci_lower"].to_numpy() <= table["ci_upper"].to_numpy()))

    def test_loocv_metrics_in_range(self) -> None:
        raw, y = labeled_training_frame(load_matrix())
        auc, brier, oof = loocv_scores(raw, y, _new_logit)
        self.assertTrue(0.0 <= auc <= 1.0)
        self.assertTrue(0.0 <= brier <= 1.0)
        self.assertEqual(oof.shape, (6,))
        self.assertTrue(np.all(oof >= 0.0) and np.all(oof <= 1.0))

    def test_predict_candidate_probability_three_rows(self) -> None:
        raw, y = labeled_training_frame(load_matrix())
        model = fit_inclusion_model(raw, y, "l2_logistic", _new_logit)
        cand = load_candidates()
        out = predict_candidate_probability(model, cand)
        self.assertEqual(out.shape[0], 3)
        self.assertTrue(np.all(out["P_include"].to_numpy() >= 0.0))
        self.assertTrue(np.all(out["P_include"].to_numpy() <= 1.0))
        self.assertTrue(np.all(out["CI_low"].to_numpy() <= out["CI_high"].to_numpy()))


if __name__ == "__main__":
    unittest.main()

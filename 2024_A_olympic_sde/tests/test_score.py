"""规范化与 TOPSIS：效益/成本方向、加权和、贴近度。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indicators.schema import Indicator
from src.mcda.score import composite_scoring, normalize_indicators, topsis_scoring

SCHEMA: list[Indicator] = [
    Indicator(code="GLOBAL_REACH", sense="benefit"),
    Indicator(code="COST_INFRA", sense="cost"),
]


class TestScoringPipeline(unittest.TestCase):
    """确定性三方案表：不写 ``results/``。"""

    def setUp(self) -> None:
        self.raw = pd.DataFrame(
            {
                "Discipline": ["A", "B", "C"],
                "GLOBAL_REACH": [1.0, 3.0, 5.0],
                "COST_INFRA": [10.0, 6.0, 2.0],
            }
        )

    def test_normalize_benefit_and_cost_in_unit_interval(self) -> None:
        norm = normalize_indicators(self.raw, SCHEMA)
        g = norm["GLOBAL_REACH"].to_numpy()
        c = norm["COST_INFRA"].to_numpy()
        self.assertTrue(np.all(g >= 0.0) and np.all(g <= 1.0))
        self.assertTrue(np.all(c >= 0.0) and np.all(c <= 1.0))
        self.assertGreater(g[2], g[0])
        self.assertGreater(c[2], c[0])
        self.assertAlmostEqual(float(g[0]), 0.0, places=8)
        self.assertAlmostEqual(float(c[2]), 1.0, places=5)

    def test_composite_rank_desc(self) -> None:
        norm = normalize_indicators(self.raw, SCHEMA)
        weights = np.array([0.5, 0.5], dtype=np.float64)
        scored = composite_scoring(norm, weights, schema=SCHEMA)
        self.assertEqual(list(scored["Rank"]), [1, 2, 3])
        self.assertEqual(scored.loc[0, "Discipline"], "C")
        np.testing.assert_allclose(float(np.sum(weights)), 1.0)

    def test_topsis_closeness_in_unit_interval(self) -> None:
        norm = normalize_indicators(self.raw, SCHEMA)
        weights = np.array([0.5, 0.5], dtype=np.float64)
        ranked = topsis_scoring(norm, weights, schema=SCHEMA)
        scores = ranked["TopsisScore"].to_numpy()
        self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0))
        self.assertEqual(int(ranked.loc[0, "Rank"]), 1)
        self.assertEqual(ranked.loc[0, "Discipline"], "C")


if __name__ == "__main__":
    unittest.main()

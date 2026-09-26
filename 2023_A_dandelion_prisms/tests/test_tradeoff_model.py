"""PRISMS 生物经济：修剪算子、窗口日、非支配集。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.biology.phenology import PhenologyParams, empty_stages
from src.decision.tradeoff_model import (
    StrategyScore,
    apply_chemical,
    apply_mechanical,
    first_true_bloom_day,
    make_weekly_mow_policy,
    nondominated_benefit_cost,
    prisms_mow_day,
)
from src.simulation.grid_engine import DailyForcing as EngineForcing


class TestTradeoffOperators(unittest.TestCase):
    """机械/化学不增密度；窗口在花期后 11 日。"""

    def test_mechanical_cuts_adults(self) -> None:
        params = PhenologyParams()
        state = empty_stages(4, params)
        state.adult[:, :] = 10.0
        state.rosette[:, :] = 8.0
        nxt = apply_mechanical(state, 0.5)
        self.assertAlmostEqual(float(nxt.adult[0, 0]), 5.0)
        self.assertLess(float(nxt.rosette[0, 0]), 8.0)
        self.assertEqual(float(nxt.seed[0, 0]), 0.0)

    def test_chemical_hits_seedlings(self) -> None:
        params = PhenologyParams()
        state = empty_stages(3, params)
        state.seedling[0, :, :] = 4.0
        nxt = apply_chemical(state, 1.0)
        self.assertLess(float(np.max(nxt.seedling)), 4.0)

    def test_prisms_window(self) -> None:
        rows = [
            EngineForcing(0, 10.0, 3.0, 270.0, 0.2, True),
            EngineForcing(90, 12.0, 3.0, 270.0, 0.2, True),
        ]
        self.assertEqual(first_true_bloom_day(rows), 90)
        self.assertEqual(prisms_mow_day(rows), 101)

    def test_weekly_ledger_counts(self) -> None:
        ledger, intervene = make_weekly_mow_policy(0.8)
        params = PhenologyParams()
        state = empty_stages(2, params)
        for day in (0, 7, 14, 15):
            forcing = EngineForcing(day, 15.0, 3.0, 270.0, 0.2, False)
            state = intervene(forcing, state)
        self.assertEqual(ledger.n_mows, 2)

    def test_pareto_max_b_min_c(self) -> None:
        a = StrategyScore("a", "a", 0, 0, 2.0, 0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 0)
        b = StrategyScore("b", "b", 0, 0, 1.0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0)
        c = StrategyScore("c", "c", 0, 0, 0.5, 0, 0, 0, 2.0, -1.5, 0, 0, 0, 0, 0)
        front = {item.name for item in nondominated_benefit_cost([a, b, c])}
        self.assertIn("a", front)
        self.assertIn("b", front)
        self.assertNotIn("c", front)


if __name__ == "__main__":
    unittest.main()

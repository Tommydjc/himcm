"""Stage-2 机理验收：对齐 ``phenology`` / ``grid_engine`` 的函数式接口。

不引入 pytest，不假设 ``DandelionLifecycle`` / ``DandelionGridEngine`` 类。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.biology.phenology import (
    PhenologyParams,
    gamma_environment,
    initial_center_adult,
    step_phenology,
)
from src.simulation.grid_engine import DailyForcing, run_hectare_days


class TestStage2Mechanisms(unittest.TestCase):
    """赛题初值、卷积尺度、K_cell 硬帽、低温休眠。"""

    def setUp(self) -> None:
        self.params = PhenologyParams()
        self.state = initial_center_adult(100, self.params, 50, 50)

    def test_initial_state(self) -> None:
        """t=0：全场仅 (50, 50) 一株开花成株。"""
        adult = self.state.adult
        self.assertEqual(adult.shape, (100, 100))
        self.assertAlmostEqual(float(np.sum(adult)), 1.0)
        self.assertEqual(adult[50, 50], 1.0)
        self.assertEqual(float(np.sum(self.state.seed)), 0.0)

    def test_grid_dimension_preservation(self) -> None:
        """一日卷积 + 物候后网格仍为 100×100。"""
        forcing = DailyForcing(
            day=0,
            temp_c=20.0,
            wind_mps=3.0,
            wind_from_deg=270.0,
            moisture=0.35,
            releasing=True,
        )
        result = run_hectare_days([forcing], params=self.params)
        self.assertEqual(result.state.adult.shape, (100, 100))
        self.assertEqual(result.state.seed.shape, (100, 100))
        self.assertEqual(result.state.rosette.shape, (100, 100))
        self.assertEqual(result.state.seedling.shape, (self.params.seedling_days, 100, 100))

    def test_carrying_capacity_constraint(self) -> None:
        """中心灌注 5e4 粒种子后推进 30 日，单格成株不超过 K_cell=15。"""
        state = self.state
        state.seed[50, 50] = 50_000.0
        for _ in range(30):
            state = step_phenology(state, 22.0, 0.40, self.params)
        self.assertLessEqual(float(np.max(state.adult)), self.params.K_cell)

    def test_temperature_dormancy(self) -> None:
        """−5 °C 时 γ 与日萌发率接近 0（VWC=0.20，避开渍水截断）。"""
        gamma = gamma_environment(-5.0, 0.20, self.params)
        germ_rate = self.params.g_seed * gamma
        self.assertLess(germ_rate, 0.05)
        self.assertLess(gamma, 0.05)


if __name__ == "__main__":
    unittest.main()

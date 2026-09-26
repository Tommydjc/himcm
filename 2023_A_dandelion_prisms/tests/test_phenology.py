"""日步物候：中央初值、25 日建成、成株硬帽、密度依赖幼苗死亡。"""

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
    apply_adult_cap,
    empty_stages,
    gamma_environment,
    initial_center_adult,
    seedling_survival,
    step_phenology,
)


class TestPhenology(unittest.TestCase):
    """阶段机不变量。"""

    def test_center_one_adult(self) -> None:
        params = PhenologyParams()
        state = initial_center_adult(100, params, 50, 50)
        self.assertEqual(state.adult[50, 50], 1.0)
        self.assertAlmostEqual(float(np.sum(state.adult)), 1.0)
        self.assertEqual(float(np.sum(state.seed)), 0.0)
        self.assertEqual(state.seedling.shape, (25, 100, 100))

    def test_gamma_waterlog_zero(self) -> None:
        self.assertEqual(gamma_environment(20.0, 0.50, PhenologyParams()), 0.0)

    def test_adult_hard_cap(self) -> None:
        params = PhenologyParams()
        capped = apply_adult_cap(np.full((4, 4), 20.0), params)
        np.testing.assert_allclose(capped, 15.0)

    def test_seedling_dies_at_capacity(self) -> None:
        params = PhenologyParams()
        adult = np.full((3, 3), params.K_cell)
        surv = seedling_survival(adult, params)
        np.testing.assert_allclose(surv, 0.0, atol=1.0e-12)

    def test_twenty_five_days_to_rosette(self) -> None:
        params = PhenologyParams()
        state = empty_stages(8, params)
        state.seed[4, 4] = 100.0
        for _ in range(26):
            state = step_phenology(state, 18.0, 0.20, params)
        self.assertGreater(float(state.rosette[4, 4]), 0.0)
        self.assertLessEqual(float(np.max(state.adult)), params.K_cell)


if __name__ == "__main__":
    unittest.main()

"""日步公顷引擎：吸收边界、卷积守恒差、365 日时限。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.biology.phenology import PhenologyParams, empty_stages
from src.physics.dispersal_kernel import build_hybrid_kernel
from src.simulation.grid_engine import (
    DailyForcing,
    disperse_seeds,
    load_daily_forcing,
    run_hectare_days,
)


class TestAbsorbingConvolution(unittest.TestCase):
    """``convolve2d(..., mode='same')`` 边界质量丢失。"""

    def test_corner_release_loses_mass(self) -> None:
        params = PhenologyParams()
        state = empty_stages(100, params)
        state.adult[0, 0] = 1.0
        kernel = build_hybrid_kernel(4.0, 225.0, dx_m=1.0, half_m=40)
        rain, lost = disperse_seeds(state.adult, kernel, params, releasing=True)
        released = 180.0
        np.testing.assert_allclose(float(np.sum(rain)) + lost, released, atol=1.0e-8)
        self.assertGreater(lost, 0.0)
        self.assertEqual(rain.shape, (100, 100))

    def test_center_release_mostly_inside(self) -> None:
        params = PhenologyParams()
        state = empty_stages(100, params)
        state.adult[50, 50] = 1.0
        kernel = build_hybrid_kernel(2.0, 0.0, dx_m=1.0, half_m=20)
        rain, lost = disperse_seeds(state.adult, kernel, params, releasing=True)
        self.assertGreater(float(np.sum(rain)), 0.5 * 180.0)
        np.testing.assert_allclose(float(np.sum(rain)) + lost, 180.0, atol=1.0e-8)


class TestHectareYear(unittest.TestCase):
    """365 日从 (50,50) 出发，限时 15 s，成株不超过 15/格。"""

    def test_365_days_under_15s(self) -> None:
        params = PhenologyParams()
        forcing = [
            DailyForcing(
                day=i,
                temp_c=18.0,
                wind_mps=3.0,
                wind_from_deg=270.0,
                moisture=0.20,
                releasing=(i % 30 == 0),
            )
            for i in range(365)
        ]
        result = run_hectare_days(forcing, params=params)
        self.assertLess(result.elapsed_s, 15.0)
        self.assertEqual(result.state.adult.shape, (100, 100))
        self.assertLessEqual(float(np.max(result.state.adult)), params.K_cell + 1.0e-9)
        self.assertGreater(float(np.sum(result.state.adult)), 0.0)

    def test_open_meteo_loader_length(self) -> None:
        rows = load_daily_forcing("temperate", n_days=365)
        self.assertEqual(len(rows), 365)
        self.assertTrue(rows[0].releasing)


if __name__ == "__main__":
    unittest.main()

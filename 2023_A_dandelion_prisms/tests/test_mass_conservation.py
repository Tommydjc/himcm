"""点源投放：格内 + 格外 = 释放量（连续密度采样账）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.params import load_biophysics
from src.physics.plume import deposit_from_point
from src.simulation.grid import cell_centers


class TestMassConservation(unittest.TestCase):
    """源在域外时，窗外质量非负且总和等于释放量。"""

    def test_point_source_budget(self) -> None:
        params = load_biophysics()
        x_c, y_c = cell_centers(params)
        n_seeds = 180.0
        field, outside = deposit_from_point(
            n_seeds,
            params.source_x_m,
            params.source_y_m,
            x_c,
            y_c,
            params,
            wind_mps=4.0,
            wind_from_deg=270.0,
        )
        self.assertGreaterEqual(outside, -1.0e-9)
        self.assertGreaterEqual(float(np.sum(field)), 0.0)
        total = float(np.sum(field)) + outside
        np.testing.assert_allclose(total, n_seeds, atol=1.0e-6)

    def test_center_source_mostly_inside(self) -> None:
        params = load_biophysics()
        x_c, y_c = cell_centers(params)
        field, outside = deposit_from_point(
            1000.0,
            50.0,
            50.0,
            x_c,
            y_c,
            params,
            wind_mps=2.0,
            wind_from_deg=0.0,
        )
        self.assertGreater(float(np.sum(field)), 0.5 * 1000.0)
        np.testing.assert_allclose(float(np.sum(field)) + outside, 1000.0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()

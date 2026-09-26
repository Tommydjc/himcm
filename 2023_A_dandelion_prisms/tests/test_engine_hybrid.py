"""引擎已改接混合核：卷积核和为 1，Phi 只认 VWC。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.params import load_biophysics
from src.physics.dispersal_kernel import build_hybrid_kernel, calculate_germination_modifier
from src.simulation.engine import WeekForcing, empty_grid, step_week
from src.simulation.grid import GridState


def _bloom_week(soil_moisture: float | None) -> WeekForcing:
    return WeekForcing(
        week=16,
        month=4,
        temp_c=18.0,
        wind_mps=3.5,
        wind_from_deg=270.0,
        smi=0.55,
        blooming=True,
        mow_eta=0.0,
        soil_moisture=soil_moisture,
    )


class TestEngineHybridKernel(unittest.TestCase):
    """``step_week`` 走 ``build_hybrid_kernel``。"""

    def test_hybrid_kernel_sums_to_one(self) -> None:
        kernel = build_hybrid_kernel(3.5, 270.0, dx_m=1.0, half_m=40)
        np.testing.assert_allclose(np.sum(kernel), 1.0, atol=1.0e-12)

    def test_phi_one_without_vwc(self) -> None:
        params = load_biophysics()
        state = empty_grid(params)
        nxt, ledger = step_week(state, _bloom_week(None), params, inject_source=True)
        self.assertEqual(ledger["phi"], 1.0)
        self.assertGreater(ledger["source_released"], 0.0)
        self.assertIsInstance(nxt, GridState)

    def test_phi_matches_modifier_with_vwc(self) -> None:
        params = load_biophysics()
        theta = 0.20
        expected = float(calculate_germination_modifier(18.0, theta))
        _nxt, ledger = step_week(empty_grid(params), _bloom_week(theta), params, inject_source=True)
        self.assertAlmostEqual(ledger["phi"], expected, places=12)
        self.assertLess(ledger["phi"], 1.0)

    def test_smi_not_used_as_theta(self) -> None:
        params = load_biophysics()
        _nxt, ledger = step_week(empty_grid(params), _bloom_week(None), params, inject_source=True)
        self.assertEqual(ledger["phi"], 1.0)


if __name__ == "__main__":
    unittest.main()

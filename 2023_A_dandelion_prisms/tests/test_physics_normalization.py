"""羽流核空间积分归一化。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.params import load_biophysics
from src.physics.plume import build_plume_kernel
from src.physics.wald import wald_mean_shape, wald_pdf


class TestPhysicsNormalization(unittest.TestCase):
    """离散核和为 1；WALD 在正轴上可积到约 1。"""

    def test_plume_kernel_sums_to_one(self) -> None:
        params = load_biophysics()
        for wind_from in (0.0, 90.0, 210.0, 315.0):
            kernel = build_plume_kernel(params, wind_mps=3.5, wind_from_deg=wind_from)
            self.assertEqual(kernel.shape, (2 * params.kernel_half_m + 1, 2 * params.kernel_half_m + 1))
            np.testing.assert_allclose(np.sum(kernel), 1.0, atol=1.0e-12)
            self.assertTrue(np.all(kernel >= -1.0e-15))

    def test_wald_pdf_integrates_near_one(self) -> None:
        mu, lam = wald_mean_shape(0.35, 3.5, 0.32, 0.40)
        r = np.linspace(1.0e-4, 80.0, 20000)
        mass = float(np.trapezoid(wald_pdf(r, mu, lam), r))
        self.assertGreater(mass, 0.95)
        self.assertLess(mass, 1.02)


if __name__ == "__main__":
    unittest.main()

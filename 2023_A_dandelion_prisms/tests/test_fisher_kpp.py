"""Fisher–KPP：解析波速与 1D 数值波前同量级。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.biology.phenology import PhenologyParams
from src.theory.fisher_kpp import (
    TAU_DAY,
    diffusion_from_hybrid_kernel,
    diffusion_from_second_moment,
    diffusion_from_wind_sigma,
    intrinsic_rate_daily,
    regress_front_speed,
    solve_fisher_1d,
    traveling_wave_speed,
)


class TestFisherKPP(unittest.TestCase):
    """\(c^*=2\\sqrt{rD}\) 与数值波速相对误差有界。"""

    def test_analytic_speed_formula(self) -> None:
        self.assertAlmostEqual(traveling_wave_speed(1.0, 1.0), 2.0, places=12)

    def test_numerical_front_near_theory(self) -> None:
        r = 0.25
        D = 4.0
        c_star = traveling_wave_speed(r, D)
        _x, _u, c_num = solve_fisher_1d(D, r, K=1.0, length_m=200.0, t_end=30.0, nx=301, n_steps=3000)
        self.assertGreater(c_num, 0.0)
        rel = abs(c_num - c_star) / c_star
        self.assertLess(rel, 0.35)

    def test_wind_sigma_matches_moment_at_unit_tau(self) -> None:
        diff_d, moment, sigma_sq = diffusion_from_hybrid_kernel(3.5, 270.0, tau=TAU_DAY)
        self.assertAlmostEqual(sigma_sq, 0.5 * moment, places=12)
        self.assertAlmostEqual(diff_d, diffusion_from_second_moment(moment, TAU_DAY), places=12)
        self.assertAlmostEqual(diff_d, diffusion_from_wind_sigma(sigma_sq, TAU_DAY), places=12)

    def test_daily_intrinsic_rate_positive_in_bloom(self) -> None:
        r_day = intrinsic_rate_daily(PhenologyParams(), gamma=0.4, release_prob=0.4)
        self.assertGreater(r_day, 0.0)

    def test_regress_front_uses_late_window(self) -> None:
        times = np.arange(1.0, 61.0)
        radii = 0.4 * times
        c_num, i0, i1 = regress_front_speed(times, radii, r_cap_m=42.0, n_late=20)
        self.assertGreater(i1, i0)
        self.assertAlmostEqual(c_num, 0.4, places=6)


if __name__ == "__main__":
    unittest.main()

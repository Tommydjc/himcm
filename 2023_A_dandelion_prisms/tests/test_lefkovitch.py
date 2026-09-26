"""Lefkovitch：非负、无产籽时总量不增、好气候谱半径 > 1。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.biology.lefkovitch import build_rates, climate_multiplier, lefkovitch_step, spectral_radius
from src.params import load_biophysics


class TestLefkovitch(unittest.TestCase):
    """阶段机不变量。"""

    def test_cold_multiplier_zero(self) -> None:
        params = load_biophysics()
        self.assertEqual(climate_multiplier(-2.0, 0.6, params), 0.0)

    def test_no_fecundity_population_does_not_increase(self) -> None:
        params = load_biophysics()
        rates = build_rates(18.0, 0.55, params)
        seed = np.array([[10.0]])
        seedling = np.array([[4.0]])
        rosette = np.array([[3.0]])
        adult = np.array([[2.0]])
        before = float(np.sum(seed + seedling + rosette + adult))
        after_fields = lefkovitch_step(seed, seedling, rosette, adult, rates)
        after = float(sum(float(np.sum(item)) for item in after_fields))
        self.assertLessEqual(after, before + 1.0e-12)
        for item in after_fields:
            self.assertTrue(np.all(item >= -1.0e-15))

    def test_bloom_spectral_radius_exceeds_one(self) -> None:
        params = load_biophysics()
        rates = build_rates(18.0, 0.55, params)
        fec = params.n_seed_per_capitulum * params.n_capitula_per_adult_week_bloom * climate_multiplier(
            18.0, 0.55, params
        )
        rho = spectral_radius(rates, fec)
        self.assertGreater(rho, 1.0)


if __name__ == "__main__":
    unittest.main()

"""自疏上限：莲座+成株不超过 \(K_{\\mathrm{eff}}\)。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.biology.self_thinning import apply_self_thinning, carrying_capacity
from src.params import load_biophysics
from src.scripts.generate_forcing import write_all_climates
from src.scripts.forcing_weeks import load_weekly_forcing
from src.params import load_climates
from src.simulation.engine import run_year


class TestSelfThinning(unittest.TestCase):
    """硬盖帽与全年仿真最大值。"""

    def test_hard_cap_on_synthetic_field(self) -> None:
        params = load_biophysics()
        smi = 0.50
        k_eff = carrying_capacity(smi, params)
        rosette = np.full((8, 8), 0.7 * k_eff + 20.0)
        adult = np.full((8, 8), 0.7 * k_eff + 15.0)
        r2, a2 = apply_self_thinning(rosette, adult, smi, params)
        np.testing.assert_allclose(r2 + a2, k_eff, atol=1.0e-10)
        self.assertTrue(np.all(r2 >= 0.0))
        self.assertTrue(np.all(a2 >= 0.0))

    def test_year_run_respects_cap(self) -> None:
        params = load_biophysics()
        write_all_climates()
        spec = load_climates()["tropical"]
        forcing = load_weekly_forcing("tropical", spec)
        _state, weekly, _snaps = run_year(forcing, params)
        k_eff = carrying_capacity(float(np.mean([w.smi for w in forcing])), params)
        for row in weekly:
            self.assertLessEqual(row["max_canopy"], params.K_max_per_m2 + 1.0e-9)
            self.assertLessEqual(row["max_canopy"], k_eff + params.K_max_per_m2)


if __name__ == "__main__":
    unittest.main()

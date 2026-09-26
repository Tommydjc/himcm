"""权重扰动归一化与 2032 候选三名次标签。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scripts.run_sensitivity import (
    IOC_CRITERIA,
    PLACE_BY_RANK,
    load_candidates,
    load_locked_ahp_weights,
    perturb_group_weights,
    run_oat_on_candidates,
    zero_youth_weights,
)


class TestWeightShock(unittest.TestCase):
    """扰动后权和为 1，青年归零后该叶子为 0。"""

    def test_perturb_renormalizes(self) -> None:
        weights, _cr, ok = load_locked_ahp_weights()
        self.assertTrue(ok)
        leaves = IOC_CRITERIA[0][1]
        plus = perturb_group_weights(weights, leaves, 1.2)
        minus = perturb_group_weights(weights, leaves, 0.8)
        np.testing.assert_allclose(float(np.sum(plus)), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(float(np.sum(minus)), 1.0, atol=1.0e-12)

    def test_zero_youth_mass(self) -> None:
        weights, _cr, _ok = load_locked_ahp_weights()
        zed = zero_youth_weights(weights)
        from src.indicators.schema import INDICATORS

        youth_idx = [i for i, item in enumerate(INDICATORS) if item.code == "YOUTH_APPEAL"][0]
        self.assertAlmostEqual(float(zed[youth_idx]), 0.0, places=12)
        np.testing.assert_allclose(float(np.sum(zed)), 1.0, atol=1.0e-12)

    def test_brisbane_places_are_medals(self) -> None:
        weights, _cr, _ok = load_locked_ahp_weights()
        ranking, shock = run_oat_on_candidates(load_candidates(), weights)
        self.assertEqual(list(ranking["Place"]), ["1st", "2nd", "3rd"])
        self.assertEqual(set(ranking["Place"]), set(PLACE_BY_RANK.values()))
        self.assertIn("youth_weight_zero", set(shock["Scenario"]))
        self.assertEqual(int(shock.loc[shock["Universe"] == "brisbane_candidates", "Criterion"].nunique()), 7)


if __name__ == "__main__":
    unittest.main()

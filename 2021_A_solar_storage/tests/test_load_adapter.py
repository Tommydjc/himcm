"""AdaptiveLoadAdapter energy conservation and blizzard A/B CSV."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent_sim.load_adapter import HVAC_CUT_FRAC_PER_C, RIGID_FLOOR_KW, AdaptiveLoadAdapter
from src.agent_sim.models import LoadSheddingAgreement
from src.agent_sim.run_simulation import run_blizzard_ab, storm_slice_hours


class LoadAdapterTests(unittest.TestCase):
    """Catalog watts × duty, HVAC 8%/C, rigid floor, no invented kWh."""

    def setUp(self) -> None:
        self.adapter = AdaptiveLoadAdapter()
        self.deal = LoadSheddingAgreement(
            consensus_reached=True,
            shed_appliances=["Electric_Clothes_Dryer", "Dishwasher"],
            hvac_temp_offset_c=2.5,
            negotiated_reduction_kW=2.1,
            rationale="test",
        )

    def test_dryer_peak_cut(self) -> None:
        hour = np.array([13])
        raw = np.array([6.0])
        out = self.adapter.apply_curtailment(raw, self.deal, hour)
        dryer = 2.4 * 1.0
        hvac = 2.2 * 0.5 * (2.5 * HVAC_CUT_FRAC_PER_C)
        expected = 6.0 - dryer - hvac
        self.assertAlmostEqual(float(out[0]), expected, places=5)

    def test_floor_and_no_increase(self) -> None:
        raw = np.array([0.35, 6.0])
        hour = np.array([21, 21])
        out = self.adapter.apply_curtailment(raw, self.deal, hour)
        self.assertGreaterEqual(float(out[0]), RIGID_FLOOR_KW - 1.0e-12)
        self.assertTrue(np.all(out <= raw + 1.0e-12))
        self.assertTrue(np.all(out >= 0.0))

    def test_raw_below_floor_not_raised(self) -> None:
        raw = np.array([0.12])
        hour = np.array([3])
        out = self.adapter.apply_curtailment(raw, self.deal, hour)
        self.assertLessEqual(float(out[0]), 0.12 + 1.0e-12)

    def test_protected_ignored(self) -> None:
        dirty = LoadSheddingAgreement(
            consensus_reached=True,
            shed_appliances=["WiFi_Router_Comm", "Electric_Clothes_Dryer"],
            hvac_temp_offset_c=0.0,
            negotiated_reduction_kW=1.0,
            rationale="x",
        )
        hour = np.array([13])
        raw = np.array([5.0])
        out = self.adapter.apply_curtailment(raw, dirty, hour)
        self.assertAlmostEqual(float(out[0]), 5.0 - 2.4 * 1.0, places=5)


class BlizzardAbTests(unittest.TestCase):
    """Days 40–42 window and CSV contract."""

    def test_storm_length(self) -> None:
        t0, t1 = storm_slice_hours()
        self.assertEqual(t1 - t0 + 1, 72)
        self.assertEqual(t0, 39 * 24)

    def test_run_writes_72_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crewai_adaptive_vs_passive.csv"
            frame = run_blizzard_ab(out_csv=path)
            self.assertTrue(path.is_file())
            self.assertEqual(len(frame), 72)
            for col in (
                "timestamp",
                "raw_load_kW",
                "negotiated_load_kW",
                "soc_passive",
                "soc_adaptive",
            ):
                self.assertIn(col, frame.columns)
            self.assertTrue(
                np.all(
                    frame["negotiated_load_kW"].to_numpy()
                    <= frame["raw_load_kW"].to_numpy() + 1.0e-9
                )
            )


if __name__ == "__main__":
    unittest.main()

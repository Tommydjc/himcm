"""20 英亩扁桃园蜂箱配比护栏。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pollination.orchard_20acres import (
    GOLDEN_K_HI,
    GOLDEN_K_LO,
    OrchardParams,
    daily_food_per_hive,
    recommend_stocking,
    run_orchard_optimization,
    sweep_stocking,
    yield_ratio,
)


class Orchard20AcresTests(unittest.TestCase):
    """竞争稀释、结实饱和、黄金带推荐。"""

    def test_competition_dilution(self) -> None:
        c_max = 1000.0
        orchard = 20000.0
        self.assertAlmostEqual(daily_food_per_hive(10, orchard, c_max), c_max)
        self.assertAlmostEqual(daily_food_per_hive(40, orchard, c_max), 500.0)

    def test_michaelis_menten(self) -> None:
        self.assertAlmostEqual(yield_ratio(0.0, 2.5), 0.0)
        self.assertAlmostEqual(yield_ratio(2.5, 2.5), 0.5)
        self.assertGreater(yield_ratio(10.0, 2.5), 0.75)

    def test_sweep_and_golden(self) -> None:
        frame, cal = sweep_stocking(OrchardParams(), k_min=5, k_max=60)
        self.assertEqual(len(frame), 56)
        self.assertGreater(cal["max_foraging_capacity_g_day"], 0.0)
        rec = recommend_stocking(frame)
        self.assertGreaterEqual(rec["K_star_golden"], GOLDEN_K_LO)
        self.assertLessEqual(rec["K_star_golden"], GOLDEN_K_HI)

    def test_pipeline_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = run_orchard_optimization(
                csv_path=root / "orchard_pollination_optimization.csv",
                fig_path=root / "fig_20acre_hive_density_tradeoff.png",
                recommendation_path=root / "orchard_pollination_recommendation.csv",
            )
            self.assertTrue((root / "orchard_pollination_optimization.csv").is_file())
            self.assertTrue((root / "fig_20acre_hive_density_tradeoff.png").is_file())
            loaded = pd.read_csv(root / "orchard_pollination_optimization.csv")
            self.assertEqual(len(loaded), len(frame))
            self.assertTrue(frame.loc[frame["K"] == 60, "food_ratio"].iloc[0] < 1.0)


if __name__ == "__main__":
    unittest.main()

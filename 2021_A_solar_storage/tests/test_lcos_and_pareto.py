"""LCOS closed-form checks and Pareto CSV contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.techno_economic.lcos_and_pareto import (
    compute_lcos,
    count_outage_events,
    plot_budget_outage_pareto,
    plot_lcos_radar,
    scan_budget_pareto,
    build_lcos_table,
)


class LcosAndParetoTests(unittest.TestCase):
    """Formula, event counting, and file exports."""

    def test_lcos_undiscounted_one_year(self) -> None:
        out = compute_lcos(
            capex_usd=100.0,
            e_discharged_y1_kwh=50.0,
            n_years=1,
            discount_rate=0.0,
            opex_usd_year=10.0,
            replacement_usd=0.0,
            lifetime_years=15.0,
        )
        self.assertAlmostEqual(out["npv_cost_usd"], 110.0, places=6)
        self.assertAlmostEqual(out["npv_energy_kwh"], 50.0, places=6)
        self.assertAlmostEqual(out["lcos_usd_per_kwh"], 2.2, places=6)

    def test_outage_event_runs(self) -> None:
        def_kw = np.array([0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 2.0])
        self.assertEqual(count_outage_events(def_kw), 3)

    def test_pareto_two_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "budget_pareto_front.csv"
            frame = scan_budget_pareto(
                budgets=[4000.0, 10000.0],
                out_csv=path,
                evaluate_8760=True,
                time_limit_s=20,
            )
            self.assertTrue(path.is_file())
            self.assertEqual(len(frame), 2)
            self.assertIn("outage_hours_8760", frame.columns)
            self.assertIn("bom", frame.columns)
            self.assertLessEqual(float(frame["capex_usd"].iloc[0]), 4000.0 + 1.0e-6)
            self.assertLessEqual(float(frame["capex_usd"].iloc[1]), 10000.0 + 1.0e-6)
            self.assertGreaterEqual(
                float(frame["outage_hours_8760"].iloc[0]),
                float(frame["outage_hours_8760"].iloc[1]) - 1.0,
            )

    def test_lcos_table_four_techs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lcos_emerging_tech.csv"
            frame = build_lcos_table(out_csv=path)
            self.assertEqual(len(frame), 4)
            self.assertTrue(np.all(np.isfinite(frame["lcos_usd_per_kwh"])))
            self.assertTrue((frame["lcos_usd_per_kwh"] > 0.0).all())
            self.assertSetEqual(set(frame["trl"]), {9, 7, 8, 4})

    def test_figures_write_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pareto = pd.DataFrame(
                {
                    "budget_usd": [4000.0, 8000.0, 10000.0, 18000.0],
                    "outage_hours_8760": [5000.0, 3500.0, 3100.0, 2800.0],
                    "lpsp_8760": [0.45, 0.34, 0.32, 0.28],
                }
            )
            lcos = pd.DataFrame(
                {
                    "name": ["LFP", "Na-ion", "VRFB", "Cement"],
                    "lcos_usd_per_kwh": [0.4, 0.2, 0.35, 2.0],
                    "energy_density_wh_kg": [140.0, 120.0, 20.0, 0.8],
                    "cycle_life": [4000, 4000, 15000, 8000],
                    "env_score": [7.0, 8.5, 8.0, 9.0],
                    "roundtrip_eff": [0.90, 0.90, 0.75, 0.60],
                }
            )
            p1 = plot_budget_outage_pareto(pareto, root / "fig_budget_outage_pareto.png")
            p3 = plot_lcos_radar(lcos, root / "fig_emerging_tech_lcos_radar.png")
            self.assertTrue(p1.is_file())
            self.assertGreater(p1.stat().st_size, 1000)
            self.assertTrue(p3.is_file())


if __name__ == "__main__":
    unittest.main()

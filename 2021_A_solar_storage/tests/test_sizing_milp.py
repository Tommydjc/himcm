"""MILP sizing / balance / budget guards."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.sizing_milp import (
    MilpCostParams,
    build_and_solve_milp,
    milp_battery_catalog,
    solve_optimal_storage,
)


class SizingMilpTests(unittest.TestCase):
    """Integer units, power balance, budget, LPSP."""

    def test_catalog_four_packs(self) -> None:
        packs = milp_battery_catalog()
        self.assertEqual(len(packs), 4)
        names = {p.model_name for p in packs}
        self.assertTrue(any("Powerwall" in n for n in names))
        self.assertIn("Enphase_IQ_10T", names)

    def test_toy_horizon_balance_and_budget(self) -> None:
        packs = milp_battery_catalog()
        n = 48
        hour = np.arange(n) % 24
        pv = np.where((hour >= 8) & (hour <= 16), 4.0, 0.0).astype(np.float64)
        load = np.full(n, 1.2, dtype=np.float64)
        w = np.ones(n, dtype=np.float64) * (8760.0 / n)
        budget = 2200.0 * 4 + 50.0
        milp = build_and_solve_milp(
            pv,
            load,
            w,
            packs,
            budget_limit=budget,
            max_allowed_outages=None,
            costs=MilpCostParams(epsilon_lpsp=0.20),
            wrap="daily",
            time_limit_s=30,
            name="toy_milp",
        )
        self.assertLess(float(milp["balance_max_abs_kw"]), 1.0e-4)
        self.assertLessEqual(float(milp["capex_usd"]), budget + 1.0e-6)
        self.assertLessEqual(float(milp["lpsp"]), 0.20 + 1.0e-6)
        x_hat = milp["x"]
        assert isinstance(x_hat, dict)
        for n_u in x_hat.values():
            self.assertIsInstance(n_u, int)
            self.assertGreaterEqual(n_u, 0)
            self.assertLessEqual(n_u, 4)
        self.assertGreater(sum(x_hat.values()), 0)

    def test_solve_writes_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sol = solve_optimal_storage(
                budget_limit=10000.0,
                max_allowed_outages=None,
                horizon="typical",
                sizing_csv=root / "optimal_sizing_solution.csv",
                dispatch_csv=root / "annual_power_dispatch.csv",
                evaluate_8760=True,
                time_limit_s=60,
            )
            self.assertTrue((root / "optimal_sizing_solution.csv").is_file())
            bom = pd.read_csv(root / "optimal_sizing_solution.csv")
            self.assertEqual(len(bom), 4)
            self.assertLessEqual(float(sol["capex_usd"]), 10000.0 + 1.0e-6)
            self.assertIn("lpsp_8760", sol)
            self.assertTrue((root / "annual_power_dispatch.csv").is_file())
            disp = pd.read_csv(root / "annual_power_dispatch.csv")
            self.assertEqual(len(disp), 8760)
            bal = (
                disp["pv_output_kW"]
                - disp["p_curt_kW"]
                + disp["p_dis_kW"]
                - disp["p_ch_kW"]
                + disp["p_def_kW"]
                - disp["load_demand_kW"]
            )
            self.assertLess(float(np.max(np.abs(bal))), 1.0e-4)


if __name__ == "__main__":
    unittest.main()

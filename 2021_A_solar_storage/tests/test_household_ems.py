"""Household EMS agents: SOC line, protected loads, DSR energy conservation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.agents.household_ems import (
    FamilyAction,
    PROTECTED,
    SOC_WARN,
    apply_family_action,
    butler_alert,
    classify_alert,
    find_blizzard_window,
    negotiate,
    run_blizzard_agents,
)
from src.load_profiler.mcmc_load import (
    DEFAULT_CATALOG,
    LoadSynthParams,
    _hvac_layer,
    _load_catalog,
)
from src.optimization.sizing_milp import milp_battery_catalog
from src.battery_model.battery_dynamics import CommercialBattery


class HouseholdEmsTests(unittest.TestCase):
    """Level-3 blizzard protocol and physical load cuts."""

    def test_level3_on_soc_line_and_cold_dark(self) -> None:
        level, _ = classify_alert(
            soc_now=0.18,
            soc_min_forecast=0.11,
            target_cut_frac=0.45,
            mean_pv_kw=0.12,
            mean_temp_c=4.0,
        )
        self.assertEqual(level, 3)

    def test_protected_never_shed(self) -> None:
        hour = np.arange(72) % 24
        t_amb = np.full(72, 5.0)
        load = np.full(72, 2.0)
        catalog = _load_catalog(DEFAULT_CATALOG)
        hvac = _hvac_layer(t_amb, catalog, LoadSynthParams())
        battery = CommercialBattery(
            "toy", 20.0, 8.0, 0.9, 0.0, 4000, "LFP", 0.1, 0.95
        )
        soc = np.linspace(0.22, 0.14, 72)
        pv = np.zeros(72)
        from src.agents.household_ems import BlizzardAlert

        alert = BlizzardAlert(3, 0, 71, 0.22, 0.14, 0.45, 0.1, 5.0, "test")
        action, _ = negotiate(
            alert, hour, t_amb, load, hvac, catalog, LoadSynthParams()
        )
        for name in action.shed_appliances:
            self.assertNotIn(name, PROTECTED)
        self.assertIn("Electric_Clothes_Dryer", action.shed_appliances)
        self.assertIn("Dishwasher", action.shed_appliances)
        self.assertAlmostEqual(action.hvac_temp_offset, -2.5)
        _ = (battery, soc, pv, milp_battery_catalog)

    def test_apply_does_not_create_energy(self) -> None:
        n = 48
        hour = np.arange(n) % 24
        t_amb = np.full(n, 4.0)
        load = np.full(n, 3.0)
        catalog = _load_catalog(DEFAULT_CATALOG)
        params = LoadSynthParams()
        hvac = _hvac_layer(t_amb, catalog, params)
        mask = np.ones(n, dtype=bool)
        action = FamilyAction(
            shed_appliances=("Electric_Clothes_Dryer", "Dishwasher"),
            hvac_temp_offset=-2.5,
            power_cut_kW=2.1,
            light_dim_frac=0.45,
        )
        load_dsr, cut = apply_family_action(
            load, hour, t_amb, hvac, action, catalog, params, mask
        )
        self.assertTrue(np.all(cut >= -1.0e-12))
        self.assertTrue(np.all(load_dsr >= -1.0e-12))
        self.assertTrue(np.all(load_dsr <= load + 1.0e-12))
        self.assertGreater(float(cut.mean()), 0.05)

    def test_blizzard_window_length(self) -> None:
        pv = np.ones(240)
        pv[40:112] = 0.05
        temp = np.full(240, 15.0)
        temp[40:112] = 3.0
        t0, t1 = find_blizzard_window(pv, temp, width_h=72)
        self.assertEqual(t1 - t0 + 1, 72)
        self.assertTrue(40 <= t0 <= 40)

    def test_run_writes_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = run_blizzard_agents(
                agreement_json=root / "family_agreement.json",
                storm_csv=root / "blizzard_dsr_dispatch.csv",
                figure_path=root / "fig_blizzard_dsr_soc.png",
            )
            self.assertTrue((root / "family_agreement.json").is_file())
            payload = json.loads((root / "family_agreement.json").read_text())
            self.assertEqual(payload["alert"]["level"], 3)
            self.assertIn("shed_appliances", payload["action"])
            self.assertGreaterEqual(
                payload["metrics"]["soc_min_dsr"],
                payload["metrics"]["soc_min_no_dsr"] - 1.0e-9,
            )
            self.assertLessEqual(
                payload["metrics"]["def_kwh_dsr"],
                payload["metrics"]["def_kwh_no_dsr"] + 1.0e-6,
            )
            self.assertLess(
                payload["metrics"]["def_kwh_dsr"],
                0.5 * payload["metrics"]["def_kwh_no_dsr"] + 1.0,
            )
            disp = np.loadtxt(
                root / "blizzard_dsr_dispatch.csv", delimiter=",", skiprows=1, usecols=(6, 7)
            )
            self.assertEqual(disp.shape[0], 8760)
            self.assertEqual(out["alert"]["level"], 3)
            _ = butler_alert
            _ = SOC_WARN


if __name__ == "__main__":
    unittest.main()

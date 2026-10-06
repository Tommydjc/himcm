"""Battery SOC / SOH physical guards."""

from __future__ import annotations

import unittest

import numpy as np

from src.battery_model.battery_dynamics import (
    annual_fade_factor,
    get_battery_by_name,
    load_commercial_batteries,
    simulate_battery_year,
    step_soc,
)


class BatteryDynamicsTests(unittest.TestCase):
    """Energy loss law, SOC window, and aging positivity."""

    def test_catalog_loads_key_chemistries(self) -> None:
        packs = load_commercial_batteries()
        names = " ".join(p.model_name.lower() for p in packs)
        self.assertIn("powerwall", names)
        self.assertTrue(any("enphase" in p.model_name.lower() for p in packs))
        pw = get_battery_by_name("Powerwall", packs)
        self.assertAlmostEqual(pw.capacity_kWh, 13.5)
        self.assertAlmostEqual(pw.continuous_power_kW, 5.0)
        self.assertAlmostEqual(pw.roundtrip_eff, 0.90)

    def test_soc_clamp_and_efficiencies(self) -> None:
        bat = get_battery_by_name("Powerwall")
        soc, soh, p_ch, _ = step_soc(0.94, p_ch_kw=5.0, p_dis_kw=0.0, battery=bat)
        self.assertLessEqual(soc, bat.soc_max + 1.0e-12)
        self.assertLess(soh, 1.0)
        self.assertGreater(p_ch, 0.0)
        soc2, _, _, p_dis = step_soc(
            bat.soc_min + 0.01, p_ch_kw=0.0, p_dis_kw=5.0, battery=bat
        )
        self.assertGreaterEqual(soc2, bat.soc_min - 1.0e-12)
        self.assertGreater(p_dis, 0.0)

    def test_charge_discharge_energy_loss(self) -> None:
        """Closed SOC trip: AC output energy < AC input energy (η_rt < 1)."""

        bat = get_battery_by_name("Powerwall")
        soc0 = bat.soc_min + 0.05
        # Phase 1: charge only — discover how much AC actually entered
        ch1 = np.zeros(12, dtype=np.float64)
        dis1 = np.zeros(12, dtype=np.float64)
        ch1[:6] = 4.0
        r1 = simulate_battery_year(ch1, dis1, bat, soc0=soc0)
        e_to_store = r1.energy_to_store_kWh
        self.assertGreater(e_to_store, 1.0)
        # Phase 2: from r1 end SOC, discharge exactly the stored chemical energy
        e_ac_out = e_to_store * bat.eta_dis * 0.98
        hours = max(int(np.ceil(e_ac_out / 3.0)), 1)
        ch2 = np.zeros(hours + 2, dtype=np.float64)
        dis2 = np.zeros(hours + 2, dtype=np.float64)
        dis2[:hours] = e_ac_out / hours
        r2 = simulate_battery_year(ch2, dis2, bat, soc0=float(r1.soc[-1]))
        e_in = r1.energy_in_kWh
        e_out = r2.energy_out_kWh
        self.assertGreater(e_in, 0.0)
        self.assertGreater(e_out, 0.0)
        self.assertLess(e_out, e_in)
        # Per-step conversion losses
        self.assertLess(r1.energy_to_store_kWh, r1.energy_in_kWh + 1.0e-9)
        self.assertLess(r2.energy_out_kWh, r2.energy_from_store_kWh + 1.0e-9)
        self.assertAlmostEqual(float(r2.soc[-1]), soc0, delta=0.08)
        self.assertTrue(np.all(r1.soc >= bat.soc_min - 1.0e-9))
        self.assertTrue(np.all(r2.soc <= bat.soc_max + 1.0e-9))

    def test_annual_soh_fade_positive(self) -> None:
        bat = get_battery_by_name("FREEDOH")
        rng = np.random.default_rng(0)
        n = 8760
        net = rng.normal(0.0, 0.8, size=n)
        ch = np.clip(net, 0.0, bat.continuous_power_kW)
        dis = np.clip(-net, 0.0, bat.continuous_power_kW)
        both = (ch > 0) & (dis > 0)
        dis[both] = 0.0
        result = simulate_battery_year(ch, dis, bat)
        self.assertGreater(result.soh_fade_year, 0.0)
        self.assertLess(result.soh_fade_year, 0.5)
        self.assertAlmostEqual(annual_fade_factor(result), 1.0 - result.soh_fade_year)
        self.assertLess(result.capacity_end_kWh, bat.capacity_kWh)


if __name__ == "__main__":
    unittest.main()

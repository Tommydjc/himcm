"""PV / load 8760-hour physical guards."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.generation.solar_pv import PVArrayParams, simulate_pv_timeseries
from src.load_profiler.mcmc_load import synthesize_load_kw
from src.scripts.run_pv_load import run_annual_pv_load


class PVLoadTests(unittest.TestCase):
    """Non-negativity, length, and net-power identity."""

    def test_pv_length_and_night(self) -> None:
        pv = simulate_pv_timeseries()
        self.assertEqual(len(pv), 8760)
        self.assertTrue((pv["pv_output_kW"] >= -1.0e-12).all())
        night = pv["ghi_W_m2"] <= 1.0
        self.assertLess(float(pv.loc[night, "pv_output_kW"].mean()), 0.15)

    def test_load_nonneg_and_hvac_coupling(self) -> None:
        pv = simulate_pv_timeseries()
        load = synthesize_load_kw(
            pv["hour"].to_numpy(), pv["temperature_c"].to_numpy()
        )
        self.assertEqual(len(load), 8760)
        self.assertTrue((load["load_demand_kW"] >= 0.0).all())
        hot = pv["temperature_c"] > 32.0
        cool = pv["temperature_c"] < 10.0
        if hot.any() and cool.any():
            self.assertGreater(
                float(load.loc[hot.to_numpy(), "load_hvac_kW"].mean()),
                0.0,
            )

    def test_pipeline_net_identity(self) -> None:
        tmp = Path("/tmp/annual_pv_load_8760h_test.csv")
        table = run_annual_pv_load(out_csv=tmp)
        self.assertEqual(list(table.columns), [
            "hour",
            "month",
            "pv_output_kW",
            "load_demand_kW",
            "net_power_kW",
        ])
        net = table["pv_output_kW"] - table["load_demand_kW"]
        np.testing.assert_allclose(table["net_power_kW"].to_numpy(), net.to_numpy())
        packed = pd.read_csv(tmp)
        self.assertEqual(len(packed), 8760)


if __name__ == "__main__":
    unittest.main()

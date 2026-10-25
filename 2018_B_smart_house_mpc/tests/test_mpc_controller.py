"""Phase-5 rolling MPC controller tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / "src"))

from control_mpc.mpc_controller import (
    STEPS_PER_DAY,
    aggregate_disturbances_15min,
    load_weather_1min,
    simulate_mpc_day,
    solve_mpc_step,
)
from control_mpc.state_space_2r2c import DEFAULT_TS_S, build_continuous_ss, discretize
from thermal_physics.building_rc import BuildingThermalEnvelope
from comfort_tariffs.models import TOUTariffSchedule
from baseline_thermostat.bang_bang import write_weather_csv


def test_aggregate_shape() -> None:
    write_weather_csv()
    D = aggregate_disturbances_15min(load_weather_1min())
    assert D.shape == (STEPS_PER_DAY, 3)


def test_solve_mpc_step_bounds() -> None:
    write_weather_csv()
    env = BuildingThermalEnvelope.from_json()
    dss = discretize(build_continuous_ss(env), DEFAULT_TS_S, method="zoh")
    D = aggregate_disturbances_15min(load_weather_1min())
    sol = solve_mpc_step(
        np.array([23.5, 25.0]),
        D,
        40,
        dss=dss,
        env=env,
        tou=TOUTariffSchedule.from_json(),
        H=8,
        maxiter=15,
    )
    assert 0.0 <= sol["u0"] <= 1.0
    assert sol["U"].shape[0] == 8


def test_simulate_mpc_day_length() -> None:
    write_weather_csv()
    # Short horizon for speed in CI-like runs.
    out = simulate_mpc_day(H=8, maxiter=12)
    assert len(out["trajectory"]) == 1440
    assert out["metrics"]["energy_kwh"] >= 0.0
    assert out["metrics"]["max_T_in_C"] < 40.0

"""Phase-1 teaching 2R2C envelope tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / "src"))

from thermal_physics.building_rc import (
    BuildingThermalEnvelope,
    ThermalDynamics2R2C,
    assert_free_float_to_outdoor,
    assert_steady_state_is_outdoor,
    rk4_step,
    simulate_rk4,
)


def test_envelope_geometry_derived() -> None:
    env = BuildingThermalEnvelope.from_json()
    assert abs(env.volume_m3 - 180.0 * 2.8) < 1e-9
    assert env.window_area_m2 == env.window_wall_ratio * env.wall_area_m2
    assert env.wall_area_m2 > env.window_area_m2 > 0.0


def test_cop_formula() -> None:
    env = BuildingThermalEnvelope.from_json()
    assert abs(env.cop(25.0) - 3.8) < 1e-12
    assert abs(env.cop(35.0) - (3.8 - 0.04 * 10.0)) < 1e-12
    assert env.cop(100.0) == env.COP_min


def test_cooling_lowers_air_temp() -> None:
    model = ThermalDynamics2R2C()
    ti, tw = 28.0, 28.0
    for _ in range(30):
        ti, tw = rk4_step(
            model,
            ti,
            tw,
            T_out=32.0,
            solar_rad=0.0,
            q_hvac=-5000.0,
            internal_gains=0.0,
            dt_s=60.0,
        )
    assert ti < 28.0


def test_steady_state_outdoor() -> None:
    assert_steady_state_is_outdoor(T_out=35.0)


def test_free_float_monotonic_and_energy() -> None:
    # 7-day free-float: monotone + ≥45% gap closed + energy balance.
    stats = assert_free_float_to_outdoor(hours=168.0, min_progress=0.45)
    assert stats["progress"] >= 0.45
    assert stats["energy_residual_rel"] < 2e-3


def test_simulate_shapes() -> None:
    model = ThermalDynamics2R2C()
    out = simulate_rk4(
        model,
        24.0,
        25.0,
        T_out=30.0,
        q_hvac=-1000.0,
        n_steps=10,
    )
    assert out["T_in"].shape == (11,)
    assert out["P_elec_W"].shape == (10,)
    assert np.all(out["P_elec_W"] >= 0.0)


def test_continuous_matrices_shapes() -> None:
    A, B, E = ThermalDynamics2R2C().continuous_matrices()
    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert E.shape == (2, 3)

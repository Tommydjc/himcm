"""Unit tests: energy/IAQ physics bounds, PMV range, MPC feasibility."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / "src"))

from comfort_tariffs.fanger_pmv import fanger_pmv
from control_mpc.arrival_predictor import arrival_probability, thermal_lead_time_h
from control_mpc.mpc_solver import solve_mpc_step
from thermal_physics.multi_zone_coupling import allocate_dampers, graph_laplacian_conductance
from thermal_physics.single_zone_rc import (
    ControlInput,
    Disturbance,
    free_float_check,
    load_building_specs,
    relative_humidity_pct,
    rk4_step,
    simulate_horizon,
)


def test_rh_conversion_bounds() -> None:
    rh = relative_humidity_pct(25.0, 10.0)
    assert 20.0 < rh < 80.0


def test_free_float_allergen_rises_when_humid() -> None:
    out = free_float_check(hours=18, T_out=32.0, RH_out=75.0)
    assert out["peak_RH_pct"] > 55.0
    assert out["peak_C_pol"] >= out["final_C_pol"] * 0.5


def test_rk4_energy_direction_cooling() -> None:
    specs = load_building_specs()
    x = np.asarray([28.0, 28.0, 10.0, 20.0], dtype=float)
    u = ControlInput(-4500.0, 0.0, 0.02, 0.5)
    # Night-like disturbance: low solar so cooling dominates.
    d = Disturbance(30.0, 12.0, 25.0, 0.0, 0.0)
    x2 = x.copy()
    for _ in range(60):
        x2 = rk4_step(x2, u, d, specs, 300.0)
    assert float(x2[0]) < float(x[0])


def test_simulate_horizon_shapes() -> None:
    specs = load_building_specs()
    controls = [ControlInput(-1000.0, -0.1, 0.04, 0.4) for _ in range(6)]
    dist = [Disturbance(30.0, 12.0, 25.0, 400.0, 1.0) for _ in range(6)]
    X, P, RH = simulate_horizon([24.0, 25.0, 10.0, 20.0], controls, dist, specs)
    assert X.shape == (7, 4)
    assert P.shape == (6,)
    assert RH.shape == (7,)
    assert np.all(P >= 0.0)


def test_pmv_near_comfort() -> None:
    pmv, ppd = fanger_pmv(23.0, 45.0)
    assert -1.0 <= pmv <= 1.0
    assert 5.0 <= ppd <= 40.0


def test_damper_sum_to_one() -> None:
    u = allocate_dampers([1.0, 0.0, 0.5, 0.0, 0.2])
    assert abs(float(np.sum(u)) - 1.0) < 1e-9
    assert np.all(u > 0.0)


def test_laplacian_psd_diagonal() -> None:
    specs = load_building_specs()
    L = graph_laplacian_conductance(specs)
    assert np.allclose(L, L.T)
    assert np.all(np.diag(L) >= 0.0)


def test_arrival_prob_and_lead() -> None:
    p = arrival_probability(0.5, distance_km=5.0, speed_km_h=40.0, prior=0.1)
    assert 0.0 <= p <= 1.0
    lead = thermal_lead_time_h(28.0, 23.0, 33.0, P_hvac_W=4500.0, C_air=3e5, R_win=0.015)
    assert 0.0 < lead <= 6.0


def test_mpc_step_returns_control() -> None:
    weather = [
        {
            "hour_of_day": 15,
            "T_out_C": 34.0,
            "w_out_g_per_kg": 14.0,
            "C_out_ug_per_m3": 40.0,
            "I_solar_W_per_m2": 600.0,
        }
        for _ in range(3)
    ]
    people = [
        {
            "T_pref_day_C": 23.0,
            "T_pref_night_C": 22.0,
            "RH_pref_pct": 45.0,
            "C_thresh_ug_per_m3": 30.0,
            "omega_T": 1.0,
            "omega_H": 0.5,
            "omega_P": 1.0,
        }
    ]
    p_occ = np.ones((3, 1))
    sol = solve_mpc_step(
        [26.0, 27.0, 11.0, 25.0],
        weather,
        p_occ,
        people,
        maxiter=15,
    )
    assert "first" in sol
    assert -5000.0 <= sol["first"].P_HVAC <= 5000.0

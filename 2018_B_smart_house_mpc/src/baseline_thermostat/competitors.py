"""
Competitor controllers for Task 1b (shared physics sandbox).

1. Programmable bang-bang with fixed weekday schedule.
2. First-gen remote App: occupied setpoints only when home.
3. Nest-like schedule learner: EMA of past occupancy hours.
4. Ecobee-like multi-sensor: occupancy + mild humidity bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from thermal_physics.single_zone_rc import (
    ControlInput,
    Disturbance,
    load_building_specs,
    relative_humidity_pct,
    rk4_step,
    electric_power_W,
)


@dataclass
class ControllerState:
    """Mutable controller memory across hours."""

    occ_ema: np.ndarray  # length 24
    last_mode: str = "off"


def _pi_power(
    T: float,
    T_set: float,
    *,
        cool_gain: float = 1800.0,
        heat_gain: float = 1200.0,
    deadband: float = 0.4,
    specs: Mapping[str, Any],
) -> float:
    """Proportional HVAC power with deadband (W)."""
    err = T - T_set
    if abs(err) <= deadband:
        return 0.0
    if err > 0.0:
        return float(np.clip(-cool_gain * err, -float(specs["P_cool_max_W"]), 0.0))
    return float(np.clip(-heat_gain * err, 0.0, float(specs["P_heat_max_W"])))


def _iaq_actuators(
    RH: float,
    C_pol: float,
    *,
    aggressive: bool,
    specs: Mapping[str, Any],
) -> tuple[float, float, float]:
    """Heuristic humidity / vent / purifier commands."""
    m_hum = 0.0
    if RH > 55.0:
        m_hum = -0.35 if aggressive else -0.15
    elif RH < 35.0:
        m_hum = 0.2
    Q = float(specs["Q_vent_min_m3_per_s"])
    u_pur = 0.2
    if C_pol > 30.0:
        Q = float(specs["Q_vent_max_m3_per_s"]) * (0.7 if aggressive else 0.4)
        u_pur = 0.9 if aggressive else 0.5
        # High outdoor allergy: prefer recirculation + HEPA over full outdoor air.
        if aggressive:
            Q = float(specs["Q_vent_min_m3_per_s"]) + 0.02
            u_pur = 1.0
    return m_hum, Q, u_pur


def programmable_setpoint(hour: int, occupied: bool) -> float:
    """Classic 4-period programmable thermostat (°C)."""
    if not occupied:
        return 27.5  # setback cooling
    if 6 <= hour < 9:
        return 23.5
    if 9 <= hour < 17:
        return 26.5
    if 17 <= hour < 22:
        return 23.0
    return 22.0


def nest_like_setpoint(hour: int, occ_ema: np.ndarray) -> float:
    """
    Nest-style learned schedule: cool more when EMA occupancy is high.
    """
    p = float(occ_ema[int(hour) % 24])
    if p < 0.25:
        return 27.0
    if p < 0.55:
        return 24.5
    return 22.8


def run_controller_week(
    name: str,
    weather: Sequence[Mapping[str, float]],
    occupancy: Sequence[float],
    *,
    specs: Mapping[str, Any] | None = None,
    x0: Sequence[float] | None = None,
) -> Dict[str, Any]:
    """
    Simulate one competitor over the weather horizon.

    Parameters
    ----------
    name :
        One of {programmable, remote_app, nest_like, ecobee_like}.
    weather :
        Hourly exogenous rows.
    occupancy :
        Fraction occupied in [0, 1] per hour.
    """
    specs = dict(specs or load_building_specs())
    n = len(weather)
    if len(occupancy) != n:
        raise ValueError("occupancy length mismatch")
    x = np.asarray(x0 or [24.0, 25.0, 10.0, 22.0], dtype=np.float64)
    mem = ControllerState(occ_ema=np.full(24, 0.4))
    rows: List[Dict[str, float]] = []
    for k in range(n):
        row = weather[k]
        hod = int(row["hour_of_day"])
        occ = float(occupancy[k])
        # EMA update uses realized occupancy (Nest learns after the fact).
        alpha = 0.15
        mem.occ_ema[hod] = (1 - alpha) * mem.occ_ema[hod] + alpha * occ
        RH = relative_humidity_pct(float(x[0]), float(x[2]))
        if name == "programmable":
            T_set = programmable_setpoint(hod, occ > 0.5)
            aggressive = False
        elif name == "remote_app":
            T_set = 23.0 if occ > 0.5 else 28.0
            aggressive = False
        elif name == "nest_like":
            T_set = nest_like_setpoint(hod, mem.occ_ema)
            aggressive = False
        elif name == "ecobee_like":
            T_set = 23.2 if occ > 0.35 else 27.0
            if RH > 58.0:
                T_set -= 0.4  # mild dehumidify assist via cooler coil
            aggressive = True
        else:
            raise ValueError(f"unknown controller {name}")

        P = _pi_power(float(x[0]), T_set, specs=specs)
        m_hum, Q, u_pur = _iaq_actuators(RH, float(x[3]), aggressive=aggressive, specs=specs)
        u = ControlInput(P, m_hum, Q, u_pur)
        d = Disturbance(
            T_out=float(row["T_out_C"]),
            w_out=float(row["w_out_g_per_kg"]),
            C_out=float(row["C_out_ug_per_m3"]),
            I_solar=float(row["I_solar_W_per_m2"]),
            n_occ=occ * 2.0,
        )
        p_elec = electric_power_W(u, d.T_out, specs)
        for _ in range(12):
            x = rk4_step(x, u, d, specs, 300.0)
        RH2 = relative_humidity_pct(float(x[0]), float(x[2]))
        rows.append(
            {
                "hour_index": float(row["hour_index"]),
                "hour_of_day": float(hod),
                "T_in": float(x[0]),
                "T_env": float(x[1]),
                "RH": RH2,
                "C_pol": float(x[3]),
                "P_elec_W": p_elec,
                "P_HVAC_W": float(u.P_HVAC),
                "T_set": T_set,
                "occupied": occ,
                "controller": 0.0,
            }
        )
    return {"name": name, "trajectory": rows, "final_state": x.tolist()}

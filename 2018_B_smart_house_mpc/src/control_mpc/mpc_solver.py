"""
Rolling-horizon MPC for AeroTherm (SciPy SLSQP).

Objective: TOU energy cost + Nash social discomfort + control slew penalty.
Hard soft-penalties for RH > 60% and T safety bounds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from comfort_tariffs.tariff_calculator import rate_usd_per_kwh
from control_mpc.arrival_predictor import (
    nash_social_discomfort,
    thermal_lead_time_h,
)
from thermal_physics.single_zone_rc import (
    ControlInput,
    Disturbance,
    electric_power_W,
    load_building_specs,
    relative_humidity_pct,
    rk4_step,
)


def _pack_u(vec: np.ndarray) -> List[ControlInput]:
    """Flattened [P, m, Q, u_pur] * H → ControlInput list."""
    H = len(vec) // 4
    out: List[ControlInput] = []
    for k in range(H):
        out.append(
            ControlInput(
                P_HVAC=float(vec[4 * k]),
                m_hum=float(vec[4 * k + 1]),
                Q_vent=float(vec[4 * k + 2]),
                u_pur=float(vec[4 * k + 3]),
            )
        )
    return out


def _bounds(H: int, specs: Mapping[str, Any]) -> List[tuple[float, float]]:
    b: List[tuple[float, float]] = []
    for _ in range(H):
        b.extend(
            [
                (-float(specs["P_cool_max_W"]), float(specs["P_heat_max_W"])),
                (-float(specs["m_dehum_max_g_per_s"]), float(specs["m_hum_max_g_per_s"])),
                (float(specs["Q_vent_min_m3_per_s"]), float(specs["Q_vent_max_m3_per_s"])),
                (0.0, 1.0),
            ]
        )
    return b


def mpc_cost(
    vec: np.ndarray,
    *,
    x0: np.ndarray,
    weather_h: Sequence[Mapping[str, float]],
    p_occ_h: np.ndarray,
    people: Sequence[Mapping[str, Any]],
    rho_fair: float,
    specs: Mapping[str, Any],
    alpha_cost: float = 1.0,
    beta_comf: float = 0.045,
    gamma_smooth: float = 1e-8,
    surge: float = 1.0,
) -> float:
    """Scalar stage cost over horizon H = len(weather_h)."""
    controls = _pack_u(vec)
    H = len(controls)
    x = x0.copy()
    J = 0.0
    u_prev = None
    for k in range(H):
        row = weather_h[k]
        u = controls[k].clip(specs)
        d = Disturbance(
            T_out=float(row["T_out_C"]),
            w_out=float(row["w_out_g_per_kg"]),
            C_out=float(row["C_out_ug_per_m3"]),
            I_solar=float(row["I_solar_W_per_m2"]),
            n_occ=float(np.sum(p_occ_h[k])),
        )
        p_elec = electric_power_W(u, d.T_out, specs)
        rate = rate_usd_per_kwh(int(row["hour_of_day"]), surge_multiplier=surge)
        J += alpha_cost * rate * (p_elec / 1000.0)
        for _ in range(3):
            x = rk4_step(x, u, d, specs, 1200.0)
        RH = relative_humidity_pct(float(x[0]), float(x[2]))
        J += beta_comf * nash_social_discomfort(
            float(x[0]),
            RH,
            float(x[3]),
            int(row["hour_of_day"]),
            p_occ_h[k],
            people,
            rho_fair=rho_fair,
        )
        # Soft CBF-style barriers.
        if RH > 60.0:
            J += 25.0 * (RH - 60.0) ** 2
        if float(x[0]) < float(specs["T_safety_min_C"]):
            J += 40.0 * (float(specs["T_safety_min_C"]) - float(x[0])) ** 2
        if float(x[0]) > float(specs["T_safety_max_C"]):
            J += 40.0 * (float(x[0]) - float(specs["T_safety_max_C"])) ** 2
        if u_prev is not None:
            du = np.asarray(
                [
                    u.P_HVAC - u_prev.P_HVAC,
                    u.m_hum - u_prev.m_hum,
                    u.Q_vent - u_prev.Q_vent,
                    u.u_pur - u_prev.u_pur,
                ]
            )
            J += gamma_smooth * float(du @ du)
        u_prev = u
    return float(J)


def solve_mpc_step(
    x0: Sequence[float],
    weather_h: Sequence[Mapping[str, float]],
    p_occ_h: np.ndarray,
    people: Sequence[Mapping[str, Any]],
    *,
    rho_fair: float = 0.35,
    specs: Mapping[str, Any] | None = None,
    surge: float = 1.0,
    maxiter: int = 40,
) -> Dict[str, Any]:
    """
    Solve one MPC window; return first-step control and diagnostics.
    """
    specs = dict(specs or load_building_specs())
    H = len(weather_h)
    x0_arr = np.asarray(x0, dtype=np.float64)
    # Warm start: cool if hot, modest dehumidify, mid vent, purifier mid.
    u0 = []
    for row in weather_h:
        cool = -2500.0 if float(row["T_out_C"]) > 28 else -800.0
        u0.extend([cool, -0.2, 0.05, 0.6])
    u0_arr = np.asarray(u0, dtype=np.float64)

    # Pre-cool nudge when TOU peak approaches and ETA lead time large.
    T_set = 23.0
    lead = thermal_lead_time_h(
        float(x0_arr[0]),
        T_set,
        float(weather_h[0]["T_out_C"]),
        P_hvac_W=float(specs["P_cool_max_W"]),
        C_air=float(specs["C_air_J_per_K"]),
        R_win=float(specs["R_win_K_per_W"]),
    )

    res = minimize(
        lambda v: mpc_cost(
            v,
            x0=x0_arr,
            weather_h=weather_h,
            p_occ_h=p_occ_h,
            people=people,
            rho_fair=rho_fair,
            specs=specs,
            surge=surge,
        ),
        u0_arr,
        method="L-BFGS-B",
        bounds=_bounds(H, specs),
        options={"maxiter": maxiter, "ftol": 1e-3, "maxfun": 200},
    )
    controls = _pack_u(res.x)
    return {
        "success": bool(res.success),
        "cost": float(res.fun),
        "controls": controls,
        "first": controls[0],
        "lead_time_h": lead,
        "nit": int(getattr(res, "nit", 0)),
    }


def simulate_mpc_week(
    weather: Sequence[Mapping[str, float]],
    scenario: Mapping[str, Any],
    *,
    horizon: int = 6,
    specs: Mapping[str, Any] | None = None,
    surge: float = 1.0,
    x0: Sequence[float] | None = None,
) -> Dict[str, Any]:
    """
    Receding-horizon closed-loop simulation for the full weather array.
    """
    specs = dict(specs or load_building_specs())
    n = len(weather)
    people = list(scenario["profiles"])
    rho = float(scenario["rho_fair"])
    p_all = np.asarray(scenario["p_arrival"], dtype=np.float64)
    occ = np.asarray(scenario["occupancy"], dtype=np.float64)
    # Use max(arrival prob, occupancy) as planning occupancy weight.
    p_plan = np.maximum(p_all, occ)

    x = np.asarray(x0 or [24.5, 25.0, 10.0, 22.0], dtype=np.float64)
    traj: List[Dict[str, float]] = []
    schedule: List[Dict[str, float]] = []

    for k in range(n):
        H = min(horizon, n - k)
        weather_h = weather[k : k + H]
        p_occ_h = p_plan[k : k + H]
        sol = solve_mpc_step(
            x,
            weather_h,
            p_occ_h,
            people,
            rho_fair=rho,
            specs=specs,
            surge=surge,
            maxiter=12,
        )
        u = sol["first"].clip(specs)
        row = weather[k]
        d = Disturbance(
            T_out=float(row["T_out_C"]),
            w_out=float(row["w_out_g_per_kg"]),
            C_out=float(row["C_out_ug_per_m3"]),
            I_solar=float(row["I_solar_W_per_m2"]),
            n_occ=float(np.sum(occ[k])),
        )
        p_elec = electric_power_W(u, d.T_out, specs)
        for _ in range(12):
            x = rk4_step(x, u, d, specs, 300.0)
        RH = relative_humidity_pct(float(x[0]), float(x[2]))
        discomfort = nash_social_discomfort(
            float(x[0]),
            RH,
            float(x[3]),
            int(row["hour_of_day"]),
            p_plan[k],
            people,
            rho_fair=rho,
        )
        traj.append(
            {
                "hour_index": float(row["hour_index"]),
                "hour_of_day": float(row["hour_of_day"]),
                "T_in": float(x[0]),
                "T_env": float(x[1]),
                "RH": RH,
                "C_pol": float(x[3]),
                "P_elec_W": p_elec,
                "P_HVAC_W": float(u.P_HVAC),
                "m_hum": float(u.m_hum),
                "Q_vent": float(u.Q_vent),
                "u_pur": float(u.u_pur),
                "occupied": float(np.mean(occ[k])),
                "nash_discomfort": discomfort,
                "lead_time_h": float(sol["lead_time_h"]),
                "mpc_success": 1.0 if sol["success"] else 0.0,
            }
        )
        schedule.append(
            {
                "hour_index": float(row["hour_index"]),
                "P_HVAC_W": float(u.P_HVAC),
                "P_elec_W": p_elec,
                "rate_usd_per_kwh": rate_usd_per_kwh(int(row["hour_of_day"]), surge_multiplier=surge),
                "lead_time_h": float(sol["lead_time_h"]),
            }
        )
    return {"trajectory": traj, "schedule": schedule, "final_state": x.tolist()}

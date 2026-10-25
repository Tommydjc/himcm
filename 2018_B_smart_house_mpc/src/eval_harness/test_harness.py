"""
Black-box stress harness: heatwave, tariff surge, sensor drift.

Returns KPI vectors suitable for a radar chart.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from baseline_thermostat.competitors import run_controller_week
from comfort_tariffs.fanger_pmv import discomfort_degree_hours, fanger_pmv
from comfort_tariffs.tariff_calculator import bill_from_power_W
from control_mpc.arrival_predictor import build_occupancy_scenario
from control_mpc.mpc_solver import simulate_mpc_week


def _kpis_from_traj(
    traj: Sequence[Mapping[str, float]],
    *,
    T_set: float = 23.0,
    surge: float = 1.0,
) -> Dict[str, float]:
    """Aggregate energy, cost, discomfort, allergen, RH-violation KPIs."""
    df = pd.DataFrame(traj)
    bill = bill_from_power_W(
        df["P_elec_W"].to_numpy(),
        df["hour_of_day"].astype(int).to_numpy(),
        surge_multiplier=surge,
    )
    occ = df["occupied"].to_numpy()
    ddH = discomfort_degree_hours(df["T_in"].to_numpy(), T_set, occ)
    allergen_hours = float(np.sum((df["C_pol"].to_numpy() > 35.0) * np.maximum(occ, 0.0)))
    rh_viol = float(np.sum(df["RH"].to_numpy() > 60.0))
    pmvs = []
    for _, row in df.iterrows():
        if float(row["occupied"]) < 0.3:
            continue
        pmv, _ = fanger_pmv(float(row["T_in"]), float(row["RH"]))
        pmvs.append(abs(pmv))
    mean_abs_pmv = float(np.mean(pmvs)) if pmvs else 0.0
    return {
        "energy_kwh": bill["energy_kwh"],
        "cost_usd": bill["cost_usd"],
        "discomfort_degree_hours": ddH,
        "allergen_over_hours": allergen_hours,
        "rh_over60_hours": rh_viol,
        "mean_abs_pmv": mean_abs_pmv,
        "peak_C_pol": float(df["C_pol"].max()),
        "mean_T_in": float(df["T_in"].mean()),
    }


def apply_heatwave(
    weather: List[Dict[str, float]],
    *,
    delta_T: float = 4.0,
) -> List[Dict[str, float]]:
    """Shift outdoor temperature up (heatwave shock)."""
    out = []
    for row in weather:
        r = dict(row)
        r["T_out_C"] = float(r["T_out_C"]) + delta_T
        out.append(r)
    return out


def apply_sensor_drift(
    traj_hook_T: float = 1.2,
) -> float:
    """Return additive temperature sensor bias used by callers."""
    return float(traj_hook_T)


def run_stress_battery(
    weather: Sequence[Mapping[str, float]],
    *,
    seed: int = 2018,
) -> Dict[str, Any]:
    """
    Run baseline nest_like vs AeroTherm-MPC under three shocks.
    """
    weather_list = [dict(r) for r in weather]
    n = len(weather_list)
    scenario = build_occupancy_scenario(n, seed=seed)
    occ_mean = scenario["occupancy"].mean(axis=1)

    cases: Dict[str, List[Dict[str, float]]] = {
        "nominal": weather_list,
        "heatwave": apply_heatwave(weather_list, delta_T=4.0),
        "tariff_surge": weather_list,
    }
    results: Dict[str, Any] = {}
    for case, w in cases.items():
        surge = 1.8 if case == "tariff_surge" else 1.0
        nest = run_controller_week("nest_like", w, occ_mean.tolist())
        mpc = simulate_mpc_week(w, scenario, horizon=3, surge=surge)
        # Sensor drift: evaluate MPC traj with biased T for discomfort only.
        nest_k = _kpis_from_traj(nest["trajectory"], surge=surge)
        mpc_k = _kpis_from_traj(mpc["trajectory"], surge=surge)
        if case == "nominal":
            drift = apply_sensor_drift(1.2)
            drifted = []
            for row in mpc["trajectory"]:
                r = dict(row)
                r["T_in"] = float(r["T_in"]) + drift
                drifted.append(r)
            results["sensor_drift_mpc"] = _kpis_from_traj(drifted, surge=1.0)
        results[f"{case}_nest"] = nest_k
        results[f"{case}_mpc"] = mpc_k
    return results


def radar_normalized(rows: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """
    Normalize selected KPIs to [0,1] (lower better) for radar plotting.
    """
    metrics = [
        "energy_kwh",
        "cost_usd",
        "discomfort_degree_hours",
        "allergen_over_hours",
        "rh_over60_hours",
    ]
    df = pd.DataFrame(rows).T
    out = df[metrics].copy()
    for m in metrics:
        mx = float(out[m].max())
        mn = float(out[m].min())
        if mx - mn < 1e-12:
            out[m] = 0.0
        else:
            out[m] = (out[m] - mn) / (mx - mn)
    return out

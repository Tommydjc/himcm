"""
One-click AeroTherm-MPC pipeline: simulate, benchmark, stress, plot, export.

Run from pack root:
  PYTHONPATH=src /opt/anaconda3/bin/python -m scripts.run_full_pipeline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACK_ROOT = Path(__file__).resolve().parents[2]
SRC = PACK_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_inhabitant.inhabitant_mas import bargain_setpoints
from baseline_thermostat.competitors import run_controller_week
from comfort_tariffs.fanger_pmv import discomfort_degree_hours
from comfort_tariffs.tariff_calculator import bill_from_power_W
from control_mpc.arrival_predictor import build_occupancy_scenario
from control_mpc.mpc_solver import simulate_mpc_week
from eval_harness.test_harness import radar_normalized, run_stress_battery
from thermal_physics.multi_zone_coupling import simulate_multi_zone
from thermal_physics.single_zone_rc import free_float_check, load_building_specs


RESULTS = PACK_ROOT / "results"
FIGS = PACK_ROOT / "paper_figures"
WEATHER = PACK_ROOT / "data" / "raw" / "weather_tmy3_summer.csv"


def load_weather() -> List[Dict[str, Any]]:
    """Load hourly weather rows as list[dict]."""
    df = pd.read_csv(WEATHER)
    return df.to_dict(orient="records")


def _traj_kpis(traj: List[Dict[str, float]], name: str) -> Dict[str, Any]:
    df = pd.DataFrame(traj)
    bill = bill_from_power_W(df["P_elec_W"], df["hour_of_day"].astype(int))
    occ = df["occupied"].to_numpy()
    return {
        "controller": name,
        "energy_kwh": bill["energy_kwh"],
        "cost_usd": bill["cost_usd"],
        "peak_energy_kwh": bill["peak_energy_kwh"],
        "discomfort_degree_hours": discomfort_degree_hours(
            df["T_in"].to_numpy(), 23.0, occ
        ),
        "allergen_over_hours": float(np.sum((df["C_pol"] > 35.0) * np.maximum(occ, 0))),
        "rh_over60_hours": float(np.sum(df["RH"] > 60.0)),
        "mean_T_in": float(df["T_in"].mean()),
        "mean_RH": float(df["RH"].mean()),
        "peak_C_pol": float(df["C_pol"].max()),
    }


def plot_indoor_outdoor(mpc_df: pd.DataFrame, weather_df: pd.DataFrame, path: Path) -> None:
    """Fig: indoor vs outdoor temperature and RH."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    t = mpc_df["hour_index"].to_numpy()
    ax.plot(t, weather_df["T_out_C"].to_numpy()[: len(t)], color="#e67e22", label="T_out")
    ax.plot(t, mpc_df["T_in"].to_numpy(), color="#2980b9", label="T_in (AeroTherm-MPC)")
    ax.axvspan(14, 19, color="#e74c3c", alpha=0.12, label="TOU peak window (daily)")
    # Mark peak hours each day.
    for day in range(int(t.max()) // 24 + 1):
        ax.axvspan(day * 24 + 14, day * 24 + 19, color="#e74c3c", alpha=0.08)
    ax.set_xlabel("Hour index")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Indoor–outdoor temperature under AeroTherm-MPC")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_peak_shift(bench: pd.DataFrame, path: Path) -> None:
    """Fig: peak-period energy bar chart across controllers."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    names = bench["controller"].tolist()
    vals = bench["peak_energy_kWh"].tolist() if "peak_energy_kWh" in bench else bench["peak_energy_kwh"].tolist()
    colors = ["#95a5a6", "#7f8c8d", "#3498db", "#9b59b6", "#c0392b"]
    ax.bar(names, vals, color=colors[: len(names)])
    ax.set_ylabel("On-peak energy (kWh)")
    ax.set_title("Peak-period HVAC site energy (TOU 14:00–19:00)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_radar(stress: Dict[str, Any], path: Path) -> None:
    """Fig: harness radar for nest vs mpc under heatwave."""
    subset = {
        "nest_heatwave": stress["heatwave_nest"],
        "mpc_heatwave": stress["heatwave_mpc"],
        "nest_nominal": stress["nominal_nest"],
        "mpc_nominal": stress["nominal_mpc"],
    }
    rad = radar_normalized(subset)
    labels = list(rad.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=300, subplot_kw=dict(polar=True))
    for name, row in rad.iterrows():
        vals = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, vals, label=str(name), linewidth=1.8)
        ax.fill(angles, vals, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("Stress harness (normalized; lower better)", pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Execute full study pipeline and write locked CSV/PNG artifacts."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    MPL = Path("/tmp/mpl")
    MPL.mkdir(exist_ok=True)

    print("[1] free-float physics check...")
    ff = free_float_check()
    print("    ", ff)

    weather = load_weather()
    # 36 h covers irregular returns with acceptable MPC wall-clock.
    horizon_hours = 36
    weather = weather[:horizon_hours]
    weather_df = pd.DataFrame(weather)

    print("[2] occupancy / geofence scenario...")
    scenario = build_occupancy_scenario(horizon_hours, seed=2018)
    occ_mean = scenario["occupancy"].mean(axis=1)

    print("[3] competitor benchmarks...")
    names = ["programmable", "remote_app", "nest_like", "ecobee_like"]
    bench_rows = []
    traj_store = {}
    for name in names:
        print(f"    running {name}...")
        out = run_controller_week(name, weather, occ_mean.tolist())
        traj_store[name] = out["trajectory"]
        bench_rows.append(_traj_kpis(out["trajectory"], name))

    print("[4] AeroTherm-MPC rolling horizon...")
    mpc = simulate_mpc_week(weather, scenario, horizon=3)
    traj_store["aerotherm_mpc"] = mpc["trajectory"]
    bench_rows.append(_traj_kpis(mpc["trajectory"], "aerotherm_mpc"))
    mpc_df = pd.DataFrame(mpc["trajectory"])
    sched_df = pd.DataFrame(mpc["schedule"])
    mpc_df.to_csv(RESULTS / "hourly_dynamics.csv", index=False)
    sched_df.to_csv(RESULTS / "mpc_schedule.csv", index=False)

    bench = pd.DataFrame(bench_rows)
    # Friendly column alias for plot
    bench["peak_energy_kWh"] = bench["peak_energy_kwh"]
    bench.to_csv(RESULTS / "competitor_benchmarks.csv", index=False)
    print(bench.to_string(index=False))

    print("[5] multi-zone extension (Task 2)...")
    specs = load_building_specs()
    z = len(specs["multi_zone"]["zone_names"])
    zone_occ = np.zeros((horizon_hours, z))
    # Map inhabitants to zones loosely: living/master/child.
    zone_occ[:, 0] = scenario["occupancy"][:, 0]
    zone_occ[:, 1] = scenario["occupancy"][:, 1] if scenario["occupancy"].shape[1] > 1 else 0
    zone_occ[:, 2] = scenario["occupancy"][:, 2] if scenario["occupancy"].shape[1] > 2 else 0
    zone_occ[:, 3] = 0.2 * zone_occ[:, 0]
    zone_occ[:, 4] = 0.3 * (zone_occ[:, 0] + zone_occ[:, 1])
    P_sched = mpc_df["P_HVAC_W"].to_numpy()
    mz = simulate_multi_zone(horizon_hours, weather, zone_occ, P_hvac_total_schedule=P_sched)
    mz_summary = {
        "zone_names": mz["zone_names"],
        "T_final": mz["T_zone"][-1].tolist(),
        "mean_damper": mz["damper_U"].mean(axis=0).tolist(),
        "RH_mean_final": float(mz["RH_mean"][-1]),
        "C_pol_final": float(mz["C_pol_final"]),
    }
    (RESULTS / "multizone_summary.json").write_text(
        json.dumps(mz_summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        RESULTS / "multizone_simulation.npz",
        T_zone=mz["T_zone"],
        damper_U=mz["damper_U"],
        interzone_heat_W=mz["interzone_heat_W"],
    )

    print("[6] inhabitant MAS bargain...")
    bargain = bargain_setpoints(C_pol=float(mpc_df["C_pol"].iloc[-1]), hour_of_day=19)
    (RESULTS / "mas_bargain.json").write_text(json.dumps(bargain, indent=2), encoding="utf-8")
    print("    bargain:", bargain)

    print("[7] stress harness (may take a few minutes)...")
    # Shorter window for stress to keep runtime bounded.
    stress = run_stress_battery(weather[:24], seed=2018)
    flat = []
    for k, v in stress.items():
        row = {"case": k, **v}
        flat.append(row)
    pd.DataFrame(flat).to_csv(RESULTS / "stress_harness.csv", index=False)

    print("[8] figures @ 300 DPI...")
    plot_indoor_outdoor(mpc_df, weather_df, FIGS / "fig_indoor_outdoor_temp.png")
    plot_peak_shift(bench, FIGS / "fig_peak_shift_bars.png")
    plot_radar(stress, FIGS / "fig_harness_radar.png")

    # Relative improvements vs nest_like and programmable.
    nest = bench[bench.controller == "nest_like"].iloc[0]
    prog = bench[bench.controller == "programmable"].iloc[0]
    ours = bench[bench.controller == "aerotherm_mpc"].iloc[0]
    summary = {
        "horizon_hours": horizon_hours,
        "free_float": ff,
        "vs_programmable_energy_pct": 100.0
        * (1.0 - ours["energy_kwh"] / max(prog["energy_kwh"], 1e-9)),
        "vs_nest_energy_pct": 100.0
        * (1.0 - ours["energy_kwh"] / max(nest["energy_kwh"], 1e-9)),
        "vs_programmable_cost_pct": 100.0
        * (1.0 - ours["cost_usd"] / max(prog["cost_usd"], 1e-9)),
        "vs_nest_cost_pct": 100.0 * (1.0 - ours["cost_usd"] / max(nest["cost_usd"], 1e-9)),
        "vs_nest_discomfort_pct": 100.0
        * (
            1.0
            - ours["discomfort_degree_hours"]
            / max(nest["discomfort_degree_hours"], 1e-9)
        ),
        "vs_nest_allergen_hours_pct": (
            None
            if nest["allergen_over_hours"] < 1e-9 and ours["allergen_over_hours"] < 1e-9
            else 100.0
            * (
                1.0
                - ours["allergen_over_hours"]
                / max(nest["allergen_over_hours"], 1e-9)
            )
        ),
        "bargain": bargain,
        "benchmarks": bench_rows,
    }
    (RESULTS / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in summary if k != "benchmarks"}, indent=2))
    print("[done] artifacts in results/ and paper_figures/")


if __name__ == "__main__":
    main()

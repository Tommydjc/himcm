"""
Black-box teaching-track evaluation harness (Prompt 5).

Runs Bang-Bang and prediction-matrix MPC through five stress scenarios,
exports KPI CSV, and writes two 300 DPI paper figures.

Run:
  PYTHONPATH=src python -m eval_harness.benchmark_harness
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline_thermostat.bang_bang import (
    MINUTES_PER_DAY,
    count_short_cycles,
    run_bang_bang_day,
    synthesize_summer_day_1min,
    write_weather_csv,
)
from comfort_tariffs.models import TOUTariffSchedule
from control_mpc.mpc_controller import simulate_mpc_day

PACK_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = PACK_ROOT / "results" / "harness_benchmark_results.csv"
RESULTS_JSON = PACK_ROOT / "results" / "harness_benchmark_summary.json"
FIG_RADAR = PACK_ROOT / "paper_figures" / "fig_harness_radar_benchmark.png"
FIG_TRACK = PACK_ROOT / "paper_figures" / "fig_temperature_tracking_comparison.png"

T_COMF_C = 23.0
SCENARIO_IDS = ("A", "B", "C", "D", "E")


# ---------------------------------------------------------------------------
# Scenario weather builders
# ---------------------------------------------------------------------------


def _reindex_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure contiguous minute_index / hour_frac for multi-day tables."""
    out = df.copy().reset_index(drop=True)
    n = len(out)
    minutes = np.arange(n, dtype=int)
    out["minute_index"] = minutes
    out["hour_of_day"] = (minutes // 60) % 24
    out["minute_of_hour"] = minutes % 60
    out["hour_frac"] = np.round(minutes / 60.0, 6)
    return out


def scenario_A_standard() -> pd.DataFrame:
    """Scenario A: standard summer day 23–38°C."""
    write_weather_csv()
    return synthesize_summer_day_1min(T_min_C=23.0, T_max_C=38.0)


def scenario_B_heatwave_3day(*, n_days: int = 3) -> pd.DataFrame:
    """
    Scenario B: consecutive extreme-heat days (~42°C peak).

    Default ``n_days=3`` per harness spec. Tests may pass ``n_days=1``.
    """
    days = []
    for d in range(n_days):
        day = synthesize_summer_day_1min(T_min_C=32.0, T_max_C=42.0, hour_peak=15.0)
        day["day_index"] = d
        days.append(day)
    return _reindex_minutes(pd.concat(days, ignore_index=True))


def scenario_C_thermal_shock() -> pd.DataFrame:
    """Scenario C: 15°C outdoor plunge within 2 hours starting at 12:00."""
    day = synthesize_summer_day_1min(T_min_C=23.0, T_max_C=38.0)
    t = day["hour_frac"].to_numpy()
    T = day["T_out_C"].to_numpy().copy()
    # Linear drop 15°C from 12:00→14:00, then hold offset.
    mask_drop = (t >= 12.0) & (t < 14.0)
    frac = (t[mask_drop] - 12.0) / 2.0
    T[mask_drop] = T[mask_drop] - 15.0 * frac
    T[t >= 14.0] = T[t >= 14.0] - 15.0
    day = day.copy()
    day["T_out_C"] = T
    return day


def scenario_D_sensor_noise_weather() -> pd.DataFrame:
    """Scenario D uses standard weather; noise injected at controller I/O."""
    return scenario_A_standard()


def scenario_E_infiltration_weather() -> pd.DataFrame:
    """Scenario E uses standard weather; 5× leak applied 17:00–19:00 in plant."""
    return scenario_A_standard()


SCENARIO_BUILDERS: Dict[str, Callable[[], pd.DataFrame]] = {
    "A": scenario_A_standard,
    "B": scenario_B_heatwave_3day,
    "C": scenario_C_thermal_shock,
    "D": scenario_D_sensor_noise_weather,
    "E": scenario_E_infiltration_weather,
}

SCENARIO_LABELS: Dict[str, str] = {
    "A": "Standard Summer Day",
    "B": "Heatwave 3-Day 42C",
    "C": "Thermal Shock -15C/2h",
    "D": "Sensor Noise sigma=1.5C",
    "E": "Infiltration Breach x5 17-19h",
}


# ---------------------------------------------------------------------------
# Metrics M1–M4
# ---------------------------------------------------------------------------


def discomfort_degree_hours(
    T_in: np.ndarray,
    *,
    T_comf: float = T_COMF_C,
    dt_hours: float = 1.0 / 60.0,
) -> float:
    """
    M2: ∫ (T_in − T_comf)^+ dt  [°C·h].

    Only positive (too-warm) violations count for summer cooling.
    """
    excess = np.maximum(np.asarray(T_in, dtype=float) - T_comf, 0.0)
    return float(np.sum(excess) * dt_hours)


def chatter_penalty(on_flags: np.ndarray) -> float:
    """
    High-frequency chatter: fraction of minutes that flip relative to starts.

    Penalizes rapid ON/OFF dither beyond raw start count.
    """
    on = np.asarray(on_flags, dtype=bool)
    if on.size < 2:
        return 0.0
    flips = int(np.sum(on[1:] != on[:-1]))
    starts = count_short_cycles(on)
    # Extra flips beyond 2*starts indicate chatter.
    return float(max(0, flips - 2 * starts))


def compute_metrics(
    traj: pd.DataFrame,
    *,
    controller: str,
    scenario: str,
) -> Dict[str, float]:
    """Compute M1–M3 scalars for one (controller, scenario) run."""
    tou = TOUTariffSchedule.from_json()
    bill = tou.bill_from_power_W(
        traj["P_elec_W"].to_numpy(),
        traj["hour_of_day"].astype(int).to_numpy(),
        dt_hours=1.0 / 60.0,
    )
    on = traj["cooling_on"].to_numpy() > 0.5
    starts = count_short_cycles(on)
    chatter = chatter_penalty(on)
    m2 = discomfort_degree_hours(traj["T_in_C"].to_numpy())
    # Mechanical life score: starts + 0.25 * chatter.
    m3 = float(starts) + 0.25 * chatter
    return {
        "controller": controller,
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "n_minutes": float(len(traj)),
        "M1_cost_usd": bill["cost_usd"],
        "M2_discomfort_Ch": m2,
        "M3_mechanical": m3,
        "short_cycle_starts": float(starts),
        "chatter_extra_flips": chatter,
        "energy_kwh": bill["energy_kwh"],
        "peak_energy_kwh": bill["peak_energy_kwh"],
        "mean_T_in_C": float(traj["T_in_C"].mean()),
        "max_T_in_C": float(traj["T_in_C"].max()),
    }


def attach_M4_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """
    M4: per-controller variance of normalized cost across scenarios.

    Lower variance ⇒ more robust. Written onto every row of that controller.
    """
    out = df.copy()
    out["M4_cost_variance"] = np.nan
    for ctrl, g in out.groupby("controller"):
        # Normalize each scenario cost by that controller's Scenario-A cost.
        a_cost = float(g.loc[g["scenario"] == "A", "M1_cost_usd"].iloc[0])
        if a_cost < 1e-12:
            ratios = g["M1_cost_usd"].to_numpy()
        else:
            ratios = g["M1_cost_usd"].to_numpy() / a_cost
        var = float(np.var(ratios))
        out.loc[out["controller"] == ctrl, "M4_cost_variance"] = var
    return out


# ---------------------------------------------------------------------------
# Black-box runners
# ---------------------------------------------------------------------------


def run_controller_on_scenario(
    controller: str,
    scenario: str,
    weather: pd.DataFrame,
    *,
    mpc_maxiter: int = 28,
    seed: int = 2018,
) -> Dict[str, Any]:
    """
    Black-box dispatch: Bang-Bang or MPC on a scenario weather table.

    Sensor noise (D) and infiltration (E) are injected via runner kwargs.
    """
    rng = np.random.default_rng(seed + ord(scenario))
    noise = 1.5 if scenario == "D" else 0.0
    infil = 5.0 if scenario == "E" else 1.0

    if controller == "bang_bang":
        out = run_bang_bang_day(
            weather,
            sensor_noise_sigma=noise,
            rng=rng,
            infiltration_factor=infil,
        )
    elif controller == "mpc":
        # Multi-day heatwave: keep horizon 24 but allow longer weather.
        out = simulate_mpc_day(
            weather=weather,
            maxiter=mpc_maxiter,
            sensor_noise_sigma=noise,
            rng=rng,
            infiltration_factor=infil,
        )
    else:
        raise ValueError(f"unknown controller {controller!r}")

    metrics = compute_metrics(out["trajectory"], controller=controller, scenario=scenario)
    return {"trajectory": out["trajectory"], "metrics": metrics}


def run_full_benchmark(
    *,
    scenarios: Tuple[str, ...] = SCENARIO_IDS,
    controllers: Tuple[str, ...] = ("bang_bang", "mpc"),
    mpc_maxiter: int = 28,
) -> Dict[str, Any]:
    """
    Execute the full stress suite; return metrics DataFrame + trajectories.

    Trajectories keyed as ``{controller}:{scenario}``.
    """
    rows: List[Dict[str, Any]] = []
    trajs: Dict[str, pd.DataFrame] = {}
    for sid in scenarios:
        weather = SCENARIO_BUILDERS[sid]()
        print(f"[harness] scenario {sid} ({SCENARIO_LABELS[sid]}), n={len(weather)}")
        for ctrl in controllers:
            print(f"  → {ctrl} …")
            out = run_controller_on_scenario(
                ctrl, sid, weather, mpc_maxiter=mpc_maxiter
            )
            rows.append(out["metrics"])
            trajs[f"{ctrl}:{sid}"] = out["trajectory"]
    df = attach_M4_robustness(pd.DataFrame(rows))
    return {"metrics": df, "trajectories": trajs}


# ---------------------------------------------------------------------------
# Figures (300 DPI)
# ---------------------------------------------------------------------------


def _radar_scores(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Invert M1–M4 to [0,1] scores (higher = better) for radar display.

    Uses cross-controller max per metric as the denominator.
    """
    metrics = ["M1_cost_usd", "M2_discomfort_Ch", "M3_mechanical", "M4_cost_variance"]
    # Aggregate: mean over scenarios for M1–M3; M4 already per-controller.
    scores: Dict[str, np.ndarray] = {}
    aggregates: Dict[str, List[float]] = {}
    for ctrl, g in df.groupby("controller"):
        aggregates[ctrl] = [
            float(g["M1_cost_usd"].mean()),
            float(g["M2_discomfort_Ch"].mean()),
            float(g["M3_mechanical"].mean()),
            float(g["M4_cost_variance"].iloc[0]),
        ]
    maxima = np.maximum(
        np.max(np.asarray(list(aggregates.values())), axis=0),
        1e-9,
    )
    for ctrl, vals in aggregates.items():
        # Higher score = better ⇒ 1 - val/max.
        scores[ctrl] = 1.0 - np.asarray(vals, dtype=float) / maxima
    return scores


def plot_radar(df: pd.DataFrame, out_path: Path) -> None:
    """Fig 1: dual-controller 4-axis performance radar."""
    labels = ["M1 Economy", "M2 Comfort", "M3 Lifetime", "M4 Robustness"]
    scores = _radar_scores(df)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=300, subplot_kw=dict(polar=True))
    style = {
        "bang_bang": ("#7f8c8d", "Bang-Bang"),
        "mpc": ("#2980b9", "AeroTherm-MPC"),
    }
    for ctrl, (color, name) in style.items():
        if ctrl not in scores:
            continue
        vals = scores[ctrl].tolist() + [float(scores[ctrl][0])]
        ax.plot(angles, vals, color=color, lw=2.0, label=name)
        ax.fill(angles, vals, color=color, alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Harness benchmark radar (higher = better)", pad=16, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_temperature_tracking(
    trajs: Dict[str, pd.DataFrame],
    out_path: Path,
    *,
    scenario: str = "A",
) -> None:
    """Fig 2: 24h Bang vs MPC tracking with outdoor + TOU peak shade."""
    bang = trajs.get(f"bang_bang:{scenario}")
    mpc = trajs.get(f"mpc:{scenario}")
    if bang is None or mpc is None:
        raise KeyError(f"missing trajectories for scenario {scenario}")
    # First civil day only if multi-day.
    bang = bang.iloc[:MINUTES_PER_DAY]
    mpc = mpc.iloc[:MINUTES_PER_DAY]
    t = bang["hour_frac"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), dpi=300, sharex=True)
    for ax in axes:
        ax.axvspan(14, 19, color="#e74c3c", alpha=0.12, label="TOU on-peak")
    axes[0].plot(t, bang["T_out_C"], color="#e67e22", lw=1.1, label="T_out")
    axes[0].plot(t, bang["T_in_C"], color="#7f8c8d", lw=1.2, label="Bang-Bang T_in")
    axes[0].plot(t, mpc["T_in_C"], color="#2980b9", lw=1.6, label="MPC T_in")
    axes[0].axhline(T_COMF_C, color="#27ae60", ls="--", lw=0.9, label="T_comf 23°C")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(fontsize=8, loc="upper left", ncol=2)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].set_title("Scenario A temperature tracking (Bang-Bang vs MPC)")

    axes[1].plot(t, bang["P_elec_W"] / 1000.0, color="#7f8c8d", lw=1.0, label="Bang kW")
    axes[1].plot(t, mpc["P_elec_W"] / 1000.0, color="#c0392b", lw=1.3, label="MPC kW")
    axes[1].set_ylabel("Site electric power (kW)")
    axes[1].set_xlabel("Hour of day")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def export_results(df: pd.DataFrame, trajs: Dict[str, pd.DataFrame]) -> None:
    """Write CSV/JSON + figures from a completed benchmark."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    summary = {
        "n_rows": int(len(df)),
        "scenarios": list(SCENARIO_IDS),
        "controllers": sorted(df["controller"].unique().tolist()),
        "by_controller": {},
    }
    for ctrl, g in df.groupby("controller"):
        summary["by_controller"][ctrl] = {
            "mean_M1_cost_usd": float(g["M1_cost_usd"].mean()),
            "mean_M2_discomfort_Ch": float(g["M2_discomfort_Ch"].mean()),
            "mean_M3_mechanical": float(g["M3_mechanical"].mean()),
            "M4_cost_variance": float(g["M4_cost_variance"].iloc[0]),
        }
    RESULTS_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_radar(df, FIG_RADAR)
    plot_temperature_tracking(trajs, FIG_TRACK, scenario="A")


def main() -> None:
    """Run full harness, export CSV/JSON/figures, print scoreboard."""
    # Scenario B is 3× day; keep maxiter moderate for wall-clock.
    out = run_full_benchmark(mpc_maxiter=25)
    export_results(out["metrics"], out["trajectories"])
    print(f"[harness] wrote {RESULTS_CSV}")
    print(f"[harness] wrote {RESULTS_JSON}")
    print(f"[harness] wrote {FIG_RADAR}")
    print(f"[harness] wrote {FIG_TRACK}")
    print("\n=== Harness scoreboard (mean over scenarios) ===")
    print(out["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()

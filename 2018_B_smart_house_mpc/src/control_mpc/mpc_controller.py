"""
Teaching-track rolling-horizon MPC using prediction matrices (Phase 5).

Decision: cooling ratio u_k ∈ [0, 1],  q_k = -u_k * Q_cool_max.
Predictor: X = 𝒜 x + ℬ Q + ℰ D  (ZOH, Ts = 15 min).
Plant:     building_rc RK4 at 1 min with ZOH-held q.

Same summer day as Bang-Bang → export mpc_optimal_dynamics.csv and comparison.

Run:
  PYTHONPATH=src python -m control_mpc.mpc_controller
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

from baseline_thermostat.bang_bang import (
    MINUTES_PER_DAY,
    WEATHER_1MIN,
    count_short_cycles,
    run_bang_bang_day,
    write_weather_csv,
)
from comfort_tariffs.models import (
    TOUTariffSchedule,
    calculate_pmv_ppd,
    discomfort_cost,
)
from control_mpc.prediction_matrices import build_from_discrete
from control_mpc.state_space_2r2c import (
    DEFAULT_TS_S,
    DiscreteSS,
    build_continuous_ss,
    discretize,
)
from thermal_physics.building_rc import (
    BuildingThermalEnvelope,
    ThermalDynamics2R2C,
    rk4_step,
)

PACK_ROOT = Path(__file__).resolve().parents[2]
MPC_CSV = PACK_ROOT / "results" / "mpc_optimal_dynamics.csv"
MPC_SUMMARY = PACK_ROOT / "results" / "mpc_optimal_summary.json"
COMPARE_JSON = PACK_ROOT / "results" / "bang_vs_mpc_summary.json"

HORIZON_STEPS = 24  # 24 * 15 min = 6 h
STEPS_PER_DAY = MINUTES_PER_DAY // 15  # 96
MINUTES_PER_CTRL = 15
T_SET_C = 23.0


@dataclass(frozen=True)
class MPCComfortConfig:
    """
    Comfort band and weights injected by MAS / harness.

    Attributes
    ----------
    T_set_C :
        Nominal comfort setpoint [°C].
    T_precool_C :
        Pre-peak target (late morning) [°C].
    T_peak_C :
        On-peak relaxed target [°C].
    T_min_C, T_max_C :
        Hard soft-constraint floors/ceilings for quadratic barriers.
    lambda_comf :
        Comfort weight scale (MAS may raise this).
    """

    T_set_C: float = T_SET_C
    T_precool_C: float = 21.5
    T_peak_C: float = 24.0
    T_min_C: float = 19.5
    T_max_C: float = 25.5
    lambda_comf: float = 0.05


DEFAULT_COMFORT = MPCComfortConfig()


def load_weather_1min() -> pd.DataFrame:
    """Load or create teaching summer-day 1-min weather."""
    if not WEATHER_1MIN.is_file():
        write_weather_csv()
    return pd.read_csv(WEATHER_1MIN)


def _aggregate_disturbances_blocks(
    weather: pd.DataFrame,
    n_steps: int,
) -> NDArray[np.float64]:
    """
    Aggregate 1-min weather to 15-min disturbance rows.

    Returns
    -------
    D : (n_steps, 3)
        Columns [T_out, solar, internal_gains] mean over each block.
    """
    blocks = []
    for s in range(n_steps):
        chunk = weather.iloc[s * MINUTES_PER_CTRL : (s + 1) * MINUTES_PER_CTRL]
        blocks.append(
            [
                float(chunk["T_out_C"].mean()),
                float(chunk["solar_W_per_m2"].mean()),
                float(chunk["internal_gains_W"].mean()),
            ]
        )
    return np.asarray(blocks, dtype=np.float64)


def aggregate_disturbances_15min(weather: pd.DataFrame) -> NDArray[np.float64]:
    """Aggregate a single civil day (1440 rows) to 96×3 disturbances."""
    if len(weather) != MINUTES_PER_DAY:
        raise ValueError("weather must be 1440 rows")
    return _aggregate_disturbances_blocks(weather, STEPS_PER_DAY)


def hour_of_ctrl_step(step: int) -> int:
    """Civil hour for control step index 0..95."""
    return int((step * MINUTES_PER_CTRL) // 60) % 24


def mpc_stage_cost(
    U_ratio: NDArray[np.floating],
    *,
    x0: NDArray[np.floating],
    D_h: NDArray[np.floating],
    hours: Sequence[int],
    calA: NDArray[np.float64],
    calB: NDArray[np.float64],
    calE: NDArray[np.float64],
    env: BuildingThermalEnvelope,
    tou: TOUTariffSchedule,
    comfort: MPCComfortConfig = DEFAULT_COMFORT,
    lambda_smooth: float = 0.008,
    u_prev: float = 0.0,
) -> float:
    """
    Scalar cost over horizon using linear prediction of T_in.

    U_ratio ∈ [0,1]^H → Q = -U_ratio * Q_cool_max.

    Peak shedding: on-peak hours get an extra u² penalty and a warmer
    comfort target. Pre-peak hours target deeper cool (MAS may raise floor).
    """
    H = len(U_ratio)
    Q = (-U_ratio * env.Q_cool_max_W).reshape(H, 1)
    X = calA @ x0.reshape(-1) + calB @ Q.reshape(-1) + calE @ D_h.reshape(-1)
    n = 2
    J = 0.0
    u_last = float(u_prev)
    dt_h = DEFAULT_TS_S / 3600.0
    lambda_comf = float(comfort.lambda_comf)
    for k in range(H):
        T_in = float(X[k * n])
        u = float(U_ratio[k])
        q = -u * env.Q_cool_max_W
        T_out = float(D_h[k, 0])
        hod = int(hours[k])
        P_elec = env.electric_power_W(q, T_out)
        rate = tou.rate_usd_per_kwh(hod)
        # Emphasize TOU: square the rate ratio vs off-peak to sharpen peak.
        J += (rate / 0.12) * rate * (P_elec / 1000.0) * dt_h

        if tou.is_on_peak(hod):
            T_target = float(comfort.T_peak_C)
            J += 0.35 * u**2  # shed compressor on peak
            w_comf = 0.6 * lambda_comf
        elif hod in (10, 11, 12, 13):
            # Respect senior floor: never pre-cool below T_min_C.
            T_target = max(float(comfort.T_precool_C), float(comfort.T_min_C))
            w_comf = 1.4 * lambda_comf
        else:
            T_target = float(comfort.T_set_C)
            w_comf = lambda_comf

        J += w_comf * (T_in - T_target) ** 2
        if T_in > float(comfort.T_max_C):
            J += 3.0 * (T_in - float(comfort.T_max_C)) ** 2
        if T_in < float(comfort.T_min_C):
            J += 2.0 * (float(comfort.T_min_C) - T_in) ** 2
        J += lambda_smooth * (u - u_last) ** 2
        u_last = u
    return float(J)


def solve_mpc_step(
    x0: NDArray[np.floating],
    D_full: NDArray[np.floating],
    step: int,
    *,
    dss: DiscreteSS,
    env: BuildingThermalEnvelope,
    tou: TOUTariffSchedule,
    H: int = HORIZON_STEPS,
    u_prev: float = 0.0,
    u_warm: Optional[NDArray[np.floating]] = None,
    maxiter: int = 40,
    comfort: MPCComfortConfig = DEFAULT_COMFORT,
) -> Dict[str, Any]:
    """Solve one receding window; return first-step ratio and diagnostics."""
    n_day = D_full.shape[0]
    H_eff = min(H, n_day - step)
    if H_eff < 1:
        raise ValueError("no remaining horizon")
    D_h = D_full[step : step + H_eff]
    hours = [hour_of_ctrl_step(step + k) for k in range(H_eff)]
    calA, calB, calE = build_from_discrete(dss, H_eff)

    if u_warm is None or len(u_warm) < H_eff:
        # Warm start: pre-cool hard late morning; idle on peak.
        u0 = np.zeros(H_eff, dtype=np.float64)
        for k in range(H_eff):
            h = hours[k]
            if h in (10, 11, 12, 13):
                u0[k] = 0.85
            elif 14 <= h < 19:
                u0[k] = 0.05
            else:
                u0[k] = 0.30
    else:
        u0 = np.asarray(u_warm[:H_eff], dtype=np.float64).copy()

    bounds = [(0.0, 1.0)] * H_eff
    res = minimize(
        lambda u: mpc_stage_cost(
            u,
            x0=np.asarray(x0, dtype=np.float64),
            D_h=D_h,
            hours=hours,
            calA=calA,
            calB=calB,
            calE=calE,
            env=env,
            tou=tou,
            comfort=comfort,
            u_prev=u_prev,
        ),
        u0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-6},
    )
    U = np.clip(res.x, 0.0, 1.0)
    return {
        "u0": float(U[0]),
        "U": U,
        "success": bool(res.success),
        "cost": float(res.fun),
        "nit": int(getattr(res, "nit", 0)),
        "H": H_eff,
    }


def plant_advance_15min(
    model: ThermalDynamics2R2C,
    T_in: float,
    T_wall: float,
    weather_chunk: pd.DataFrame,
    q_hvac: float,
    *,
    infiltration_factor: float = 1.0,
    infiltration_hours: Tuple[int, int] = (17, 19),
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Advance plant 15×1 min with held HVAC power; return state + minute rows.

    When ``infiltration_factor`` > 1 during ``infiltration_hours``, add an
    extra conductance (factor−1)/R_win · (T_out−T_in) as internal gain (breach).
    """
    env = model.env
    tou = TOUTariffSchedule.from_json()
    rows: List[Dict[str, float]] = []
    q = env.clip_q_hvac(q_hvac)
    h0, h1 = infiltration_hours
    for _, w in weather_chunk.iterrows():
        T_out = float(w["T_out_C"])
        solar = float(w["solar_W_per_m2"])
        gains = float(w["internal_gains_W"])
        hod = int(w["hour_of_day"])
        if infiltration_factor > 1.0 and h0 <= hod < h1:
            # Extra leak ≈ (factor-1) × window conductance × ΔT.
            gains = gains + (infiltration_factor - 1.0) * (T_out - T_in) / env.R_win_C_per_W
        P_elec = env.electric_power_W(q, T_out)
        rate = tou.rate_usd_per_kwh(hod)
        cost = P_elec * (env.dt_s / 3600.0) / 1000.0 * rate
        pmv, ppd = calculate_pmv_ppd(T_in, rh=50.0)
        rows.append(
            {
                "minute_index": float(w["minute_index"]),
                "hour_of_day": float(hod),
                "hour_frac": float(w["hour_frac"]),
                "T_out_C": T_out,
                "solar_W_per_m2": solar,
                "internal_gains_W": gains,
                "T_in_C": T_in,
                "T_wall_C": T_wall,
                "u_ratio": float(abs(q) / env.Q_cool_max_W) if q < 0 else 0.0,
                "cooling_on": 1.0 if q < -1.0 else 0.0,
                "q_hvac_W": q,
                "P_elec_W": P_elec,
                "rate_usd_per_kwh": rate,
                "cost_usd": cost,
                "pmv": pmv,
                "ppd": ppd,
                "discomfort_cost": discomfort_cost(pmv),
                "on_peak": 1.0 if tou.is_on_peak(hod) else 0.0,
            }
        )
        T_in, T_wall = rk4_step(
            model,
            T_in,
            T_wall,
            T_out=T_out,
            solar_rad=solar,
            q_hvac=q,
            internal_gains=gains,
            dt_s=env.dt_s,
        )
    return T_in, T_wall, rows


def simulate_mpc_day(
    *,
    weather: Optional[pd.DataFrame] = None,
    T_in0: float = 23.5,
    T_wall0: float = 25.0,
    H: int = HORIZON_STEPS,
    maxiter: int = 35,
    logger: Any = None,
    comfort: MPCComfortConfig = DEFAULT_COMFORT,
    sensor_noise_sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    infiltration_hours: Tuple[int, int] = (17, 19),
    infiltration_factor: float = 1.0,
) -> Dict[str, Any]:
    """
    Closed-loop receding-horizon day on teaching weather.

    Parameters
    ----------
    logger :
        Optional ``RunLogger``; each control step emits ``mpc_step``.
    comfort :
        MAS / harness comfort band and λ_comf.
    sensor_noise_sigma :
        Gaussian σ [°C] on measured T_in fed to the optimizer (plant stays true).
    infiltration_factor :
        Scale (>1) of extra window conductance during ``infiltration_hours``.
    """
    weather = weather if weather is not None else load_weather_1min()
    env = BuildingThermalEnvelope.from_json()
    model = ThermalDynamics2R2C(env)
    tou = TOUTariffSchedule.from_json()
    css = build_continuous_ss(env)
    dss = discretize(css, DEFAULT_TS_S, method="zoh")
    # Support multi-day weather for harness (N * 1440 minutes).
    n_min = len(weather)
    if n_min % MINUTES_PER_CTRL != 0:
        raise ValueError("weather length must be divisible by 15 min")
    n_steps = n_min // MINUTES_PER_CTRL
    D_full = _aggregate_disturbances_blocks(weather, n_steps)
    gen = rng if rng is not None else np.random.default_rng(2018)

    T_in, T_wall = float(T_in0), float(T_wall0)
    all_rows: List[Dict[str, float]] = []
    u_prev = 0.3
    U_warm: Optional[NDArray[np.float64]] = None
    schedule: List[Dict[str, float]] = []

    for step in range(n_steps):
        noise = float(gen.normal(0.0, sensor_noise_sigma)) if sensor_noise_sigma > 0 else 0.0
        x0 = np.array([T_in + noise, T_wall], dtype=np.float64)
        hod = hour_of_ctrl_step(step)
        sol = solve_mpc_step(
            x0,
            D_full,
            step,
            dss=dss,
            env=env,
            tou=tou,
            H=H,
            u_prev=u_prev,
            u_warm=U_warm,
            maxiter=maxiter,
            comfort=comfort,
        )
        u0 = sol["u0"]
        if logger is not None:
            logger.mpc_step(
                ctrl_step=step,
                hour_of_day=hod,
                u0=u0,
                cost=float(sol["cost"]),
                nit=int(sol["nit"]),
                success=bool(sol["success"]),
                rate_usd_per_kwh=float(tou.rate_usd_per_kwh(hod)),
                T_in_C=float(T_in),
            )
        q = -u0 * env.Q_cool_max_W
        chunk = weather.iloc[step * MINUTES_PER_CTRL : (step + 1) * MINUTES_PER_CTRL]
        T_in, T_wall, rows = plant_advance_15min(
            model,
            T_in,
            T_wall,
            chunk,
            q,
            infiltration_factor=infiltration_factor,
            infiltration_hours=infiltration_hours,
        )
        all_rows.extend(rows)
        schedule.append(
            {
                "ctrl_step": float(step),
                "hour_of_day": float(hod),
                "u_ratio": u0,
                "q_hvac_W": q,
                "mpc_cost": float(sol["cost"]),
                "mpc_success": 1.0 if sol["success"] else 0.0,
                "nit": float(sol["nit"]),
            }
        )
        u_prev = u0
        # Shift warm start.
        U_tail = sol["U"][1:] if len(sol["U"]) > 1 else np.array([u0])
        U_warm = np.concatenate([U_tail, U_tail[-1:]])

    traj = pd.DataFrame(all_rows)
    bill = tou.bill_from_power_W(
        traj["P_elec_W"].to_numpy(),
        traj["hour_of_day"].astype(int).to_numpy(),
        dt_hours=env.dt_s / 3600.0,
    )
    on_flags = traj["cooling_on"].to_numpy() > 0.5
    metrics = {
        "controller": "aerotherm_mpc",
        "energy_kwh": bill["energy_kwh"],
        "cost_usd": bill["cost_usd"],
        "peak_energy_kwh": bill["peak_energy_kwh"],
        "peak_cost_usd": bill["peak_cost_usd"],
        "peak_cost_share": bill["peak_cost_share"],
        "short_cycle_starts": count_short_cycles(on_flags),
        "cooling_minutes": int(np.sum(on_flags)),
        "peak_cooling_minutes": int(
            np.sum(on_flags & (traj["on_peak"].to_numpy() > 0.5))
        ),
        "mean_T_in_C": float(traj["T_in_C"].mean()),
        "max_T_in_C": float(traj["T_in_C"].max()),
        "min_T_in_C": float(traj["T_in_C"].min()),
        "mean_abs_pmv": float(np.mean(np.abs(traj["pmv"].to_numpy()))),
        "mean_discomfort_cost": float(traj["discomfort_cost"].mean()),
        "mean_u_peak": float(
            traj.loc[traj["on_peak"] > 0.5, "u_ratio"].mean()
            if (traj["on_peak"] > 0.5).any()
            else 0.0
        ),
        "mean_u_prepeak": float(
            traj.loc[traj["hour_of_day"].isin([11, 12, 13]), "u_ratio"].mean()
            if traj["hour_of_day"].isin([11, 12, 13]).any()
            else 0.0
        ),
        "H": H,
        "Ts_s": DEFAULT_TS_S,
    }
    return {
        "trajectory": traj,
        "schedule": pd.DataFrame(schedule),
        "metrics": metrics,
    }


def compare_to_bang(mpc_metrics: Dict[str, Any], bang_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Relative savings MPC vs Bang-Bang (positive = MPC better)."""

    def pct(new: float, old: float) -> Optional[float]:
        if abs(old) < 1e-12:
            return None
        return float(100.0 * (1.0 - new / old))

    return {
        "bang": bang_metrics,
        "mpc": mpc_metrics,
        "cost_savings_pct": pct(mpc_metrics["cost_usd"], bang_metrics["cost_usd"]),
        "energy_savings_pct": pct(mpc_metrics["energy_kwh"], bang_metrics["energy_kwh"]),
        "peak_energy_savings_pct": pct(
            mpc_metrics["peak_energy_kwh"], bang_metrics["peak_energy_kwh"]
        ),
        "peak_cost_savings_pct": pct(
            mpc_metrics["peak_cost_usd"], bang_metrics["peak_cost_usd"]
        ),
        "short_cycle_reduction": int(
            bang_metrics["short_cycle_starts"] - mpc_metrics["short_cycle_starts"]
        ),
    }


def plot_bang_vs_mpc(
    bang_traj: pd.DataFrame,
    mpc_traj: pd.DataFrame,
    out_path: Path,
) -> None:
    """300 DPI T_in and electric power comparison with peak shading."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = bang_traj["hour_frac"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), dpi=300, sharex=True)
    for ax in axes:
        ax.axvspan(14, 19, color="#e74c3c", alpha=0.12, label="TOU on-peak")
    axes[0].plot(t, bang_traj["T_in_C"], color="#7f8c8d", lw=1.2, label="Bang-Bang T_in")
    axes[0].plot(t, mpc_traj["T_in_C"], color="#2980b9", lw=1.6, label="MPC T_in")
    axes[0].plot(t, bang_traj["T_out_C"], color="#e67e22", lw=1.0, alpha=0.8, label="T_out")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].set_title("Teaching day: Bang-Bang vs prediction-matrix MPC")

    axes[1].plot(t, bang_traj["P_elec_W"] / 1000.0, color="#7f8c8d", lw=1.0, label="Bang-Bang kW")
    axes[1].plot(t, mpc_traj["P_elec_W"] / 1000.0, color="#c0392b", lw=1.3, label="MPC kW")
    axes[1].set_ylabel("Site electric power (kW)")
    axes[1].set_xlabel("Hour of day")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Run MPC day, compare to Bang-Bang, export locked artifacts."""
    from eval_harness.run_logger import new_session

    log = new_session(component="mpc_controller", track="teaching", H=HORIZON_STEPS)
    print("[mpc] ensuring weather + Bang-Bang baseline...")
    write_weather_csv()
    bang = run_bang_bang_day()
    log.bang_day(bang["metrics"])
    print("[mpc] rolling-horizon MPC (H=24, Ts=15min)...")
    mpc = simulate_mpc_day(logger=log)
    MPC_CSV.parent.mkdir(parents=True, exist_ok=True)
    mpc["trajectory"].to_csv(MPC_CSV, index=False)
    MPC_SUMMARY.write_text(json.dumps(mpc["metrics"], indent=2), encoding="utf-8")
    cmp = compare_to_bang(mpc["metrics"], bang["metrics"])
    COMPARE_JSON.write_text(json.dumps(cmp, indent=2), encoding="utf-8")

    fig_path = PACK_ROOT / "paper_figures" / "fig_bang_vs_mpc_day.png"
    plot_bang_vs_mpc(bang["trajectory"], mpc["trajectory"], fig_path)
    log.compare_batch(cmp)
    log.export(
        paths={
            "mpc_dynamics_csv": str(MPC_CSV),
            "mpc_summary_json": str(MPC_SUMMARY),
            "compare_json": str(COMPARE_JSON),
            "fig_bang_vs_mpc": str(fig_path),
        }
    )

    print(f"[mpc] wrote {MPC_CSV}")
    print(f"[mpc] wrote {MPC_SUMMARY}")
    print(f"[mpc] wrote {COMPARE_JSON}")
    print(f"[mpc] wrote {fig_path}")
    print(f"[mpc] jsonl {log.path}")
    print("\n=== Bang-Bang vs AeroTherm-MPC (same day) ===")
    print(
        f"  Bang cost ${bang['metrics']['cost_usd']:.3f}  "
        f"energy {bang['metrics']['energy_kwh']:.3f} kWh  "
        f"peak share {100 * bang['metrics']['peak_cost_share']:.1f}%  "
        f"starts {bang['metrics']['short_cycle_starts']}"
    )
    print(
        f"  MPC  cost ${mpc['metrics']['cost_usd']:.3f}  "
        f"energy {mpc['metrics']['energy_kwh']:.3f} kWh  "
        f"peak share {100 * mpc['metrics']['peak_cost_share']:.1f}%  "
        f"starts {mpc['metrics']['short_cycle_starts']}"
    )
    if cmp["cost_savings_pct"] is not None:
        print(f"  Cost savings:   {cmp['cost_savings_pct']:.1f}%")
    if cmp["energy_savings_pct"] is not None:
        print(f"  Energy savings: {cmp['energy_savings_pct']:.1f}%")
    if cmp["peak_energy_savings_pct"] is not None:
        print(f"  Peak energy↓:   {cmp['peak_energy_savings_pct']:.1f}%")
    print(
        f"  Prepeak mean u: {mpc['metrics']['mean_u_prepeak']:.3f}  "
        f"Peak mean u: {mpc['metrics']['mean_u_peak']:.3f}"
    )
    print(
        f"  T_in MPC mean/max: {mpc['metrics']['mean_T_in_C']:.2f} / "
        f"{mpc['metrics']['max_T_in_C']:.2f}"
    )


if __name__ == "__main__":
    main()

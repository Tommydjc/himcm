"""
Teaching-track Bang-Bang hysteresis thermostat (Phase 3).

Exposes the TOU peak-period cost spike on a 1-minute summer hot day.
Uses ``building_rc`` + ``tou_tariffs_v2`` (teaching track only).

Run:
  PYTHONPATH=src python -m baseline_thermostat.bang_bang
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from comfort_tariffs.models import (
    TOUTariffSchedule,
    calculate_pmv_ppd,
    discomfort_cost,
)
from thermal_physics.building_rc import (
    BuildingThermalEnvelope,
    ThermalDynamics2R2C,
    rk4_step,
)

PACK_ROOT = Path(__file__).resolve().parents[2]
WEATHER_1MIN = PACK_ROOT / "data" / "raw" / "weather_summer_day_1min.csv"
RESULTS_CSV = PACK_ROOT / "results" / "baseline_bang_bang_dynamics.csv"
SUMMARY_JSON = PACK_ROOT / "results" / "baseline_bang_bang_summary.json"

T_ON_C = 24.0   # start cooling when above
T_OFF_C = 22.0  # stop cooling when below
Q_COOL_MAX_W = 5000.0
MINUTES_PER_DAY = 24 * 60


def synthesize_summer_day_1min(
    *,
    T_min_C: float = 23.0,
    T_max_C: float = 38.0,
    hour_peak: float = 15.0,
    I_max: float = 750.0,
) -> pd.DataFrame:
    """
    Build a 1440-row summer outdoor forcing table.

    T_out peaks at ``hour_peak`` (default 15:00) and bottoms ~12 h earlier.
    """
    minutes = np.arange(MINUTES_PER_DAY, dtype=int)
    hour_frac = minutes / 60.0
    T_mean = 0.5 * (T_min_C + T_max_C)
    amp = 0.5 * (T_max_C - T_min_C)
    T_out = T_mean + amp * np.cos(2.0 * np.pi * (hour_frac - hour_peak) / 24.0)
    # Solar: daytime lobe 06:00–18:00.
    solar = np.clip(
        I_max * np.sin(np.pi * np.maximum(hour_frac - 6.0, 0.0) / 12.0),
        0.0,
        None,
    )
    solar = np.where((hour_frac >= 6.0) & (hour_frac <= 18.0), solar, 0.0)
    # Mild internal gains when "home" evenings/mornings.
    hod = minutes // 60
    internal = np.where(
        (hod <= 7) | (hod >= 17),
        350.0,
        120.0,
    ).astype(float)

    df = pd.DataFrame(
        {
            "minute_index": minutes,
            "hour_of_day": hod,
            "minute_of_hour": minutes % 60,
            "hour_frac": np.round(hour_frac, 6),
            "T_out_C": np.round(T_out, 4),
            "solar_W_per_m2": np.round(solar, 3),
            "internal_gains_W": internal,
        }
    )
    return df


def write_weather_csv(path: Path = WEATHER_1MIN) -> Path:
    """Write (or overwrite) the teaching summer-day 1-min weather file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = synthesize_summer_day_1min()
    df.to_csv(path, index=False)
    return path


@dataclass
class BangBangController:
    """
    Deadband cooling thermostat with hysteresis.

    Rules
    -----
    - If T_in > T_on  → cooling ON  (q = -Q_max)
    - If T_in < T_off → cooling OFF (q = 0)
    - Else keep previous minute state.
    """

    T_on_C: float = T_ON_C
    T_off_C: float = T_OFF_C
    Q_cool_max_W: float = Q_COOL_MAX_W
    cooling_on: bool = False

    def step(self, T_in: float) -> Tuple[float, bool]:
        """Return (q_hvac_W, cooling_on) for the current minute."""
        if T_in > self.T_on_C:
            self.cooling_on = True
        elif T_in < self.T_off_C:
            self.cooling_on = False
        q = -self.Q_cool_max_W if self.cooling_on else 0.0
        return float(q), bool(self.cooling_on)


def count_short_cycles(on_flags: np.ndarray) -> int:
    """
    Count compressor start events (OFF→ON transitions).

    High start counts indicate short-cycling wear.
    """
    on = np.asarray(on_flags, dtype=bool)
    if on.size == 0:
        return 0
    starts = int(np.sum((~on[:-1]) & on[1:]))
    if bool(on[0]):
        starts += 1
    return starts


def run_bang_bang_day(
    weather: Optional[pd.DataFrame] = None,
    *,
    T_in0: float = 23.5,
    T_wall0: float = 25.0,
    envelope: Optional[BuildingThermalEnvelope] = None,
    tou: Optional[TOUTariffSchedule] = None,
    sensor_noise_sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    infiltration_hours: Tuple[int, int] = (17, 19),
    infiltration_factor: float = 1.0,
) -> Dict[str, Any]:
    """
    Simulate one or more civil days at Δt = 60 s under Bang-Bang cooling.

    Length may be ``k * 1440`` minutes for multi-day harness scenarios.

    Returns
    -------
    dict
        ``trajectory`` DataFrame and ``metrics`` summary dict.
    """
    if weather is None:
        if WEATHER_1MIN.is_file():
            weather = pd.read_csv(WEATHER_1MIN)
        else:
            write_weather_csv()
            weather = pd.read_csv(WEATHER_1MIN)
    if len(weather) < 60 or len(weather) % 60 != 0:
        raise ValueError(f"weather length must be a positive multiple of 60, got {len(weather)}")

    env = envelope or BuildingThermalEnvelope.from_json()
    model = ThermalDynamics2R2C(env)
    tariff = tou or TOUTariffSchedule.from_json()
    ctrl = BangBangController()
    dt = float(env.dt_s)
    gen = rng if rng is not None else np.random.default_rng(2018)
    h0, h1 = infiltration_hours

    T_in = float(T_in0)
    T_wall = float(T_wall0)
    rows: List[Dict[str, float]] = []
    on_flags: List[bool] = []

    for _, w in weather.iterrows():
        noise = float(gen.normal(0.0, sensor_noise_sigma)) if sensor_noise_sigma > 0 else 0.0
        q, on = ctrl.step(T_in + noise)
        on_flags.append(on)
        T_out = float(w["T_out_C"])
        solar = float(w["solar_W_per_m2"])
        gains = float(w["internal_gains_W"])
        hod = int(w["hour_of_day"])
        if infiltration_factor > 1.0 and h0 <= hod < h1:
            gains = gains + (infiltration_factor - 1.0) * (T_out - T_in) / env.R_win_C_per_W
        P_elec = env.electric_power_W(q, T_out)
        rate = tariff.rate_usd_per_kwh(hod)
        dt_h = dt / 3600.0
        cost_step = P_elec * dt_h / 1000.0 * rate
        pmv, ppd = calculate_pmv_ppd(T_in, rh=50.0)
        c_dis = discomfort_cost(pmv)

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
                "cooling_on": 1.0 if on else 0.0,
                "q_hvac_W": q,
                "P_elec_W": P_elec,
                "rate_usd_per_kwh": rate,
                "cost_usd": cost_step,
                "pmv": pmv,
                "ppd": ppd,
                "discomfort_cost": c_dis,
                "on_peak": 1.0 if tariff.is_on_peak(hod) else 0.0,
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
            dt_s=dt,
        )

    traj = pd.DataFrame(rows)
    bill = tariff.bill_from_power_W(
        traj["P_elec_W"].to_numpy(),
        traj["hour_of_day"].astype(int).to_numpy(),
        dt_hours=dt / 3600.0,
    )
    short_cycles = count_short_cycles(np.asarray(on_flags))
    cooling_minutes = int(np.sum(traj["cooling_on"].to_numpy() > 0.5))
    peak_minutes_on = int(
        np.sum(
            (traj["cooling_on"].to_numpy() > 0.5)
            & (traj["on_peak"].to_numpy() > 0.5)
        )
    )
    metrics = {
        "controller": "bang_bang",
        "energy_kwh": bill["energy_kwh"],
        "cost_usd": bill["cost_usd"],
        "peak_energy_kwh": bill["peak_energy_kwh"],
        "peak_cost_usd": bill["peak_cost_usd"],
        "peak_cost_share": bill["peak_cost_share"],
        "short_cycle_starts": short_cycles,
        "cooling_minutes": cooling_minutes,
        "peak_cooling_minutes": peak_minutes_on,
        "mean_T_in_C": float(traj["T_in_C"].mean()),
        "max_T_in_C": float(traj["T_in_C"].max()),
        "min_T_in_C": float(traj["T_in_C"].min()),
        "mean_abs_pmv": float(np.mean(np.abs(traj["pmv"].to_numpy()))),
        "mean_discomfort_cost": float(traj["discomfort_cost"].mean()),
        "T_on_C": T_ON_C,
        "T_off_C": T_OFF_C,
        "Q_cool_max_W": Q_COOL_MAX_W,
    }
    return {"trajectory": traj, "metrics": metrics}


def export_results(
    out: Dict[str, Any],
    *,
    csv_path: Path = RESULTS_CSV,
    summary_path: Path = SUMMARY_JSON,
) -> None:
    """Write minute trajectory CSV and JSON metrics."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out["trajectory"].to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(out["metrics"], indent=2), encoding="utf-8")


def _print_metrics(metrics: Dict[str, Any]) -> None:
    """Console table emphasizing peak-cost waste."""
    rows = [
        ("Daily energy (kWh)", f"{metrics['energy_kwh']:.3f}"),
        ("Daily cost (USD)", f"${metrics['cost_usd']:.3f}"),
        ("On-peak energy (kWh)", f"{metrics['peak_energy_kwh']:.3f}"),
        ("On-peak cost (USD)", f"${metrics['peak_cost_usd']:.3f}"),
        ("On-peak cost share", f"{100.0 * metrics['peak_cost_share']:.1f}%"),
        ("Compressor starts", f"{metrics['short_cycle_starts']}"),
        ("Cooling minutes", f"{metrics['cooling_minutes']} / {MINUTES_PER_DAY}"),
        ("Peak cooling minutes", f"{metrics['peak_cooling_minutes']}"),
        ("Mean / max T_in (°C)", f"{metrics['mean_T_in_C']:.2f} / {metrics['max_T_in_C']:.2f}"),
        ("Mean |PMV|", f"{metrics['mean_abs_pmv']:.3f}"),
    ]
    print("\n=== Bang-Bang summer-day baseline ===")
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k:<{width}}  {v}")
    share = 100.0 * float(metrics["peak_cost_share"])
    if share >= 60.0:
        print(
            f"\n  !! Peak-period cost share {share:.1f}% ≥ 60% — "
            "classic Bang-Bang TOU waste exposed."
        )
    else:
        print(
            f"\n  Note: peak cost share is {share:.1f}% "
            "(report measured value; do not invent ≥60%)."
        )


def main() -> None:
    """Generate weather, run 24h Bang-Bang, export CSV/JSON, print KPIs."""
    from eval_harness.run_logger import new_session

    log = new_session(component="bang_bang", track="teaching")
    weather_path = write_weather_csv()
    print(f"[bang_bang] wrote weather {weather_path}")
    out = run_bang_bang_day()
    export_results(out)
    log.bang_day(out["metrics"])
    log.export(
        paths={
            "dynamics_csv": str(RESULTS_CSV),
            "summary_json": str(SUMMARY_JSON),
            "weather_csv": str(weather_path),
        }
    )
    print(f"[bang_bang] wrote {RESULTS_CSV}")
    print(f"[bang_bang] wrote {SUMMARY_JSON}")
    print(f"[bang_bang] jsonl {log.path}")
    _print_metrics(out["metrics"])
    assert len(out["trajectory"]) == MINUTES_PER_DAY


if __name__ == "__main__":
    main()

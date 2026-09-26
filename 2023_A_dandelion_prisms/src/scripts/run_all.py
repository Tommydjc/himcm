#!/usr/bin/env python3
"""全量仿真：三气候基线、干旱、割草帕累托、KPP 验算、影响因子。

运行::

    PYTHONPATH=. python -m src.scripts.run_all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from src.biology.self_thinning import carrying_capacity
from src.decision.impact import cover_to_spread_score, score_species
from src.decision.pareto import PolicyPoint, mowing_cost, nondominated
from src.params import load_biophysics, load_climates, load_impact_contract
from src.scripts.forcing_weeks import load_weekly_forcing
from src.scripts.generate_forcing import write_all_climates
from src.simulation.engine import run_year
from src.theory.fisher_kpp import (
    diffusion_from_kernel,
    intrinsic_rate,
    solve_fisher_1d,
    traveling_wave_speed,
)

RESULTS: Path = PACK_ROOT / "results"
SNAP_DIR: Path = RESULTS / "snapshots"
CHECKPOINT_MONTHS: tuple[int, ...] = (1, 2, 3, 6, 12)


def _ensure_forcing() -> None:
    weather = PACK_ROOT / "data" / "raw" / "weather" / "temperate_hourly.csv"
    if not weather.is_file():
        write_all_climates()


def _month_end_rows(weekly: list[dict[str, float]], climate: str, scenario: str) -> list[dict[str, object]]:
    """取每月最后一周作为月末指标。"""
    frame = pd.DataFrame(weekly)
    rows: list[dict[str, object]] = []
    for month in range(1, 13):
        block = frame[frame["month"] == month]
        if block.empty:
            continue
        last = block.iloc[-1]
        rows.append(
            {
                "climate": climate,
                "scenario": scenario,
                "month": int(month),
                "week": int(last["week"]),
                "n_plants": float(last["n_plants"]),
                "n_adult": float(last["n_adult"]),
                "n_rosette": float(last["n_rosette"]),
                "n_seedling": float(last["n_seedling"]),
                "n_seed": float(last["n_seed"]),
                "cover_frac": float(last["cover_frac"]),
                "front_m": float(last["front_m"]),
                "max_canopy": float(last["max_canopy"]),
                "produced_seeds": float(last["produced_seeds"]),
            }
        )
    return rows


def run_baselines() -> pd.DataFrame:
    """三气候无干预 52 周。"""
    params = load_biophysics()
    climates = load_climates()
    monthly_rows: list[dict[str, object]] = []
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in climates.items():
        forcing = load_weekly_forcing(name, spec)
        _state, weekly, snaps = run_year(forcing, params)
        monthly_rows.extend(_month_end_rows(weekly, name, "baseline"))
        pd.DataFrame(weekly).to_csv(RESULTS / f"weekly_{name}_baseline.csv", index=False)
        for week, field in snaps.items():
            np.save(SNAP_DIR / f"{name}_week{week:02d}.npy", field)
    table = pd.DataFrame(monthly_rows)
    table.to_csv(RESULTS / "monthly_metrics.csv", index=False)
    return table


def run_drought() -> pd.DataFrame:
    """温带 SMI×0.5 对照。"""
    params = load_biophysics()
    climates = load_climates()
    spec = climates["temperate"]
    forcing = load_weekly_forcing("temperate", spec, smi_scale=0.5)
    _state, weekly, _snaps = run_year(forcing, params)
    rows = _month_end_rows(weekly, "temperate", "drought_smi_x0.5")
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "drought_monthly.csv", index=False)
    return table


def run_mowing_sweep() -> pd.DataFrame:
    """温带割草间隔 × 强度网格。"""
    params = load_biophysics()
    spec = load_climates()["temperate"]
    intervals = (0, 1, 2, 4, 6, 8, 13, 26)
    etas = (0.4, 0.7, 1.0)
    points: list[PolicyPoint] = []
    raw_rows: list[dict[str, object]] = []
    for interval in intervals:
        eta_list = (0.0,) if interval == 0 else etas
        for eta in eta_list:
            forcing = load_weekly_forcing("temperate", spec, mow_interval=interval, mow_eta=eta)
            _state, weekly, _snaps = run_year(forcing, params)
            last = weekly[-1]
            n_mows = sum(1 for item in forcing if item.mow_eta > 0.0)
            cost = mowing_cost(n_mows, eta)
            point = PolicyPoint(
                interval_weeks=interval,
                eta=eta,
                n_mows=n_mows,
                cover=float(last["cover_frac"]),
                cost=cost,
                n_plants=float(last["n_plants"]),
            )
            points.append(point)
            raw_rows.append(
                {
                    "interval_weeks": interval,
                    "eta": eta,
                    "n_mows": n_mows,
                    "cover_frac": point.cover,
                    "n_plants": point.n_plants,
                    "cost": cost,
                    "on_front": False,
                }
            )
    front = set((p.interval_weeks, p.eta) for p in nondominated(points))
    for row in raw_rows:
        row["on_front"] = (int(row["interval_weeks"]), float(row["eta"])) in front
    table = pd.DataFrame(raw_rows).sort_values(["cost", "cover_frac"])
    table.to_csv(RESULTS / "bioeconomic_pareto.csv", index=False)
    return table


def run_kpp_check() -> pd.DataFrame:
    """常气候矩匹配 + 1D PDE 波速。"""
    params = load_biophysics()
    temp_c = 18.0
    smi = 0.55
    wind = 3.5
    wind_from = 210.0
    tau = 1.0
    D = diffusion_from_kernel(params, wind, wind_from, tau)
    r = intrinsic_rate(params, temp_c, smi, blooming=True, tau=tau)
    k_eff = carrying_capacity(smi, params)
    c_star = traveling_wave_speed(max(r, 0.0), D)
    _x, _u, c_num = solve_fisher_1d(D, max(r, 1.0e-6), k_eff, length_m=400.0, t_end=40.0)
    row = {
        "D_m2_per_week": D,
        "r_per_week": r,
        "K_eff": k_eff,
        "c_star_m_per_week": c_star,
        "c_num_m_per_week": c_num,
        "rel_err": abs(c_num - c_star) / max(c_star, 1.0e-9),
    }
    table = pd.DataFrame([row])
    table.to_csv(RESULTS / "kpp_check.csv", index=False)
    return table


def run_impact(monthly: pd.DataFrame) -> pd.DataFrame:
    """官方 Req 2：三物种 SAW。"""
    contract = load_impact_contract()
    temperate = monthly[(monthly["climate"] == "temperate") & (monthly["scenario"] == "baseline")]
    cover12 = float(temperate.loc[temperate["month"] == 12, "cover_frac"].iloc[0])
    model_spread = cover_to_spread_score(cover12)
    prepared: list[dict[str, object]] = []
    meta: list[dict[str, object]] = []
    for spec in contract["species"]:
        spread = model_spread if spec["use_model_spread"] else float(spec["spread_manual"])
        prepared.append(
            {
                "code": spec["code"],
                "spread": spread,
                "competition": float(spec["competition"]),
                "economic": float(spec["economic"]),
                "health": float(spec["health"]),
                "ecology": float(spec["ecology"]),
                "utility": float(spec["utility"]),
            }
        )
        meta.append(spec)
    scored = score_species(prepared, contract["weights"])
    rows: list[dict[str, object]] = []
    for spec, raw, sc in zip(meta, prepared, scored):
        rows.append(
            {
                "code": spec["code"],
                "name": spec["name"],
                "region": spec["region"],
                "spread": raw["spread"],
                "spread_from_model": bool(spec["use_model_spread"]),
                "temperate_cover12": cover12 if spec["use_model_spread"] else np.nan,
                "competition": raw["competition"],
                "economic": raw["economic"],
                "health": raw["health"],
                "ecology": raw["ecology"],
                "utility": raw["utility"],
                "impact": sc["impact"],
            }
        )
    table = pd.DataFrame(rows).sort_values("impact", ascending=False)
    table.to_csv(RESULTS / "impact_factors.csv", index=False)
    return table


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    _ensure_forcing()
    monthly = run_baselines()
    run_drought()
    pareto = run_mowing_sweep()
    kpp = run_kpp_check()
    impact = run_impact(monthly)
    summary = {
        "n_monthly_rows": int(len(monthly)),
        "n_pareto_policies": int(len(pareto)),
        "n_pareto_front": int(pareto["on_front"].sum()),
        "kpp_rel_err": float(kpp.iloc[0]["rel_err"]),
        "impact_top": str(impact.iloc[0]["code"]),
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

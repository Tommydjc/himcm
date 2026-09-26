#!/usr/bin/env python3
"""日步 365 仿真 → 5 节点截面 → Fisher–KPP 对照 → 论文图。

运行::

    PYTHONPATH=. python -m src.scripts.run_monthly_analysis

只写 ``monthly_population_metrics.csv`` / ``daily_front_radius.csv`` /
``daily_kpp_check.csv`` 与 ``results/snapshots/hectare_day*.npy``。
不覆盖周步 ``monthly_metrics.csv``、``kpp_check.csv``。
图只从上述 CSV/NPY 读取。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from numpy.typing import NDArray

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from src.biology.phenology import PhenologyParams, StageState, gamma_environment
from src.simulation.grid_engine import (
    CENTER_COL,
    CENTER_ROW,
    DailyForcing,
    load_daily_forcing,
    run_hectare_days,
)
from src.theory.fisher_kpp import (
    TAU_DAY,
    diffusion_from_wind_sigma,
    intrinsic_rate_daily,
    mean_diffusion_from_winds,
    regress_front_speed,
    relative_wave_error,
    traveling_wave_speed,
)

SNAPSHOT_DAYS: tuple[int, ...] = (30, 60, 90, 180, 365)
MONTH_LABELS: tuple[int, ...] = (1, 2, 3, 6, 12)
OCCUPY_THRESH: float = 0.05
R_CAP_M: float = 42.0
LOG_VMIN: float = 1.0e-2
WONG_BLUE = "#0072B2"
WONG_VERM = "#D55E00"
WONG_ORANGE = "#E69F00"


def plants_field(state: StageState) -> NDArray[np.float64]:
    r"""建成株密度 \(L+R+A\)（株/m²），形状 ``(n, n)``。"""
    return state.seedling_total + state.rosette + state.adult


def clone_state(state: StageState) -> StageState:
    """深拷贝阶段场。"""
    return StageState(
        seed=state.seed.copy(),
        seedling=state.seedling.copy(),
        rosette=state.rosette.copy(),
        adult=state.adult.copy(),
    )


def radius_r95_m(
    field: NDArray[np.float64],
    threshold: float = OCCUPY_THRESH,
    center_row: int = CENTER_ROW,
    center_col: int = CENTER_COL,
) -> float:
    """相对发源格的 95% 占用半径（m）。"""
    rows, cols = np.where(field >= threshold)
    if rows.size == 0:
        return 0.0
    dist = np.hypot(
        rows.astype(np.float64) - float(center_row),
        cols.astype(np.float64) - float(center_col),
    )
    return float(np.percentile(dist, 95))


def cover_percent(field: NDArray[np.float64], threshold: float = OCCUPY_THRESH) -> float:
    """1 公顷被占格百分比。"""
    return float(100.0 * np.mean(field >= threshold))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.usetex": False,
            "mathtext.default": "regular",
        }
    )


def theory_from_forcing(
    forcings: list[DailyForcing],
    params: PhenologyParams,
) -> dict[str, float]:
    r"""由强迫与物候估 \(D,r,c^*\)（不读仿真锋面）。"""
    releasing = [row for row in forcings if row.releasing]
    sample = releasing if releasing else forcings
    speeds = np.array([row.wind_mps for row in sample], dtype=np.float64)
    dirs = np.array([row.wind_from_deg for row in sample], dtype=np.float64)
    diff_d, moment, sigma_sq = mean_diffusion_from_winds(speeds, dirs, tau=TAU_DAY)
    d_sigma = diffusion_from_wind_sigma(sigma_sq, tau=TAU_DAY)
    temps = np.array([row.temp_c for row in forcings], dtype=np.float64)
    moist = np.array([row.moisture for row in forcings], dtype=np.float64)
    gammas = np.array(
        [gamma_environment(float(t), float(m), params) for t, m in zip(temps, moist)],
        dtype=np.float64,
    )
    gamma_bar = float(np.mean(gammas))
    p_rel = float(np.mean([1.0 if row.releasing else 0.0 for row in forcings]))
    r_day = intrinsic_rate_daily(params, gamma_bar, p_rel, tau=TAU_DAY)
    c_star = traveling_wave_speed(max(r_day, 0.0), diff_d)
    return {
        "D_m2_per_day": diff_d,
        "D_from_sigma": d_sigma,
        "M2_m2": moment,
        "sigma_wind_sq": sigma_sq,
        "r_per_day": r_day,
        "gamma_mean": gamma_bar,
        "release_prob": p_rel,
        "c_star_m_per_day": c_star,
    }


def run_year_with_captures(
    forcings: list[DailyForcing],
    params: PhenologyParams,
) -> tuple[dict[int, StageState], NDArray[np.float64], NDArray[np.float64], float]:
    r"""积分全年，返回快照、逐日 \(R_{95}\)、累积落籽、耗时。"""
    snapshots: dict[int, StageState] = {}
    n = len(forcings)
    r95 = np.zeros(n, dtype=np.float64)
    rain_cum = np.zeros(n, dtype=np.float64)

    def on_day(day: int, state: StageState) -> None:
        t_elapsed = int(day) + 1
        idx = min(max(t_elapsed - 1, 0), n - 1)
        plants = plants_field(state)
        r95[idx] = radius_r95_m(plants)
        if t_elapsed in SNAPSHOT_DAYS:
            snapshots[t_elapsed] = clone_state(state)

    result = run_hectare_days(forcings, params=params, on_day=on_day)
    rain_cum[:] = np.cumsum(result.n_seed_rain)
    return snapshots, r95, rain_cum, result.elapsed_s


def write_tables(
    climate: str,
    snapshots: dict[int, StageState],
    r95: NDArray[np.float64],
    rain_cum: NDArray[np.float64],
    theory: dict[str, float],
    c_num: float,
    fit_i0: int,
    fit_i1: int,
    elapsed_s: float,
    out_dir: Path,
    snap_dir: Path,
) -> tuple[Path, Path, Path]:
    """写 5 节点表、逐日锋面、日步 KPP 验算。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)
    rel = relative_wave_error(c_num, theory["c_star_m_per_day"])
    rows: list[dict[str, object]] = []
    for day, month in zip(SNAPSHOT_DAYS, MONTH_LABELS):
        state = snapshots[day]
        plants = plants_field(state)
        np.save(snap_dir / f"hectare_day{day:03d}.npy", plants)
        np.save(snap_dir / f"hectare_adult_day{day:03d}.npy", state.adult)
        cover = cover_percent(plants)
        rows.append(
            {
                "climate": climate,
                "day": day,
                "month": month,
                "n_adult": float(np.sum(state.adult)),
                "n_seed_rain_cum": float(rain_cum[day - 1]),
                "n_seed_bank": float(np.sum(state.seed)),
                "n_plants": float(np.sum(plants)),
                "r95_m": float(r95[day - 1]),
                "cover_pct": cover,
                "saturated": int(cover >= 90.0 or r95[day - 1] >= R_CAP_M),
                "D_m2_per_day": theory["D_m2_per_day"],
                "r_per_day": theory["r_per_day"],
                "c_star_m_per_day": theory["c_star_m_per_day"],
                "c_num_m_per_day": c_num,
                "rel_err": rel,
                "fit_day0": fit_i0 + 1,
                "fit_day1": fit_i1 + 1,
                "elapsed_s": elapsed_s,
            }
        )
    monthly_path = out_dir / "monthly_population_metrics.csv"
    pd.DataFrame(rows).to_csv(monthly_path, index=False)
    daily_path = out_dir / "daily_front_radius.csv"
    pd.DataFrame(
        {
            "day": np.arange(1, r95.size + 1, dtype=np.int64),
            "r95_m": r95,
            "n_seed_rain_cum": rain_cum,
        }
    ).to_csv(daily_path, index=False)
    kpp_path = out_dir / "daily_kpp_check.csv"
    pd.DataFrame(
        [
            {
                "climate": climate,
                "D_m2_per_day": theory["D_m2_per_day"],
                "D_from_sigma": theory["D_from_sigma"],
                "M2_m2": theory["M2_m2"],
                "sigma_wind_sq": theory["sigma_wind_sq"],
                "r_per_day": theory["r_per_day"],
                "gamma_mean": theory["gamma_mean"],
                "release_prob": theory["release_prob"],
                "c_star_m_per_day": theory["c_star_m_per_day"],
                "c_num_m_per_day": c_num,
                "rel_err": rel,
                "fit_day0": fit_i0 + 1,
                "fit_day1": fit_i1 + 1,
            }
        ]
    ).to_csv(kpp_path, index=False)
    return monthly_path, daily_path, kpp_path


def plot_spatial_spread_5stages(
    snap_dir: Path,
    monthly_csv: Path,
    fig_dir: Path,
) -> Path:
    r"""5 联排共用 Log 色标；标 (50,50) 与 \(R_{95}\) 前锋。"""
    table = pd.read_csv(monthly_csv)
    fields: list[NDArray[np.float64]] = []
    for day in SNAPSHOT_DAYS:
        path = snap_dir / f"hectare_day{day:03d}.npy"
        fields.append(np.load(path))
    vmax = max(float(field.max()) for field in fields)
    vmax = max(vmax, LOG_VMIN * 10.0)
    norm = LogNorm(vmin=LOG_VMIN, vmax=vmax)
    fig, axes = plt.subplots(1, 5, figsize=(13.6, 2.9), constrained_layout=True)
    im = None
    for ax, field, day, month in zip(axes, fields, SNAPSHOT_DAYS, MONTH_LABELS):
        shown = np.maximum(field, LOG_VMIN)
        im = ax.imshow(
            shown,
            origin="lower",
            cmap="YlGn",
            norm=norm,
            extent=(0.0, 100.0, 0.0, 100.0),
        )
        ax.contour(
            np.linspace(0.5, 99.5, field.shape[1]),
            np.linspace(0.5, 99.5, field.shape[0]),
            field,
            levels=[OCCUPY_THRESH],
            colors="#D55E00",
            linewidths=1.1,
        )
        ax.plot(50.5, 50.5, marker="*", color="white", markersize=9, markeredgecolor="k", markeredgewidth=0.6)
        row = table.loc[table["day"] == day].iloc[0]
        ax.set_title(f"{month} mo  (t={day}d)\n$R_{{95}}$={row['r95_m']:.1f} m")
        ax.set_xlabel("east (m)")
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
    axes[0].set_ylabel("north (m)")
    assert im is not None
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label=r"$L+R+A$ (plants / m$^2$, log)")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_spatial_spread_5stages.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_pde_vs_simulation(
    daily_csv: Path,
    kpp_csv: Path,
    monthly_csv: Path,
    fig_dir: Path,
) -> Path:
    r"""离散 \(R_{95}(t)\) 对连续 \(c^* t\) 与后期回归直线。"""
    daily = pd.read_csv(daily_csv)
    kpp = pd.read_csv(kpp_csv).iloc[0]
    monthly = pd.read_csv(monthly_csv)
    days = daily["day"].to_numpy(dtype=np.float64)
    radii = daily["r95_m"].to_numpy(dtype=np.float64)
    c_star = float(kpp["c_star_m_per_day"])
    c_num = float(kpp["c_num_m_per_day"])
    t0 = float(kpp["fit_day0"])
    t1 = float(kpp["fit_day1"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(days, radii, color=WONG_BLUE, lw=1.8, label=r"CA $R_{95}(t)$")
    mask = (days >= t0) & (days <= t1)
    if np.any(mask):
        t_fit = days[mask]
        r0 = float(radii[mask][0])
        ax.plot(
            t_fit,
            r0 + c_star * (t_fit - t_fit[0]),
            color=WONG_VERM,
            ls="--",
            lw=1.6,
            label=f"KPP slope c* = {c_star:.3f} m/d",
        )
        if c_num > 0.0:
            ax.plot(
                t_fit,
                r0 + c_num * (t_fit - t_fit[0]),
                color=WONG_ORANGE,
                lw=2.0,
                label=f"late fit c_num = {c_num:.3f} m/d",
            )
    ax.scatter(
        monthly["day"],
        monthly["r95_m"],
        c="k",
        s=36,
        zorder=5,
        label="Req-1 nodes",
    )
    ax.axhline(R_CAP_M, color="#999999", ls=":", lw=1.0, label=f"regression cap {R_CAP_M:.0f} m")
    ax.set_xlabel("day")
    ax.set_ylabel(r"$R_{95}$ (m)")
    ax.set_xlim(1.0, 365.0)
    ax.set_ylim(0.0, 72.0)
    ax.legend(frameon=False, fontsize=8)
    rel = float(kpp["rel_err"])
    t_exit = R_CAP_M / max(c_star, 1.0e-9)
    ax.set_title("Fisher-KPP slope check   relative error = %.3f" % rel)
    ax.text(
        0.02,
        0.98,
        "c-star t exits 1 ha by day ~%.0f\nslope window days %.0f to %.0f" % (t_exit, t0, t1),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="#333333",
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_pde_vs_simulation_wave.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="365-day hectare snapshots + KPP check")
    parser.add_argument("--climate", default="temperate", choices=("temperate", "arid", "tropical"))
    args = parser.parse_args()
    params = PhenologyParams()
    forcings = load_daily_forcing(args.climate, n_days=365)
    if len(forcings) < 365:
        raise RuntimeError(f"forcing length {len(forcings)} < 365")
    theory = theory_from_forcing(forcings, params)
    snapshots, r95, rain_cum, elapsed = run_year_with_captures(forcings, params)
    missing = [day for day in SNAPSHOT_DAYS if day not in snapshots]
    if missing:
        raise RuntimeError(f"missing snapshots {missing}")
    times = np.arange(1, r95.size + 1, dtype=np.float64)
    c_num, i0, i1 = regress_front_speed(times, r95, r_cap_m=R_CAP_M, n_late=30)
    results_dir = PACK_ROOT / "results"
    snap_dir = results_dir / "snapshots"
    fig_dir = PACK_ROOT / "paper_figures"
    monthly_path, daily_path, kpp_path = write_tables(
        args.climate,
        snapshots,
        r95,
        rain_cum,
        theory,
        c_num,
        i0,
        i1,
        elapsed,
        results_dir,
        snap_dir,
    )
    _style()
    fig1 = plot_spatial_spread_5stages(snap_dir, monthly_path, fig_dir)
    fig2 = plot_pde_vs_simulation(daily_path, kpp_path, monthly_path, fig_dir)
    print(f"elapsed_s={elapsed:.3f}")
    print(f"D={theory['D_m2_per_day']:.6f} D_sigma={theory['D_from_sigma']:.6f}")
    print(f"r={theory['r_per_day']:.6f} c*={theory['c_star_m_per_day']:.6f} c_num={c_num:.6f}")
    print(f"rel_err={relative_wave_error(c_num, theory['c_star_m_per_day']):.6f} fit={i0 + 1}-{i1 + 1}")
    print(f"wrote {monthly_path.relative_to(PACK_ROOT)}")
    print(f"wrote {daily_path.relative_to(PACK_ROOT)}")
    print(f"wrote {kpp_path.relative_to(PACK_ROOT)}")
    print(f"wrote {fig1.relative_to(PACK_ROOT)}")
    print(f"wrote {fig2.relative_to(PACK_ROOT)}")


if __name__ == "__main__":
    main()

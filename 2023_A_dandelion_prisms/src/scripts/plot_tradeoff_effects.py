#!/usr/bin/env python3
"""四策略情景对照图：空间场、逐日轨迹、损益分解。

只读 ``results/bioeconomic_tradeoff.csv``、``tradeoff_daily_traces.csv``
与 ``results/snapshots/tradeoff_*_day*.npy``。缺文件时先跑积分。

运行::

    PYTHONPATH=. python -m src.scripts.plot_tradeoff_effects
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

from src.decision.tradeoff_model import (
    EFFECT_SNAP_DAYS,
    export_strategy_artifacts,
    first_true_bloom_day,
    load_daily_forcing,
    plot_tradeoff,
    prisms_mow_day,
)

FIG_DIR: Path = PACK_ROOT / "paper_figures"
SNAP_DIR: Path = PACK_ROOT / "results" / "snapshots"
LOG_VMIN: float = 1.0e-2
STRATEGIES: tuple[tuple[str, str, str], ...] = (
    ("wild", "S1 wild", "#009E73"),
    ("weekly_mow", "S2 weekly mow", "#E69F00"),
    ("chemical", "S3 herbicide", "#D55E00"),
    ("prisms", "S4 PRISMS", "#0072B2"),
)
DAY_TITLES: dict[int, str] = {
    90: "day 90  early bloom",
    120: "day 120  after window",
    180: "day 180  mid year",
    365: "day 365  year end",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.usetex": False,
            "mathtext.default": "regular",
        }
    )


def _ensure_artifacts(force: bool) -> None:
    traces = PACK_ROOT / "results" / "tradeoff_daily_traces.csv"
    needed = [
        SNAP_DIR / f"tradeoff_{name}_day{day:03d}.npy"
        for name, _label, _c in STRATEGIES
        for day in EFFECT_SNAP_DAYS
    ]
    missing = (not traces.is_file()) or any(not path.is_file() for path in needed)
    if force or missing:
        export_strategy_artifacts("temperate", results_dir=PACK_ROOT / "results")


def plot_spatial_effects(snap_dir: Path, fig_dir: Path) -> Path:
    """4 策略 × 4 时点建成株热图，共用 Log 色标。"""
    fields: list[list[NDArray[np.float64]]] = []
    vmax = LOG_VMIN
    for name, _label, _color in STRATEGIES:
        row: list[NDArray[np.float64]] = []
        for day in EFFECT_SNAP_DAYS:
            path = snap_dir / f"tradeoff_{name}_day{day:03d}.npy"
            field = np.load(path)
            row.append(field)
            vmax = max(vmax, float(field.max()))
        fields.append(row)
    norm = LogNorm(vmin=LOG_VMIN, vmax=max(vmax, LOG_VMIN * 10.0))
    fig, axes = plt.subplots(4, 4, figsize=(11.2, 10.2), constrained_layout=True)
    im = None
    for i, (name, label, _color) in enumerate(STRATEGIES):
        for j, day in enumerate(EFFECT_SNAP_DAYS):
            ax = axes[i, j]
            shown = np.maximum(fields[i][j], LOG_VMIN)
            im = ax.imshow(
                shown,
                origin="lower",
                cmap="YlGn",
                norm=norm,
                extent=(0.0, 100.0, 0.0, 100.0),
            )
            ax.contour(
                np.linspace(0.5, 99.5, shown.shape[1]),
                np.linspace(0.5, 99.5, shown.shape[0]),
                fields[i][j],
                levels=[0.05],
                colors="#D55E00",
                linewidths=0.7,
            )
            ax.plot(50.5, 50.5, marker="*", color="white", markersize=6, markeredgecolor="k", markeredgewidth=0.4)
            if i == 0:
                ax.set_title(DAY_TITLES[day], fontsize=9)
            if j == 0:
                ax.set_ylabel(label)
            ax.set_xticks([0, 50, 100])
            ax.set_yticks([0, 50, 100])
            if i < 3:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
    assert im is not None
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="L+R+A plants / m2 (log)")
    fig.suptitle("Same 1 ha, four management regimes", fontsize=12)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_strategy_spatial_effects.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_timeseries(trace_csv: Path, fig_dir: Path) -> Path:
    """覆盖率、开花成株、建成株总量随日变化。"""
    table = pd.read_csv(trace_csv)
    forcings = load_daily_forcing("temperate", n_days=365)
    t_cut = prisms_mow_day(forcings)
    t_bloom = first_true_bloom_day(forcings)
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 7.6), sharex=True, constrained_layout=True)
    series = (
        ("cover", "occupied fraction", False),
        ("flowers", "flowering adults (releasing days)", True),
        ("n_plants", "standing plants L+R+A", True),
    )
    color_of = {name: color for name, _label, color in STRATEGIES}
    label_of = {name: label for name, label, _color in STRATEGIES}
    for ax, (col, ylab, logy) in zip(axes, series):
        for name, _label, _color in STRATEGIES:
            sub = table[table["strategy"] == name]
            ax.plot(sub["day"], sub[col], color=color_of[name], lw=1.5, label=label_of[name])
        ax.axvspan(59.0, 120.0, color="#0072B2", alpha=0.06)
        ax.axvline(t_bloom + 1, color="#0072B2", ls=":", lw=0.9)
        ax.axvline(t_cut + 1, color="#0072B2", ls="--", lw=0.9)
        ax.set_ylabel(ylab)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(ax.get_ylim()[0], 1.0e-2))
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0].set_title("Daily trajectories (band = Mar-Apr; dashed = PRISMS cut)")
    axes[-1].set_xlabel("day")
    axes[-1].set_xlim(1.0, 365.0)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_strategy_timeseries.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_accounts(score_csv: Path, fig_dir: Path) -> Path:
    """益处分解 vs 成本分解，按策略并排。"""
    table = pd.read_csv(score_csv)
    order = [name for name, _label, _c in STRATEGIES]
    table = table.set_index("strategy").loc[order].reset_index()
    labels = [label for _n, label, _c in STRATEGIES]
    x = np.arange(len(table), dtype=np.float64)
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    ax_b.bar(x, table["B_pollinator"], color="#0072B2", width=0.62, label="pollinator")
    ax_b.bar(x, table["B_soil"], bottom=table["B_pollinator"], color="#009E73", width=0.62, label="soil")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=15)
    ax_b.set_ylabel("benefit (wild-normalized)")
    ax_b.set_title("B_eco stack")
    ax_b.legend(frameon=False, fontsize=8)
    ax_c.bar(x, table["C_turf"], color="#D55E00", width=0.62, label="turf damage")
    ax_c.bar(x, table["C_mow"], bottom=table["C_turf"], color="#E69F00", width=0.62, label="mowing")
    ax_c.bar(
        x,
        table["C_herb"],
        bottom=table["C_turf"] + table["C_mow"],
        color="#882255",
        width=0.62,
        label="herbicide",
    )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, rotation=15)
    ax_c.set_ylabel("cost (relative)")
    ax_c.set_title("C_cost stack")
    ax_c.legend(frameon=False, fontsize=8)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_strategy_accounts.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot four-strategy effect panels")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    _style()
    _ensure_artifacts(args.force_rerun)
    score_csv = PACK_ROOT / "results" / "bioeconomic_tradeoff.csv"
    trace_csv = PACK_ROOT / "results" / "tradeoff_daily_traces.csv"
    paths = [
        plot_spatial_effects(SNAP_DIR, FIG_DIR),
        plot_timeseries(trace_csv, FIG_DIR),
        plot_accounts(score_csv, FIG_DIR),
        plot_tradeoff(pd.read_csv(score_csv), FIG_DIR),
    ]
    for path in paths:
        print(f"wrote {path.relative_to(PACK_ROOT)}")


if __name__ == "__main__":
    main()

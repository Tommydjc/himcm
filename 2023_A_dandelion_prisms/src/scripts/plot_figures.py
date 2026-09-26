#!/usr/bin/env python3
"""只读 results/ 出图：热力扩散、风玫瑰、帕累托。

运行::

    PYTHONPATH=. python -m src.scripts.plot_figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

FIG_DIR: Path = PACK_ROOT / "paper_figures"
SNAP_DIR: Path = PACK_ROOT / "results" / "snapshots"
WONG_BLUE = "#0072B2"
WONG_GREEN = "#009E73"
WONG_ORANGE = "#E69F00"
WONG_VERM = "#D55E00"
MONTH_WEEKS: tuple[tuple[int, int], ...] = ((1, 4), (2, 8), (3, 13), (6, 26), (12, 52))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_heatmaps(climate: str = "temperate") -> Path:
    """1/2/3/6/12 月建成株热图。"""
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.8), constrained_layout=True)
    vmax = 0.0
    fields: list[np.ndarray] = []
    for _month, week in MONTH_WEEKS:
        path = SNAP_DIR / f"{climate}_week{week:02d}.npy"
        field = np.load(path)
        fields.append(field)
        vmax = max(vmax, float(field.max()))
    vmax = max(vmax, 1.0)
    for ax, field, (month, _week) in zip(axes, fields, MONTH_WEEKS):
        shown = np.maximum(field, 1.0e-3)
        im = ax.imshow(
            shown,
            origin="lower",
            cmap="YlGn",
            norm=LogNorm(vmin=1.0e-2, vmax=vmax),
            extent=(0, 100, 0, 100),
        )
        ax.set_title(f"{month} month")
        ax.set_xlabel("east (m)")
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
    axes[0].set_ylabel("north (m)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label=r"plants / m$^2$ (log)")
    out = FIG_DIR / f"fig_heatmap_{climate}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_wind_rose(climate: str = "temperate") -> Path:
    """小时风向风速玫瑰。"""
    weather = pd.read_csv(PACK_ROOT / "data" / "raw" / "weather" / f"{climate}_hourly.csv")
    theta = np.deg2rad(weather["wind_from_deg"].to_numpy())
    speed = weather["wind_speed_mps"].to_numpy()
    bins = np.linspace(0.0, 2.0 * np.pi, 17)
    fig = plt.figure(figsize=(4.4, 4.4))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    counts, edges = np.histogram(theta, bins=bins, weights=speed)
    width = edges[1] - edges[0]
    ax.bar(edges[:-1], counts / counts.max(), width=width, color=WONG_BLUE, edgecolor="white")
    ax.set_title(f"{climate} wind rose (speed-weighted)")
    out = FIG_DIR / f"fig_windrose_{climate}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_pareto() -> Path:
    """割草成本–盖度。"""
    table = pd.read_csv(PACK_ROOT / "results" / "bioeconomic_pareto.csv")
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    off = table[~table["on_front"]]
    on = table[table["on_front"]].sort_values("cost")
    ax.scatter(off["cost"], off["cover_frac"], c=WONG_ORANGE, s=36, label="dominated", zorder=2)
    ax.plot(on["cost"], on["cover_frac"], color=WONG_GREEN, lw=1.6, zorder=3)
    ax.scatter(on["cost"], on["cover_frac"], c=WONG_GREEN, s=44, label="Pareto", zorder=4)
    ax.set_xlabel(r"relative cost $C=n(1+\eta/2)$")
    ax.set_ylabel("12-month cover fraction")
    ax.legend(frameon=False)
    out = FIG_DIR / "fig_pareto_mowing.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_climate_cover() -> Path:
    """三气候月末覆盖率。"""
    table = pd.read_csv(PACK_ROOT / "results" / "monthly_metrics.csv")
    base = table[table["scenario"] == "baseline"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    colors = {"temperate": WONG_GREEN, "arid": WONG_VERM, "tropical": WONG_BLUE}
    for name, sub in base.groupby("climate"):
        ax.plot(sub["month"], sub["cover_frac"], marker="o", color=colors.get(str(name), "k"), label=str(name))
    drought_path = PACK_ROOT / "results" / "drought_monthly.csv"
    if drought_path.is_file():
        drought = pd.read_csv(drought_path)
        ax.plot(drought["month"], drought["cover_frac"], marker="s", ls="--", color=WONG_ORANGE, label="temperate drought")
    ax.set_xlabel("month")
    ax.set_ylabel("cover fraction")
    ax.set_xticks([1, 2, 3, 6, 12])
    ax.legend(frameon=False)
    out = FIG_DIR / "fig_climate_cover.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    _style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_heatmaps("temperate"),
        plot_heatmaps("arid"),
        plot_heatmaps("tropical"),
        plot_wind_rose("temperate"),
        plot_pareto(),
        plot_climate_cover(),
    ]
    for path in paths:
        print(f"wrote {path.relative_to(PACK_ROOT)}")


if __name__ == "__main__":
    main()

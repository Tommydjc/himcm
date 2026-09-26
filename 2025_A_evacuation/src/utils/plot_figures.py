#!/usr/bin/env python3
"""从 ``results/experiment_summary.csv`` 绘制 HiMCM 论文插图。

输出（300 DPI，Times New Roman，Wong 色盲友好色）::

    paper_figures/fig1_personnel_tradeoff.png
    paper_figures/fig2_hazard_impact.png
    paper_figures/fig3_redundancy_tradeoff.png

运行::

    PYTHONPATH=. python src/utils/plot_figures.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "results" / "experiment_summary.csv"
FIG_DIR = ROOT / "paper_figures"

WONG_BLUE = "#0072B2"
WONG_ORANGE = "#E69F00"
WONG_GREEN = "#009E73"
WONG_VERMILLION = "#D55E00"
WONG_SKY = "#56B4E9"
WONG_PURPLE = "#CC79A7"

SCENARIO_COLOR: dict[str, str] = {
    "daycare": WONG_BLUE,
    "warehouse": WONG_ORANGE,
    "office": WONG_GREEN,
}

SCENARIO_LABEL: dict[str, str] = {
    "daycare": "Daycare",
    "warehouse": "Warehouse",
    "office": "Office",
}


def _configure_style() -> None:
    """Times New Roman；若系统无该字体则回退衬线族。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = "Times New Roman" if "Times New Roman" in available else "serif"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rows(csv_path: Path = CSV_PATH) -> list[dict[str, str]]:
    """读取蒙特卡洛原始行。"""
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing experiment CSV: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    """样本均值与 95% t 置信区间 ``(mean, lo, hi)``。

    .. math::

        \\bar{x} \\pm t_{n-1,\\,0.975}\\, s / \\sqrt{n}
    """
    x = np.asarray(values, dtype=float)
    n = int(x.size)
    mean = float(np.mean(x)) if n else float("nan")
    if n < 2:
        return mean, mean, mean
    std = float(np.std(x, ddof=1))
    if std == 0.0:
        return mean, mean, mean
    se = std / float(np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    half = tcrit * se
    return mean, mean - half, mean + half


def _group_clear_times(
    rows: list[dict[str, str]],
    scenario: str,
) -> dict[int, np.ndarray]:
    """无烟工况下按人数收集 ``t_clear``（秒）。"""
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["scenario"] != scenario:
            continue
        if row["hazard_mode"] != "clear":
            continue
        n_agents = int(row["n_agents"])
        buckets[n_agents].append(float(row["t_clear"]))
    return {k: np.asarray(v, dtype=float) for k, v in sorted(buckets.items())}


def inflection_staffing(staffing: list[int], means: list[float]) -> tuple[int, float]:
    """边际效益最大的增员步：``argmax (T(n)-T(n+1))``，返回到达的人数与节约秒数。"""
    if len(staffing) < 2:
        return staffing[0], 0.0
    best_n = staffing[1]
    best_gain = -1.0e18
    for i in range(len(staffing) - 1):
        gain = float(means[i] - means[i + 1])
        if gain > best_gain:
            best_gain = gain
            best_n = staffing[i + 1]
    return best_n, best_gain


def plot_fig1_personnel_tradeoff(rows: list[dict[str, str]], save_path: Path) -> None:
    """Req 3：Daycare / Warehouse 人数 vs 清空时间（柱+折线，95% CI）。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=False)
    panels = (("daycare", axes[0]), ("warehouse", axes[1]))

    for scenario, ax in panels:
        grouped = _group_clear_times(rows, scenario)
        if not grouped:
            raise ValueError(f"no clear-condition rows for {scenario}")
        ns = list(grouped.keys())
        stats_rows = [mean_ci95(grouped[n]) for n in ns]
        means = [s[0] for s in stats_rows]
        yerr = np.vstack(
            (
                [m - lo for m, lo, _hi in stats_rows],
                [hi - m for m, _lo, hi in stats_rows],
            )
        )
        color = SCENARIO_COLOR[scenario]
        x = np.arange(len(ns), dtype=float)
        ax.bar(
            x,
            means,
            width=0.55,
            color=color,
            alpha=0.35,
            edgecolor=color,
            linewidth=1.0,
            label="Mean $T_{\\mathrm{clear}}$",
            zorder=2,
        )
        ax.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="-o",
            color=color,
            ecolor=WONG_VERMILLION,
            elinewidth=1.4,
            capsize=4.5,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.4,
            label="95% CI (Student $t$)",
            zorder=3,
        )
        knee, gain = inflection_staffing(ns, means)
        knee_x = float(ns.index(knee))
        ax.scatter(
            [knee_x],
            [means[ns.index(knee)]],
            s=160,
            marker="*",
            color=WONG_VERMILLION,
            zorder=4,
            label="Best marginal-return knee",
        )
        ax.annotate(
            f"knee: {knee} firefighters\n"
            f"$\\Delta T_{{\\mathrm{{clear}}}}={gain:.1f}$ s",
            xy=(knee_x, means[ns.index(knee)]),
            xytext=(knee_x + 0.28, means[ns.index(knee)] + 0.08 * max(means)),
            fontsize=9,
            color=WONG_VERMILLION,
            arrowprops={"arrowstyle": "->", "color": WONG_VERMILLION, "lw": 0.9},
        )
        ax.set_xticks(x, [str(n) for n in ns])
        ax.set_xlabel("Number of firefighters")
        ax.set_ylabel("Total sweep time $T_{\\mathrm{clear}}$ (seconds)")
        ax.set_title(SCENARIO_LABEL[scenario])
        ax.set_ylim(0.0, max(means) * 1.28)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Req 3: Personnel–clearance time trade-off (ideal, no smoke)",
        y=1.02,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_hazard_impact(rows: list[dict[str, str]], save_path: Path) -> None:
    """Req 4：理想 vs 浓烟下各建筑脆弱人群疏散成功率（百分比）箱线。"""
    order = ("office", "daycare", "warehouse")
    modes = ("clear", "smoke")
    mode_label = {"clear": "Ideal (no smoke)", "smoke": "Dynamic smoke"}
    mode_color = {"clear": WONG_SKY, "smoke": WONG_VERMILLION}

    data: dict[tuple[str, str], list[float]] = {}
    for scenario in order:
        for mode in modes:
            vals = [
                100.0 * float(row["vulnerable_success_rate"])
                for row in rows
                if row["scenario"] == scenario and row["hazard_mode"] == mode
            ]
            data[(scenario, mode)] = vals
            if not vals:
                raise ValueError(f"empty boxplot sample for {scenario}/{mode}")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    width = 0.34
    positions: list[float] = []
    plot_data: list[list[float]] = []
    colors: list[str] = []
    xticks: list[float] = []
    xticklabels: list[str] = []

    for i, scenario in enumerate(order):
        center = float(i)
        xticks.append(center)
        xticklabels.append(SCENARIO_LABEL[scenario])
        for j, mode in enumerate(modes):
            pos = center + (-width / 2.0 if j == 0 else width / 2.0)
            positions.append(pos)
            plot_data.append(data[(scenario, mode)])
            colors.append(mode_color[mode])

    box = ax.boxplot(
        plot_data,
        positions=positions,
        widths=width * 0.9,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.3},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 3.5, "alpha": 0.55},
        showfliers=True,
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_edgecolor("#222222")
        patch.set_alpha(0.85)

    ax.set_xticks(xticks, xticklabels)
    ax.set_ylabel("Vulnerable-population salvage rate (percentage)")
    ax.set_ylim(-5.0, 108.0)
    ax.set_title("Req 4: Ideal vs dynamic-smoke impact on salvage success")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=mode_color[m], edgecolor="#222222", label=mode_label[m])
        for m in modes
    ]
    ax.legend(handles=legend_handles, loc="center right", frameon=True)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_redundancy_tradeoff(rows: list[dict[str, str]], save_path: Path) -> None:
    """Req 3 补充：人数 vs 重复扫荡冗余度（无烟，95% CI）。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for scenario, marker in (("daycare", "o"), ("warehouse", "s")):
        buckets: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            if row["scenario"] != scenario or row["hazard_mode"] != "clear":
                continue
            buckets[int(row["n_agents"])].append(100.0 * float(row["redundancy_rate"]))
        if not buckets:
            continue
        ns = sorted(buckets)
        stats_rows = [mean_ci95(np.asarray(buckets[n], dtype=float)) for n in ns]
        means = [s[0] for s in stats_rows]
        yerr = np.vstack(
            (
                [m - lo for m, lo, _hi in stats_rows],
                [hi - m for m, _lo, hi in stats_rows],
            )
        )
        ax.errorbar(
            ns,
            means,
            yerr=yerr,
            fmt=f"-{marker}",
            color=SCENARIO_COLOR[scenario],
            ecolor=SCENARIO_COLOR[scenario],
            elinewidth=1.3,
            capsize=4.0,
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=1.3,
            label=SCENARIO_LABEL[scenario],
        )
    ax.set_xlabel("Number of firefighters")
    ax.set_xticks([2, 3, 4, 5])
    ax.set_ylabel("Redundancy rate (percentage of extra room entries / $|R|$)")
    ax.set_title("Req 3: Sweep redundancy vs crew size (ideal, no smoke)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """读取 CSV 并写出三张 300 DPI 插图。"""
    _configure_style()
    rows = load_rows()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_fig1_personnel_tradeoff(rows, FIG_DIR / "fig1_personnel_tradeoff.png")
    plot_fig2_hazard_impact(rows, FIG_DIR / "fig2_hazard_impact.png")
    plot_fig3_redundancy_tradeoff(rows, FIG_DIR / "fig3_redundancy_tradeoff.png")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()

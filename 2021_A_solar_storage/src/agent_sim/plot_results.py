r"""Paper figure: 72 h crewAI DSR vs passive house (from measured CSV).

Numbers on the axes and in callouts are computed from
``results/crewai_adaptive_vs_passive.csv``. The mixed-pack SOC floor is
about 17.8\% (not absolute 0\%); outage hours are hours with
\(P_{\mathrm{def}}>10^{-4}\,\mathrm{kW}\).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.optimization.sizing_milp import PACK_ROOT, RESULTS_DIR

CSV_PATH: Path = RESULTS_DIR / "crewai_adaptive_vs_passive.csv"
FIG_PATH: Path = PACK_ROOT / "paper_figures" / "fig_crewai_load_curtailment.png"
SOC_WARN: float = 0.20
DEF_EPS_KW: float = 1.0e-4


def _set_times_font() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def _outage_mask(def_kw: np.ndarray) -> np.ndarray:
    return np.asarray(def_kw, dtype=np.float64) > DEF_EPS_KW


def _longest_run(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for flag in mask:
        if flag:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def _span_outages(ax: plt.Axes, t: np.ndarray, mask: np.ndarray, color: str) -> None:
    """Shade contiguous outage runs on the time axis."""

    n = int(mask.size)
    i = 0
    labeled = False
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        t1 = t[j] if j < n else t[-1] + np.timedelta64(1, "h")
        ax.axvspan(
            t[i],
            t1,
            color=color,
            alpha=0.12,
            zorder=0,
            label="Unserved hours (passive)" if (not labeled and color == "#c45911") else None,
        )
        labeled = True
        i = j


def plot_crewai_comparison(
    csv_path: Path | None = None,
    fig_path: Path | None = None,
) -> Path:
    """Read the A/B CSV and write the two-panel 300 DPI figure."""

    src = Path(csv_path) if csv_path is not None else CSV_PATH
    dest = Path(fig_path) if fig_path is not None else FIG_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(src)
    need = {
        "timestamp",
        "raw_load_kW",
        "negotiated_load_kW",
        "soc_passive",
        "soc_adaptive",
        "p_def_passive_kW",
        "p_def_adaptive_kW",
    }
    missing = need.difference(frame.columns)
    if missing:
        raise ValueError(f"{src} missing {missing}")

    t = pd.to_datetime(frame["timestamp"])
    raw = frame["raw_load_kW"].to_numpy(dtype=np.float64)
    dsr = frame["negotiated_load_kW"].to_numpy(dtype=np.float64)
    soc_p = frame["soc_passive"].to_numpy(dtype=np.float64)
    soc_a = frame["soc_adaptive"].to_numpy(dtype=np.float64)
    def_p = frame["p_def_passive_kW"].to_numpy(dtype=np.float64)
    def_a = frame["p_def_adaptive_kW"].to_numpy(dtype=np.float64)
    shed = np.maximum(raw - dsr, 0.0)
    mask_p = _outage_mask(def_p)
    mask_a = _outage_mask(def_a)
    soc_floor = float(np.min(soc_p))
    n_out_p = int(np.sum(mask_p))
    n_out_a = int(np.sum(mask_a))
    def_e_p = float(np.sum(def_p))
    def_e_a = float(np.sum(def_a))
    shed_e = float(np.sum(shed))
    run_p = _longest_run(mask_p)
    run_a = _longest_run(mask_a)

    _set_times_font()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.08},
    )

    ax0 = axes[0]
    ax0.plot(t, raw, color="#7f7f7f", lw=1.6, label="Rigid load (passive)")
    ax0.plot(t, dsr, color="#1f4e79", lw=1.8, label="crewAI negotiated load")
    ax0.fill_between(
        t,
        dsr,
        raw,
        where=raw > dsr + 1.0e-6,
        interpolate=True,
        color="#c00000",
        alpha=0.28,
        label="Shedded non-essential power",
    )
    ax0.set_ylabel("Household load (kW)")
    ax0.set_ylim(bottom=0.0)
    ax0.grid(True, ls="--", alpha=0.35)
    ax0.spines["top"].set_visible(False)
    ax0.legend(loc="upper right", frameon=False, ncol=1)
    ax0.set_title(
        "Days 40–42: demand response vs. rigid household (measured CSV)",
        pad=8,
    )
    ax0.text(
        0.01,
        0.04,
        f"Energy shed = {shed_e:.1f} kWh over 72 h",
        transform=ax0.transAxes,
        fontsize=8,
        color="#9c0006",
    )

    ax1 = axes[1]
    _span_outages(ax1, t.to_numpy(), mask_p, "#c45911")
    ax1.plot(t, soc_p, color="#c45911", lw=1.7, label="SOC, passive (A)")
    ax1.plot(t, soc_a, color="#1f4e79", lw=1.8, label="SOC, crewAI DSR (B)")
    ax1.axhline(SOC_WARN, color="#c00000", ls="--", lw=1.15, label="20% SOC warning line")
    ax1.axhline(
        soc_floor,
        color="#833c0c",
        ls=":",
        lw=1.15,
        label=f"Pack $SOC_{{\\min}}$ = {100.0 * soc_floor:.1f}% (usable empty)",
    )
    ax1.set_ylabel("Battery SOC (dimensionless)")
    ax1.set_ylim(0.0, 1.02)
    ax1.set_xlabel("Time (America/Phoenix local, 2021)")
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax1.legend(loc="upper right", frameon=False, ncol=1)

    # Callouts from this CSV only — do not claim 0% SOC or 32 h / 9.2% SOC.
    if np.any(mask_p):
        t_empty = t.to_numpy()[int(np.argmax(mask_p & (soc_p <= soc_floor + 1.0e-9)))]
        ax1.annotate(
            f"A: usable SOC floor, then {n_out_p} h unserved\n"
            f"(longest run {run_p} h, {def_e_p:.1f} kWh deficit)",
            xy=(t_empty, soc_floor),
            xytext=(28, 55),
            textcoords="offset points",
            fontsize=8,
            color="#c45911",
            arrowprops={"arrowstyle": "->", "color": "#c45911", "lw": 0.9},
        )
    ax1.annotate(
        f"B: same $SOC_{{\\min}}$ floor; unserved hours {n_out_a}\n"
        f"(longest run {run_a} h, {def_e_a:.1f} kWh deficit)",
        xy=(t.iloc[int(np.argmin(soc_a))], float(np.min(soc_a))),
        xytext=(8, -42),
        textcoords="offset points",
        fontsize=8,
        color="#1f4e79",
        arrowprops={"arrowstyle": "->", "color": "#1f4e79", "lw": 0.9},
    )

    fig.align_ylabels(axes)
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dest}")
    print(
        f"CSV stats: SOC_min={soc_floor:.4f}  "
        f"outage A/B={n_out_p}/{n_out_a} h  "
        f"def A/B={def_e_p:.2f}/{def_e_a:.2f} kWh  "
        f"shed={shed_e:.2f} kWh"
    )
    return dest


def main() -> None:
    """CLI: ``python -m src.agent_sim.plot_results``."""

    plot_crewai_comparison()


if __name__ == "__main__":
    main()

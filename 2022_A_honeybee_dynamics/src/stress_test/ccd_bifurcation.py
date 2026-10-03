r"""多重现实环境胁迫与 CCD 分岔相变分析。

胁迫
    - A 新烟碱：\(\mu_F'=\mu_F(1+\mathrm{PesticideIndex})\)
    - B 瓦螨：\(\gamma_B'=\gamma_B e^{-v}\)，内勤死亡乘 2（寿命腰斩）
    - C 倒春寒：霜冻窗内觅食通量 = 0，纯耗库存

分岔
    扫描外勤夏日死亡 \(\mu_F\in[0.08,0.40]\)，步长 0.005；
    记录夏季峰、越冬存活、内勤提前出巢转化年龄代理 \(1/\bar r\)。

图
    - ``fig_colony_dynamics_healthy_vs_ccd.png``
    - ``fig_ccd_bifurcation_curve.png``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.autograd_engine.torch_sim import DifferentiableColonySimulator

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
RESULTS_DIR: Path = PACK_ROOT / "results"
FIGS_DIR: Path = PACK_ROOT / "paper_figures"

# 分岔扫描
MU_F_MIN: float = 0.08
MU_F_MAX: float = 0.40
MU_F_STEP: float = 0.005
# 健康 / CCD 对照点
MU_HEALTHY: float = 0.10
MU_CCD: float = 0.28
# 绝对区界（场胁迫下的越冬成蜂）：Safe ≥ 5000 > Compensatory ≥ 3000 > Collapse
N_SAFE: float = 5000.0
N_COLLAPSE: float = 3000.0
SUMMER_DOY0: int = 150
SUMMER_DOY1: int = 240
FROST_START: float = 100.0
FROST_END: float = 118.0


@dataclass(frozen=True, slots=True)
class StressBundle:
    """三重胁迫强度契约。"""

    pesticide_index: float = 0.0
    varroa_load: float = 0.0
    varroa_hive_mort_mult: float = 1.0
    frost_start: float = -1.0
    frost_end: float = -1.0
    label: str = "baseline"


STRESS_NONE = StressBundle(label="healthy_baseline")
# 分岔扫：田间三重胁迫（强度标定使 μ_F∈[0.08,0.40] 内出现崩溃）
STRESS_FIELD = StressBundle(
    pesticide_index=0.80,
    varroa_load=1.50,
    varroa_hive_mort_mult=2.5,
    frost_start=FROST_START,
    frost_end=FROST_END,
    label="field_triple_stress",
)
# 对照时序图：同场胁迫 + 高 μ_F
STRESS_CCD = STRESS_FIELD


def _apply_stress(sim: DifferentiableColonySimulator, stress: StressBundle) -> None:
    """把胁迫写入仿真器 buffer。"""

    sim.configure_stress(
        pesticide_index=stress.pesticide_index,
        varroa_load=stress.varroa_load,
        varroa_hive_mort_mult=stress.varroa_hive_mort_mult,
        frost_start=stress.frost_start,
        frost_end=stress.frost_end,
    )


def simulate_year(
    mu_f: float,
    stress: StressBundle = STRESS_NONE,
    days: int = 365,
) -> Dict[str, np.ndarray]:
    """无梯度前向一年；返回 NumPy 轨迹字典。"""

    sim = DifferentiableColonySimulator(
        days=days,
        dt=1.0,
        method="euler",
        forager_mortality=float(mu_f),
    )
    _apply_stress(sim, stress)
    with torch.no_grad():
        out = sim.forward()
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def summarize_trajectory(traj: Mapping[str, np.ndarray], mu_f: float) -> Dict[str, float]:
    """从轨迹提取峰、越冬、转化年龄代理。"""

    adults = np.asarray(traj["adults"], dtype=np.float64)
    recruit = np.asarray(traj["recruit_rate"], dtype=np.float64)
    doy = np.arange(1, len(adults) + 1)
    summer = (doy >= SUMMER_DOY0) & (doy <= SUMMER_DOY1)
    r_bar = float(recruit[summer].mean()) if summer.any() else float(recruit.mean())
    # 转化年龄代理：平均日转化率的倒数（日）；率越高越早熟出巢
    age_proxy = 1.0 / max(r_bar, 1.0e-8)
    return {
        "mu_F": float(mu_f),
        "n_peak_summer": float(adults[summer].max()) if summer.any() else float(adults.max()),
        "n_peak_year": float(adults.max()),
        "n_overwinter": float(adults[-1]),
        "mean_recruit_rate_summer": r_bar,
        "conversion_age_proxy_days": age_proxy,
        "n_min": float(adults.min()),
        "day_peak": float(doy[int(np.argmax(adults))]),
    }


def run_bifurcation_sweep(
    mu_min: float = MU_F_MIN,
    mu_max: float = MU_F_MAX,
    step: float = MU_F_STEP,
    stress: StressBundle | None = None,
) -> pd.DataFrame:
    r"""扫描 \(\mu_F\)，写出每点摘要行。默认挂田间三重胁迫。"""

    bundle = STRESS_FIELD if stress is None else stress
    mus = np.round(np.arange(mu_min, mu_max + 0.5 * step, step), 5)
    rows: list[Dict[str, float]] = []
    for mu in mus:
        traj = simulate_year(float(mu), stress=bundle)
        rows.append(summarize_trajectory(traj, float(mu)))
    frame = pd.DataFrame(rows)
    frame["stress_label"] = bundle.label
    return frame


def detect_tipping_point(
    frame: pd.DataFrame,
    n_collapse: float = N_COLLAPSE,
) -> Dict[str, float]:
    r"""检测相变拐点 \(\mu_F^*\)：首次跌破崩溃线，辅以最大负斜率。

    Returns:
        ``mu_star_threshold`` 为硬崩溃拐点；若无跌破则为 NaN。
        ``mu_star_slope`` 为 \(|\partial N/\partial\mu|\) 最大处。
    """

    mu = frame["mu_F"].to_numpy(dtype=np.float64)
    n_end = frame["n_overwinter"].to_numpy(dtype=np.float64)
    below = np.where(n_end < n_collapse)[0]
    mu_thr = float(mu[below[0]]) if below.size else float("nan")
    n_thr = float(n_end[below[0]]) if below.size else float("nan")
    dndmu = np.gradient(n_end, mu)
    i_slope = int(np.argmin(dndmu))
    # 软拐点：首次离开 Safe Zone（若已分区）
    mu_leave_safe = float("nan")
    if "zone" in frame.columns:
        safe = frame["zone"].to_numpy()
        leave = np.where(safe != "Safe Zone")[0]
        if leave.size:
            mu_leave_safe = float(mu[leave[0]])
    return {
        "mu_star_threshold": mu_thr,
        "n_at_threshold": n_thr,
        "mu_star_slope": float(mu[i_slope]),
        "slope_min": float(dndmu[i_slope]),
        "mu_leave_safe": mu_leave_safe,
        "n_collapse_line": float(n_collapse),
    }


def assign_zones(
    frame: pd.DataFrame,
    n_safe: float = N_SAFE,
    n_collapse: float = N_COLLAPSE,
) -> pd.DataFrame:
    """按越冬规模划分 Safe / Compensatory / Collapse。"""

    out = frame.copy()
    n = out["n_overwinter"].to_numpy(dtype=np.float64)
    zones = np.full(len(out), "Compensatory Zone", dtype=object)
    zones[n >= n_safe] = "Safe Zone"
    zones[n < n_collapse] = "Collapse Zone"
    out["zone"] = zones
    out["safe_cut"] = float(n_safe)
    out["comp_cut"] = float(n_collapse)
    return out


def plot_healthy_vs_ccd(
    healthy: Dict[str, np.ndarray],
    ccd: Dict[str, np.ndarray],
    out_path: Path | None = None,
) -> Path:
    """双子图：健康 vs CCD 时序（成蜂分室 + 子脾）。"""

    if out_path is None:
        out_path = FIGS_DIR / "fig_colony_dynamics_healthy_vs_ccd.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doy = np.arange(1, len(healthy["adults"]) + 1)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)

    def _panel(ax: Any, traj: Dict[str, np.ndarray], title: str, accent: str) -> None:
        ax.plot(doy, traj["B"], color="#c47b1e", lw=1.4, label="brood B")
        ax.plot(doy, traj["H"], color="#2c6e49", lw=1.4, label="hive H")
        ax.plot(doy, traj["F"], color="#1d4e89", lw=1.4, label="foragers F")
        ax.plot(doy, traj["adults"], color="#222222", ls="--", lw=1.1, label="adults H+F")
        ax.axvspan(FROST_START, FROST_END, color="#9ecae1", alpha=0.25, label="frost window")
        ax.set_xlabel("day of year")
        ax.set_title(title, color=accent)
        ax.set_xlim(1, 365)

    _panel(
        axes[0],
        healthy,
        rf"Healthy  ($\mu_F={MU_HEALTHY}$, no stress)",
        "#2c6e49",
    )
    _panel(
        axes[1],
        ccd,
        rf"CCD cascade  ($\mu_F={MU_CCD}$ + field triple stress)",
        "#b23a48",
    )
    axes[0].set_ylabel("bees")
    axes[0].legend(fontsize=7, loc="upper right", ncol=1)
    # 标注哺育断崖：CCD 子脾最低谷附近
    b = ccd["B"]
    i_cliff = int(np.argmin(b[120:220]) + 120) if len(b) > 220 else int(np.argmin(b))
    axes[1].annotate(
        "brood cliff /\nhive avalanche",
        xy=(doy[i_cliff], b[i_cliff]),
        xytext=(doy[i_cliff] + 25, max(b) * 0.55 + 1.0),
        fontsize=8,
        color="#b23a48",
        arrowprops={"arrowstyle": "->", "color": "#b23a48", "lw": 0.9},
    )
    fig.suptitle("Colony dynamics: healthy baseline vs CCD under triple stress", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def plot_bifurcation_curve(
    frame: pd.DataFrame,
    tip: Dict[str, float],
    out_path: Path | None = None,
) -> Path:
    """外勤死亡率 vs 越冬存活：三区着色 + 拐点标注。"""

    if out_path is None:
        out_path = FIGS_DIR / "fig_ccd_bifurcation_curve.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mu = frame["mu_F"].to_numpy(dtype=np.float64)
    n_end = frame["n_overwinter"].to_numpy(dtype=np.float64)
    n_peak = frame["n_peak_summer"].to_numpy(dtype=np.float64)
    safe_cut = float(frame["safe_cut"].iloc[0])
    comp_cut = float(frame["comp_cut"].iloc[0])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    # 背景三区
    ax.axhspan(safe_cut, max(n_end.max() * 1.05, safe_cut * 1.05), color="#d8f3dc", alpha=0.55, zorder=0)
    ax.axhspan(comp_cut, safe_cut, color="#fff3bf", alpha=0.55, zorder=0)
    ax.axhspan(0.0, comp_cut, color="#ffc9c9", alpha=0.55, zorder=0)

    ax.plot(mu, n_peak, color="#868e96", lw=1.2, ls="--", label="summer peak adults")
    ax.plot(mu, n_end, color="#1d4e89", lw=2.2, marker="o", ms=3.2, label="overwintering adults")
    ax.axhline(N_SAFE, color="#2c6e49", ls=":", lw=1.1, label=rf"$N_{{\mathrm{{safe}}}}={N_SAFE:.0f}$")
    ax.axhline(N_COLLAPSE, color="#b23a48", ls=":", lw=1.2, label=rf"$N_{{\mathrm{{collapse}}}}={N_COLLAPSE:.0f}$")

    mu_soft = tip.get("mu_star_soft", float("nan"))
    mu_star = tip["mu_star_threshold"]
    if np.isfinite(mu_soft):
        ax.axvline(mu_soft, color="#e67700", lw=1.3, ls="--", label=rf"soft tip $\mu_F^{{(s)}}={mu_soft:.3f}$")
    if np.isfinite(mu_star):
        ax.axvline(mu_star, color="#b23a48", lw=1.4, label=rf"hard tip $\mu_F^*={mu_star:.3f}$")
        ax.scatter([mu_star], [tip["n_at_threshold"]], color="#b23a48", s=48, zorder=5)

    # 区标签
    x_mid = 0.5 * (mu.min() + mu.max())
    ymax = max(float(n_end.max()), safe_cut) * 1.02
    ax.text(x_mid, ymax * 0.92, "Safe Zone", ha="center", fontsize=9, color="#2c6e49")
    ax.text(x_mid, 0.5 * (safe_cut + comp_cut), "Compensatory Zone", ha="center", fontsize=9, color="#e67700")
    ax.text(x_mid, 0.45 * comp_cut, "Collapse Zone", ha="center", fontsize=9, color="#b23a48")

    ax.set_xlabel(r"forager mortality $\mu_F$ (day$^{-1}$)")
    ax.set_ylabel("colony adult size")
    ax.set_title("CCD bifurcation: overwintering size vs forager death rate")
    ax.set_xlim(mu.min(), mu.max())
    ax.set_ylim(0.0, ymax)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path.resolve()


def run_ccd_bifurcation_pipeline() -> Tuple[pd.DataFrame, Dict[str, float]]:
    """端到端：对照轨迹 → 分岔扫 → CSV / 拐点 / 两张图。"""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    print("simulating healthy vs CCD trajectories ...")
    healthy = simulate_year(MU_HEALTHY, stress=STRESS_NONE)
    ccd = simulate_year(MU_CCD, stress=STRESS_CCD)
    fig1 = plot_healthy_vs_ccd(healthy, ccd)
    print(f"wrote {fig1}")

    print(
        f"bifurcation sweep mu_F in [{MU_F_MIN}, {MU_F_MAX}] step {MU_F_STEP} "
        f"(stress={STRESS_FIELD.label}) ..."
    )
    sweep = run_bifurcation_sweep(stress=STRESS_FIELD)
    sweep = assign_zones(sweep, n_safe=N_SAFE, n_collapse=N_COLLAPSE)
    # 软拐点：跌破 Safe；硬拐点：跌破 Collapse
    tip_soft = detect_tipping_point(sweep, n_collapse=N_SAFE)
    tip = detect_tipping_point(sweep, n_collapse=N_COLLAPSE)
    tip["mu_star_soft"] = tip_soft["mu_star_threshold"]
    tip["n_at_soft"] = tip_soft["n_at_threshold"]
    tip["n_safe_line"] = float(N_SAFE)

    csv_path = RESULTS_DIR / "ccd_bifurcation_sweep.csv"
    sweep.to_csv(csv_path, index=False)
    tip_path = RESULTS_DIR / "ccd_tipping_point.csv"
    pd.DataFrame([tip]).to_csv(tip_path, index=False)
    print(f"wrote {csv_path}")
    print(f"wrote {tip_path}")
    print(
        f"tipping soft(Safe→Comp) mu*={tip['mu_star_soft']!s}  "
        f"hard(→Collapse) mu*={tip['mu_star_threshold']!s}  "
        f"steepest-slope mu={tip['mu_star_slope']:.4f}"
    )

    # 转化年龄随 μ_F 的附列已在 sweep 中
    fig2 = plot_bifurcation_curve(sweep, tip)
    print(f"wrote {fig2}")

    # 额外：三重胁迫下短扫，写入对照表（不覆盖主分岔）
    print(f"triple-stress snapshot at mu_F={MU_CCD} ...")
    ccd_sum = summarize_trajectory(ccd, MU_CCD)
    healthy_sum = summarize_trajectory(healthy, MU_HEALTHY)
    snap = pd.DataFrame(
        [
            {**healthy_sum, "stress_label": STRESS_NONE.label},
            {**ccd_sum, "stress_label": STRESS_CCD.label},
        ]
    )
    snap_path = RESULTS_DIR / "ccd_healthy_vs_stress_snapshot.csv"
    snap.to_csv(snap_path, index=False)
    print(f"wrote {snap_path}")
    return sweep, tip


def main() -> None:
    """CLI。"""

    run_ccd_bifurcation_pipeline()


if __name__ == "__main__":
    main()

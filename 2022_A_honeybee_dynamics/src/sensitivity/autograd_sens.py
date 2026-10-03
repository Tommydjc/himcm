r"""Autograd 全参数弹性分析流水线（Req 2）。

目标
    - \(J_{\mathrm{pop}}=H(365)+F(365)\) 越冬存活成蜂
    - \(J_{\mathrm{nectar}}=N(270)\) 秋末储蜜（越冬前粮食结余）

弹性
    \[
    S_{\theta_i}=\frac{\theta_i^*}{J^*}\,\frac{\partial J}{\partial\theta_i}
    \]
    含义：参数增加 1% 时，目标相对变化约 \(S_{\theta_i}\,\%\)（弹性定义）。

输出
    - ``results/autograd_elasticity_ranking.csv``
    - ``paper_figures/fig_autograd_elasticity_bars.png``（300 DPI）
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.autograd_engine.torch_sim import DifferentiableColonySimulator

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
RESULTS_DIR: Path = PACK_ROOT / "results"
FIGS_DIR: Path = PACK_ROOT / "paper_figures"

# 轨迹下标：forward 存 t=1..365 于 index 0..364
DAY_POP: int = 365
DAY_NECTAR: int = 270
LABEL_POP: str = "J_pop_H+F_day365"
LABEL_NECTAR: str = "J_nectar_N_day270"

# 论文图用显示名
DISPLAY_NAMES: Dict[str, str] = {
    "laying_rate_max": r"$L_{\max}$ (laying rate)",
    "forager_mortality": r"$d_{F,\max}$ (forager mortality)",
    "pollen_intake_rate": r"$\eta_P$ (pollen intake)",
    "nectar_intake_rate": r"$\eta_N$ (nectar intake)",
    "social_inhibition_k": r"$\sigma_F$ (social inhibition)",
}


def _day_index(day: int) -> int:
    """年积日 → 轨迹下标（day 1 → 0）。"""

    if day < 1:
        raise ValueError(f"day must be >= 1, got {day}")
    return day - 1


def compute_elasticities(
    sim: DifferentiableColonySimulator | None = None,
) -> pd.DataFrame:
    r"""对叶子参数求 \(J_{\mathrm{pop}}\) 与 \(J_{\mathrm{nectar}}\) 的梯度与弹性。

    对 \(J_{\mathrm{pop}}\) 执行 ``backward(retain_graph=True)`` 后读 ``.grad``；
    随后 ``zero_grad`` 再对 \(J_{\mathrm{nectar}}\) 做第二次 ``backward``。

    Args:
        sim: 可微仿真器；``None`` 时新建默认 365 日欧拉实例。

    Returns:
        DataFrame，列
        ``objective, parameter, theta, J_star, dJ_dtheta, elasticity, abs_elasticity``，
        按 ``objective`` 分组后各自 ``abs_elasticity`` 降序。
    """

    if sim is None:
        sim = DifferentiableColonySimulator(days=365, dt=1.0, method="euler")

    leaves = sim.leaf_parameter_dict()
    traj = sim.forward()

    idx_pop = _day_index(DAY_POP)
    idx_nec = _day_index(DAY_NECTAR)
    # 越冬成蜂：H(365)+F(365)；与 adults[-1] 等价，显式写出以贴合规格
    j_pop = traj["H"][idx_pop] + traj["F"][idx_pop]
    j_nectar = traj["N"][idx_nec]

    rows: List[Dict[str, float | str]] = []

    # --- J_pop ---
    sim.zero_grad(set_to_none=True)
    j_pop.backward(retain_graph=True)
    j_pop_star = float(j_pop.detach())
    for name, param in leaves.items():
        if param.grad is None:
            raise RuntimeError(f"grad for {name} is None under J_pop")
        theta = float(param.detach())
        djd = float(param.grad.detach())
        elast = (theta / j_pop_star) * djd if j_pop_star != 0.0 else 0.0
        rows.append(
            {
                "objective": LABEL_POP,
                "parameter": name,
                "theta": theta,
                "J_star": j_pop_star,
                "dJ_dtheta": djd,
                "elasticity": elast,
                "abs_elasticity": abs(elast),
            }
        )

    # --- J_nectar ---
    sim.zero_grad(set_to_none=True)
    j_nectar.backward()
    j_nec_star = float(j_nectar.detach())
    for name, param in leaves.items():
        if param.grad is None:
            raise RuntimeError(f"grad for {name} is None under J_nectar")
        theta = float(param.detach())
        djd = float(param.grad.detach())
        elast = (theta / j_nec_star) * djd if j_nec_star != 0.0 else 0.0
        rows.append(
            {
                "objective": LABEL_NECTAR,
                "parameter": name,
                "theta": theta,
                "J_star": j_nec_star,
                "dJ_dtheta": djd,
                "elasticity": elast,
                "abs_elasticity": abs(elast),
            }
        )

    frame = pd.DataFrame(rows)
    frame = (
        frame.sort_values(["objective", "abs_elasticity"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return frame


def print_top_ranking(frame: pd.DataFrame, top_k: int = 5) -> None:
    """终端打印各目标弹性 |S| 前 ``top_k``，并标出正/负向第一名。"""

    for objective in frame["objective"].unique():
        sub = frame.loc[frame["objective"] == objective].copy()
        print(f"\n=== Elasticity ranking | {objective}  (J*={sub['J_star'].iloc[0]:.4g}) ===")
        show = sub.head(top_k)
        for i, row in enumerate(show.itertuples(index=False), start=1):
            sign = "+" if row.elasticity >= 0 else ""
            print(
                f"  #{i}  {row.parameter:22s}  S={sign}{row.elasticity: .6f}  "
                f"∂J/∂θ={row.dJ_dtheta: .4g}  θ={row.theta:g}"
            )
        pos = sub.loc[sub["elasticity"] > 0]
        neg = sub.loc[sub["elasticity"] < 0]
        if len(pos):
            best_pos = pos.loc[pos["elasticity"].idxmax()]
            print(
                f"  >> top positive driver : {best_pos['parameter']}  "
                f"S={best_pos['elasticity']:+.6f}"
            )
        else:
            print("  >> top positive driver : (none)")
        if len(neg):
            best_neg = neg.loc[neg["elasticity"].idxmin()]
            print(
                f"  >> top negative fragility: {best_neg['parameter']}  "
                f"S={best_neg['elasticity']:+.6f}"
            )
        else:
            print("  >> top negative fragility: (none)")


def plot_elasticity_bars(
    frame: pd.DataFrame,
    out_path: Path | None = None,
    objective: str = LABEL_POP,
) -> Path:
    r"""横向正负双色条形图：参数对蜂群规模 \(J_{\mathrm{pop}}\) 的弹性。

    Args:
        frame: ``compute_elasticities`` 输出。
        out_path: 保存路径；默认 ``paper_figures/fig_autograd_elasticity_bars.png``。
        objective: 绘图目标；默认越冬成蜂。

    Returns:
        写出文件的绝对路径。
    """

    if out_path is None:
        out_path = FIGS_DIR / "fig_autograd_elasticity_bars.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sub = frame.loc[frame["objective"] == objective].copy()
    if sub.empty:
        raise ValueError(f"no rows for objective={objective!r}")
    sub = sub.sort_values("elasticity")
    labels = [DISPLAY_NAMES.get(str(p), str(p)) for p in sub["parameter"]]
    values = sub["elasticity"].to_numpy(dtype=np.float64)
    colors = ["#b23a48" if v < 0 else "#1d4e89" for v in values]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.barh(labels, values, color=colors, height=0.62)
    ax.axvline(0.0, color="black", lw=0.9)
    ax.set_xlabel(
        r"elasticity $S_\theta=(\theta/J)\,(\partial J/\partial\theta)$"
        "\n(1% parameter rise → ≈S% change in overwintering adults)"
    )
    j_star = float(sub["J_star"].iloc[0])
    ax.set_title(
        rf"Autograd elasticities for $J_{{\mathrm{{pop}}}}=H(365)+F(365)$"
        f"\n$J^*={j_star:.1f}$ adults; red = fragility, blue = driver"
    )
    # 标出最脆弱（最负）参数
    i_min = int(np.argmin(values))
    ax.annotate(
        "top fragility",
        xy=(values[i_min], i_min),
        xytext=(values[i_min] - 0.02 * (abs(values).max() + 1e-9), i_min + 0.35),
        fontsize=8,
        color="#b23a48",
        arrowprops={"arrowstyle": "->", "color": "#b23a48", "lw": 0.8},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path.resolve()


def run_elasticity_pipeline(
    csv_path: Path | None = None,
    fig_path: Path | None = None,
) -> pd.DataFrame:
    """端到端：求导 → 排序 CSV → 打印 Top-5 → 画条形图。

    Args:
        csv_path: 默认 ``results/autograd_elasticity_ranking.csv``。
        fig_path: 默认 ``paper_figures/fig_autograd_elasticity_bars.png``。

    Returns:
        完整弹性表（含两个目标）。
    """

    if csv_path is None:
        csv_path = RESULTS_DIR / "autograd_elasticity_ranking.csv"
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    frame = compute_elasticities()
    # 主排行榜：按 |S| 全局降序，但保留 objective 列；另写 pop 优先块
    frame.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")
    print_top_ranking(frame, top_k=5)
    fig = plot_elasticity_bars(frame, out_path=fig_path, objective=LABEL_POP)
    print(f"wrote {fig}")
    return frame


def main() -> None:
    """CLI 入口。"""

    run_elasticity_pipeline()


if __name__ == "__main__":
    main()

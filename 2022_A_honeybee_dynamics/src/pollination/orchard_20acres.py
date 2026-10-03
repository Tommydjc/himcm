r"""20 英亩商业扁桃园授粉供需匹配与蜂箱最优配比（Req 3 / Section 5 风格）。

场景（USDA / 加州杏仁协会数量级情景，非台站普查）
    - 面积 20 acre \(\approx 8.1\,\mathrm{ha}=81\,000\,\mathrm{m}^2\)
    - 110 株/acre，株花约 25 000 朵（花期 2–3 月情景）

竞争
    \[
    \mathrm{DailyFoodPerHive}(K)
    =\min\Bigl(C_{\max},\; N_{\mathrm{orchard}}/K\Bigr)
    \]

结实
    \[
    \mathrm{YieldRatio}(V)=\frac{V}{V+V_{50}}
    \]

净收益
    \[
    \Pi(K)=R_{\mathrm{full}}\cdot\mathrm{YieldRatio}(V)
    -c_{\mathrm{rent}}K
    \]

单箱最大采集能力
    \[
    C_{\max}=F_{\mathrm{pkg}}\cdot\eta_N
    \]
其中 \(\eta_N\) 取自 ``DifferentiableColonySimulator`` 叶子参数；
\(F_{\mathrm{pkg}}\) 为商业授粉群外勤规模（强群进园情景）。
田块日蜜源按“在约 2 箱/英亩处开始稀释”标定，使黄金带内出现收益–健康权衡。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.autograd_engine.torch_sim import DifferentiableColonySimulator

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
RESULTS_DIR: Path = PACK_ROOT / "results"
FIGS_DIR: Path = PACK_ROOT / "paper_figures"

# --- USDA / 扩展文献量级情景 ---
ACRES: float = 20.0
AREA_M2: float = 81000.0  # 官方 20 acre
TREES_PER_ACRE: float = 110.0
FLOWERS_PER_TREE: float = 25000.0
BLOOM_DAYS: int = 35
BLOOM_DOY_START: int = 100
BLOOM_DOY_END: int = 134  # inclusive span ≈ 35 d
# 商业授粉群外勤规模（进园强群情景）
COMMERCIAL_FORAGERS: float = 10000.0
# 稀释拐点：约 2.1 箱/英亩（42 箱）时 DailyFood 开始 < C_max
DILUTION_ONSET_K: float = 42.0
# 有效授粉访花（朵/外勤·日）；远小于题面 2000 总访花
EFF_VISITS_PER_FORAGER_DAY: float = 55.0
V50_VISITS_PER_FLOWER: float = 2.5
# 经济：充分授粉时全园坚果产值（USD）；租箱费
YIELD_VALUE_FULL_USD: float = 70_000.0  # ≈ $3500/acre × 20
RENT_USD_PER_HIVE: float = 200.0
K_MIN: int = 5
K_MAX: int = 60
GOLDEN_K_LO: int = 20
GOLDEN_K_HI: int = 40


@dataclass(frozen=True, slots=True)
class OrchardParams:
    r"""20 英亩扁桃园契约。

    Attributes
    ----------
    acres, area_m2 :
        地块面积。
    trees_per_acre, flowers_per_tree :
        栽植与花量情景。
    bloom_days :
        花期长度（日）。
    commercial_foragers :
        进园强群外勤数 \(F_{\mathrm{pkg}}\)。
    dilution_onset_k :
        竞争稀释起始箱数（标定 \(N_{\mathrm{orchard}}\)）。
    eff_visits_per_forager_day :
        落在本园的有效授粉访花（朵/外勤·日）。
    v50 :
        Michaelis–Menten 半饱和访花次数（次/花）。
    yield_value_full_usd :
        YieldRatio=1 时全园产值（USD）。
    rent_usd_per_hive :
        租箱费（USD/箱）。
    """

    acres: float = ACRES
    area_m2: float = AREA_M2
    trees_per_acre: float = TREES_PER_ACRE
    flowers_per_tree: float = FLOWERS_PER_TREE
    bloom_days: int = BLOOM_DAYS
    commercial_foragers: float = COMMERCIAL_FORAGERS
    dilution_onset_k: float = DILUTION_ONSET_K
    eff_visits_per_forager_day: float = EFF_VISITS_PER_FORAGER_DAY
    v50: float = V50_VISITS_PER_FLOWER
    yield_value_full_usd: float = YIELD_VALUE_FULL_USD
    rent_usd_per_hive: float = RENT_USD_PER_HIVE

    @property
    def n_trees(self) -> float:
        """全园株数。"""

        return self.acres * self.trees_per_acre

    @property
    def n_flowers(self) -> float:
        """全园有效花总数。"""

        return self.n_trees * self.flowers_per_tree

    def orchard_daily_nectar_g(self, max_foraging_capacity_g_day: float) -> float:
        r"""田块日蜜源 \(N_{\mathrm{orchard}}=K_{\mathrm{dilute}}\cdot C_{\max}\)（g/day）。"""

        return float(self.dilution_onset_k) * float(max_foraging_capacity_g_day)

    def nectar_g_per_flower_lifetime(self, max_foraging_capacity_g_day: float) -> float:
        """由日蜜源反推的单花一生蜜当量（g/花，含花粉能量折算）。"""

        daily = self.orchard_daily_nectar_g(max_foraging_capacity_g_day)
        return daily * float(self.bloom_days) / self.n_flowers


def calibrate_max_foraging_capacity(
    commercial_foragers: float = COMMERCIAL_FORAGERS,
    bloom_start: int = BLOOM_DOY_START,
    bloom_end: int = BLOOM_DOY_END,
) -> Dict[str, float]:
    r"""用 ``torch_sim`` 的 \(\eta_N\) 与商业群 \(F_{\mathrm{pkg}}\) 标定 \(C_{\max}\)。

    同时记录花期窗内仿真外勤均值（对照，不直接用作产能，因早春 \(\Omega(t)\) 偏低）。

    Returns:
        含 ``max_foraging_capacity_g_day``、``eta_N``、``mean_F_bloom_sim`` 等。
    """

    sim = DifferentiableColonySimulator(days=365, dt=1.0, method="euler")
    with torch.no_grad():
        out = sim.forward()
    i0 = bloom_start - 1
    i1 = bloom_end
    mean_f_sim = float(out["F"][i0:i1].mean().item())
    eta_n = float(sim.nectar_intake_rate.detach())
    f_pkg = float(commercial_foragers)
    c_max = f_pkg * eta_n
    return {
        "max_foraging_capacity_g_day": c_max,
        "commercial_foragers": f_pkg,
        "mean_F_bloom_sim": mean_f_sim,
        "eta_N": eta_n,
        "bloom_start": float(bloom_start),
        "bloom_end": float(bloom_end),
    }


def daily_food_per_hive(
    k: int,
    orchard_daily_nectar_g: float,
    max_foraging_capacity_g_day: float,
) -> float:
    r"""竞争稀释后的单箱日粮 \(\min(C_{\max},\, N_{\mathrm{orchard}}/K)\)。"""

    if k <= 0:
        raise ValueError("K must be positive")
    return float(min(max_foraging_capacity_g_day, orchard_daily_nectar_g / float(k)))


def yield_ratio(visits_per_flower: float, v50: float) -> float:
    r"""Michaelis–Menten 结实率 \(V/(V+V_{50})\)。"""

    v = max(0.0, float(visits_per_flower))
    return v / (v + float(v50))


def hive_health_index(food_ratio: float) -> Dict[str, float]:
    r"""花期后蜂群健康代理：储蜜指数与幼虫存活率。

    Args:
        food_ratio:
            ``DailyFoodPerHive / C_max``，取值 \((0,1]\)。

    Returns:
        ``nectar_store_index``, ``brood_survival``, ``hive_health`` \(\in[0,1]\)。
    """

    r = float(np.clip(food_ratio, 0.0, 1.0))
    nectar_store = r
    brood_survival = 0.35 + 0.60 * r
    health = 0.5 * nectar_store + 0.5 * brood_survival
    return {
        "nectar_store_index": nectar_store,
        "brood_survival": brood_survival,
        "hive_health": health,
    }


def evaluate_stocking(
    k: int,
    params: OrchardParams,
    mean_f_pkg: float,
    max_foraging_capacity_g_day: float,
    orchard_daily_nectar_g: float,
) -> Dict[str, float]:
    r"""单个蜂箱密度 \(K\) 下的访花、产量、收益与健康。"""

    food = daily_food_per_hive(
        k, orchard_daily_nectar_g, max_foraging_capacity_g_day
    )
    food_ratio = food / max(max_foraging_capacity_g_day, 1.0e-12)
    visits_day = (
        float(k) * mean_f_pkg * params.eff_visits_per_forager_day * food_ratio
    )
    visits_per_flower = (visits_day * float(params.bloom_days)) / params.n_flowers
    yld = yield_ratio(visits_per_flower, params.v50)
    gross = params.yield_value_full_usd * yld
    rent = params.rent_usd_per_hive * float(k)
    net = gross - rent
    health = hive_health_index(food_ratio)
    return {
        "K": float(k),
        "hives_per_acre": float(k) / params.acres,
        "daily_food_per_hive_g": food,
        "food_ratio": food_ratio,
        "visits_per_flower": visits_per_flower,
        "yield_ratio": yld,
        "gross_usd": gross,
        "rent_usd": rent,
        "net_profit_usd": net,
        **health,
    }


def sweep_stocking(
    params: OrchardParams | None = None,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    r"""扫描 \(K\in[k_{\min},k_{\max}]\)，返回表与标定字典。"""

    if params is None:
        params = OrchardParams()
    cal = calibrate_max_foraging_capacity(
        commercial_foragers=params.commercial_foragers
    )
    c_max = cal["max_foraging_capacity_g_day"]
    nectar_daily = params.orchard_daily_nectar_g(c_max)
    nectar_flower = params.nectar_g_per_flower_lifetime(c_max)
    cal["orchard_daily_nectar_g"] = nectar_daily
    cal["nectar_g_per_flower_lifetime"] = nectar_flower
    rows = [
        evaluate_stocking(
            k,
            params,
            mean_f_pkg=cal["commercial_foragers"],
            max_foraging_capacity_g_day=c_max,
            orchard_daily_nectar_g=nectar_daily,
        )
        for k in range(int(k_min), int(k_max) + 1)
    ]
    frame = pd.DataFrame(rows)
    frame["acres"] = params.acres
    frame["area_m2"] = params.area_m2
    frame["n_trees"] = params.n_trees
    frame["n_flowers"] = params.n_flowers
    frame["orchard_daily_nectar_g"] = nectar_daily
    frame["nectar_g_per_flower_lifetime"] = nectar_flower
    frame["max_foraging_capacity_g_day"] = c_max
    frame["commercial_foragers"] = cal["commercial_foragers"]
    frame["mean_F_bloom_sim"] = cal["mean_F_bloom_sim"]
    frame["rent_usd_per_hive"] = params.rent_usd_per_hive
    frame["in_golden_band"] = (frame["K"] >= GOLDEN_K_LO) & (frame["K"] <= GOLDEN_K_HI)
    return frame, cal


def recommend_stocking(frame: pd.DataFrame) -> Dict[str, float]:
    r"""在黄金带内取净收益最大的 \(K^\star\)；并报告全局最优作对照。"""

    golden = frame.loc[frame["in_golden_band"]]
    if golden.empty:
        raise ValueError("golden band is empty")
    i_g = int(golden["net_profit_usd"].idxmax())
    i_all = int(frame["net_profit_usd"].idxmax())
    return {
        "K_star_golden": float(frame.loc[i_g, "K"]),
        "net_star_golden": float(frame.loc[i_g, "net_profit_usd"]),
        "health_at_golden": float(frame.loc[i_g, "hive_health"]),
        "yield_at_golden": float(frame.loc[i_g, "yield_ratio"]),
        "K_star_global": float(frame.loc[i_all, "K"]),
        "net_star_global": float(frame.loc[i_all, "net_profit_usd"]),
    }


def plot_tradeoff(
    frame: pd.DataFrame,
    rec: Dict[str, float],
    out_path: Path | None = None,
) -> Path:
    """蜂箱密度权衡图：净收益 + 结实率 + 蜂群健康，标注 20–40 箱黄金带。"""

    if out_path is None:
        out_path = FIGS_DIR / "fig_20acre_hive_density_tradeoff.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    k = frame["K"].to_numpy(dtype=np.float64)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax1 = plt.subplots(figsize=(9.2, 5.0))
    ax1.axvspan(
        GOLDEN_K_LO,
        GOLDEN_K_HI,
        color="#d8f3dc",
        alpha=0.65,
        zorder=0,
        label="golden band 1–2 hives/acre (20–40)",
    )
    ln1 = ax1.plot(
        k,
        frame["net_profit_usd"] / 1000.0,
        color="#1d4e89",
        lw=2.2,
        label="net profit (thousand USD)",
    )
    ax1.set_xlabel(r"number of hives $K$ on 20 acres")
    ax1.set_ylabel("net profit (thousand USD)", color="#1d4e89")
    ax1.tick_params(axis="y", labelcolor="#1d4e89")

    ax2 = ax1.twinx()
    ln2 = ax2.plot(
        k,
        frame["yield_ratio"],
        color="#2c6e49",
        lw=1.6,
        ls="--",
        label="yield ratio",
    )
    ln3 = ax2.plot(
        k,
        frame["hive_health"],
        color="#b23a48",
        lw=1.6,
        ls=":",
        label="hive health index",
    )
    ax2.set_ylabel("yield ratio / hive health", color="#333333")
    ax2.set_ylim(0.0, 1.05)

    k_star = rec["K_star_golden"]
    y_star = rec["net_star_golden"] / 1000.0
    ax1.axvline(k_star, color="#2c6e49", lw=1.3, alpha=0.9)
    ax1.scatter([k_star], [y_star], color="#2c6e49", s=56, zorder=5)
    y_lo, y_hi = ax1.get_ylim()
    ax1.annotate(
        rf"$K^\star={k_star:.0f}$ in golden band"
        f"\nnet=${rec['net_star_golden']:,.0f}",
        xy=(k_star, y_star),
        xytext=(min(k_star + 5.0, float(k.max()) - 8.0), y_star - 0.12 * (y_hi - y_lo)),
        fontsize=8,
        color="#2c6e49",
        arrowprops={"arrowstyle": "->", "color": "#2c6e49", "lw": 0.8},
    )
    mid_y = y_lo + 0.08 * (y_hi - y_lo)
    ax1.text(
        0.5 * (GOLDEN_K_LO + GOLDEN_K_HI),
        mid_y,
        "1–2 hives / acre",
        ha="center",
        fontsize=9,
        color="#2c6e49",
        fontweight="bold",
    )

    from matplotlib.patches import Patch

    handles = ln1 + ln2 + ln3 + [
        Patch(
            facecolor="#d8f3dc",
            edgecolor="none",
            alpha=0.65,
            label="golden band 1–2 hives/acre",
        )
    ]
    ax1.legend(handles, [h.get_label() for h in handles], fontsize=8, loc="center left")
    ax1.set_title(
        "20-acre almond orchard: hive density tradeoff\n"
        r"(competition dilution + Michaelis–Menten yield $-$ \$200/hive rent)"
    )
    ax1.set_xlim(float(k.min()), float(k.max()))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path.resolve()


def run_orchard_optimization(
    csv_path: Path | None = None,
    fig_path: Path | None = None,
    recommendation_path: Path | None = None,
) -> pd.DataFrame:
    """端到端：标定 → 扫描 → CSV → 黄金带推荐 → 出图。"""

    if csv_path is None:
        csv_path = RESULTS_DIR / "orchard_pollination_optimization.csv"
    if recommendation_path is None:
        recommendation_path = RESULTS_DIR / "orchard_pollination_recommendation.csv"
    csv_path = Path(csv_path)
    recommendation_path = Path(recommendation_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation_path.parent.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    params = OrchardParams()
    frame, cal = sweep_stocking(params)
    rec = recommend_stocking(frame)
    frame.to_csv(csv_path, index=False)

    meta = {**asdict(params), **cal, **rec}
    pd.DataFrame([meta]).to_csv(recommendation_path, index=False)

    fig = plot_tradeoff(frame, rec, out_path=fig_path)
    print(f"wrote {csv_path}")
    print(f"wrote {recommendation_path}")
    print(f"wrote {fig}")
    print(
        f"orchard nectar={cal['orchard_daily_nectar_g']:.1f} g/day  "
        f"C_max={cal['max_foraging_capacity_g_day']:.1f} g/day  "
        f"F_pkg={cal['commercial_foragers']:.0f}  "
        f"eta_N={cal['eta_N']:.3f}"
    )
    print(
        f"recommend golden K*={rec['K_star_golden']:.0f}  "
        f"net=${rec['net_star_golden']:,.0f}  "
        f"yield={rec['yield_at_golden']:.3f}  "
        f"health={rec['health_at_golden']:.3f}  "
        f"(global K*={rec['K_star_global']:.0f})"
    )
    return frame


def main() -> None:
    """CLI。"""

    run_orchard_optimization()


if __name__ == "__main__":
    main()

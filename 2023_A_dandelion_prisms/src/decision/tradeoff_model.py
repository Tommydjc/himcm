#!/usr/bin/env python3
"""PRISMS 风格日步生物经济权衡（课程 Req 2/3 扩展，非官方 SAW）。

.. math::

    B_{\\mathrm{eco}}=B_{\\mathrm{pollinator}}+B_{\\mathrm{soil}},
    \\qquad
    B_{\\mathrm{pollinator}}=\\alpha\\sum_t F(t)\\,\\mathbf{1}_{t\\in\\mathrm{EarlySpring}},
    \\qquad
    B_{\\mathrm{soil}}=\\beta\\,\\overline{\\mathrm{Biomass}},
    \\qquad
    C_{\\mathrm{turf}}=\\gamma\\,(\\mathrm{cover}_{365})^{1.3},
    \\qquad
    C_{\\mathrm{cost}}=C_{\\mathrm{turf}}+c_m n(1+\\eta/2)+c_h n_{\\mathrm{spray}}.

\\(\\alpha,\\beta,\\gamma,c_m,c_h\\) 是相对权，不是市场美元。
四策略在日步 ``grid_engine`` 上积分；图只从写出的 CSV 读取。

不覆盖周步论文表 ``results/bioeconomic_pareto.csv``
（\\(C=n(1+\\eta/2)\\) 网格，见 ``pareto.py``）。
本模块写入 ``results/bioeconomic_tradeoff.csv``。

运行::

    PYTHONPATH=. python -m src.decision.tradeoff_model
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from src.biology.phenology import PhenologyParams, StageState
from src.simulation.grid_engine import DailyForcing, load_daily_forcing, run_hectare_days

OCCUPY_THRESH: float = 0.05
EARLY_SPRING_MONTHS: frozenset[int] = frozenset({3, 4})
GROWING_MONTHS: frozenset[int] = frozenset({4, 5, 6, 7, 8, 9, 10})
EFFECT_SNAP_DAYS: tuple[int, ...] = (90, 120, 180, 365)
WONG_BLUE = "#0072B2"
WONG_GREEN = "#009E73"
WONG_ORANGE = "#E69F00"
WONG_VERM = "#D55E00"


def month_of_doy(doy: int) -> int:
    """非闰年年积日 0–364 → 月 1–12。"""
    month_ends = (31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)
    day = doy + 1
    for month, end in enumerate(month_ends, start=1):
        if day <= end:
            return month
    return 12


@dataclass(frozen=True, slots=True)
class EcoWeights:
    """相对生态经济权（无货币单位）。

    Parameters
    ----------
    alpha, beta, gamma :
        传粉、土壤、草坪损失权。
    c_mow, c_herb :
        单次机械有效成本系数、单次 2,4-D 喷施系数。
    turf_exp :
        草坪损失指数，默认 1.3。
    """

    alpha: float = 1.00
    beta: float = 0.60
    gamma: float = 1.20
    c_mow: float = 0.025
    c_herb: float = 0.120
    turf_exp: float = 1.30


@dataclass
class ActionLedger:
    """作业计数。"""

    n_mows: int = 0
    n_sprays: int = 0
    eta_mow: float = 0.0
    eta_chem: float = 0.0


@dataclass
class DailyTrace:
    """逐日花朵、生物量、覆盖。"""

    flowers: NDArray[np.float64]
    biomass: NDArray[np.float64]
    cover: NDArray[np.float64]
    n_plants: NDArray[np.float64]


@dataclass
class StrategyScore:
    """一条治理策略的目标与分解。"""

    name: str
    label: str
    B_pollinator: float
    B_soil: float
    B_eco: float
    C_turf: float
    C_mow: float
    C_herb: float
    C_cost: float
    net: float
    cover_365: float
    n_plants_365: float
    n_adult_365: float
    n_mows: int
    n_sprays: int
    on_front: bool = False


def apply_mechanical(state: StageState, eta: float) -> StageState:
    """低底盘机械修剪：成株强减，莲座部分耐踩，种子库不动。"""
    keep = 1.0 - min(max(float(eta), 0.0), 1.0)
    return StageState(
        seed=state.seed,
        seedling=state.seedling * (1.0 - 0.20 * (1.0 - keep)),
        rosette=state.rosette * (1.0 - 0.45 * (1.0 - keep)),
        adult=state.adult * keep,
    )


def apply_chemical(state: StageState, eta: float) -> StageState:
    """2,4-D 叶面传导：成株/莲座/幼苗下降，种子库残留。"""
    keep = 1.0 - min(max(float(eta), 0.0), 1.0)
    return StageState(
        seed=state.seed,
        seedling=state.seedling * (1.0 - 0.75 * (1.0 - keep)),
        rosette=state.rosette * (1.0 - 0.90 * (1.0 - keep)),
        adult=state.adult * keep,
    )


def first_true_bloom_day(forcings: list[DailyForcing]) -> int:
    """第一个非 day-0 强迫释放日（早春花期起点）。"""
    for row in forcings:
        if row.releasing and row.day > 0:
            return int(row.day)
    return 90


def prisms_mow_day(forcings: list[DailyForcing], bloom_keep_days: int = 14, pre_seed_days: int = 3) -> int:
    r"""花托飞散前 ``pre_seed_days`` 日修剪：\(t^*=t_{\mathrm{bloom}}+14-3\)。"""
    t_bloom = first_true_bloom_day(forcings)
    return int(t_bloom + bloom_keep_days - pre_seed_days)


def make_wild_policy() -> tuple[ActionLedger, object]:
    """策略 1：不干预。"""
    ledger = ActionLedger()

    def intervene(_forcing: DailyForcing, state: StageState) -> StageState:
        return state

    return ledger, intervene


def make_weekly_mow_policy(eta: float = 0.85) -> tuple[ActionLedger, object]:
    """策略 2：每 7 日机械修剪。"""
    ledger = ActionLedger(eta_mow=eta)

    def intervene(forcing: DailyForcing, state: StageState) -> StageState:
        if forcing.day > 0 and forcing.day % 7 == 0:
            ledger.n_mows += 1
            return apply_mechanical(state, eta)
        return state

    return ledger, intervene


def make_chemical_policy(eta: float = 0.90, period_days: int = 21) -> tuple[ActionLedger, object]:
    """策略 3：生长季定期 2,4-D。"""
    ledger = ActionLedger(eta_chem=eta)

    def intervene(forcing: DailyForcing, state: StageState) -> StageState:
        month = month_of_doy(forcing.day)
        if month in GROWING_MONTHS and forcing.day % period_days == 0:
            ledger.n_sprays += 1
            return apply_chemical(state, eta)
        return state

    return ledger, intervene


def make_prisms_policy(
    forcings: list[DailyForcing],
    eta: float = 0.95,
) -> tuple[ActionLedger, object]:
    """策略 4：早春留花 14 日，飞散前 3 日单次低底盘修剪。"""
    ledger = ActionLedger(eta_mow=eta)
    t_cut = prisms_mow_day(forcings)

    def intervene(forcing: DailyForcing, state: StageState) -> StageState:
        if forcing.day == t_cut:
            ledger.n_mows += 1
            return apply_mechanical(state, eta)
        return state

    return ledger, intervene


def _cover_frac(state: StageState) -> float:
    plants = state.seedling_total + state.rosette + state.adult
    return float(np.mean(plants >= OCCUPY_THRESH))


def simulate_strategy(
    name: str,
    label: str,
    forcings: list[DailyForcing],
    ledger: ActionLedger,
    intervene: object,
    params: PhenologyParams,
    weights: EcoWeights,
    snap_dir: Path | None = None,
    snap_days: tuple[int, ...] = EFFECT_SNAP_DAYS,
) -> tuple[StrategyScore, DailyTrace]:
    """跑全年并按公式计分。可选写入建成株空间截面。"""
    n = len(forcings)
    flowers = np.zeros(n, dtype=np.float64)
    biomass = np.zeros(n, dtype=np.float64)
    cover = np.zeros(n, dtype=np.float64)
    n_plants = np.zeros(n, dtype=np.float64)
    wanted = set(snap_days)

    def on_day(day: int, state: StageState) -> None:
        idx = min(max(int(day), 0), n - 1)
        releasing = forcings[idx].releasing
        flowers[idx] = float(np.sum(state.adult)) if releasing else 0.0
        biomass[idx] = float(np.sum(state.rosette + state.adult))
        cover[idx] = _cover_frac(state)
        plants = state.seedling_total + state.rosette + state.adult
        n_plants[idx] = float(np.sum(plants))
        elapsed = int(day) + 1
        if snap_dir is not None and elapsed in wanted:
            np.save(snap_dir / f"tradeoff_{name}_day{elapsed:03d}.npy", plants)

    result = run_hectare_days(forcings, params=params, on_day=on_day, intervene=intervene)
    spring = np.array(
        [month_of_doy(row.day) in EARLY_SPRING_MONTHS for row in forcings],
        dtype=bool,
    )
    poll = float(np.sum(flowers[spring]))
    soil = float(np.mean(biomass))
    cover_365 = float(cover[-1])
    b_pol = weights.alpha * poll
    b_soil = weights.beta * soil
    b_eco = b_pol + b_soil
    c_turf = weights.gamma * (max(cover_365, 0.0) ** weights.turf_exp)
    c_mow = weights.c_mow * float(ledger.n_mows) * (1.0 + 0.5 * ledger.eta_mow)
    c_herb = weights.c_herb * float(ledger.n_sprays)
    c_cost = c_turf + c_mow + c_herb
    score = StrategyScore(
        name=name,
        label=label,
        B_pollinator=b_pol,
        B_soil=b_soil,
        B_eco=b_eco,
        C_turf=c_turf,
        C_mow=c_mow,
        C_herb=c_herb,
        C_cost=c_cost,
        net=b_eco - c_cost,
        cover_365=cover_365,
        n_plants_365=float(n_plants[-1]),
        n_adult_365=float(result.n_adult[-1]),
        n_mows=ledger.n_mows,
        n_sprays=ledger.n_sprays,
    )
    trace = DailyTrace(flowers=flowers, biomass=biomass, cover=cover, n_plants=n_plants)
    return score, trace


def nondominated_benefit_cost(scores: list[StrategyScore]) -> list[StrategyScore]:
    r"""最大化 \(B_{eco}\)、最小化 \(C_{cost}\) 的非支配集。"""
    front: list[StrategyScore] = []
    for cand in scores:
        dominated = False
        for other in scores:
            if other is cand:
                continue
            le = other.B_eco >= cand.B_eco - 1.0e-12 and other.C_cost <= cand.C_cost + 1.0e-12
            sl = other.B_eco > cand.B_eco + 1.0e-12 or other.C_cost < cand.C_cost - 1.0e-12
            if le and sl:
                dominated = True
                break
        if not dominated:
            front.append(cand)
    return sorted(front, key=lambda s: (s.C_cost, -s.B_eco))


def _rescale_benefits_to_wild(scores: list[StrategyScore], weights: EcoWeights) -> None:
    r"""把传粉/生物量除以野化基线，使 \(B\) 与 \(C_{turf}\) 同量级。"""
    wild = next((item for item in scores if item.name == "wild"), None)
    if wild is None:
        return
    poll0 = max(wild.B_pollinator / max(weights.alpha, 1.0e-12), 1.0e-12)
    soil0 = max(wild.B_soil / max(weights.beta, 1.0e-12), 1.0e-12)
    for item in scores:
        raw_pol = item.B_pollinator / max(weights.alpha, 1.0e-12)
        raw_soil = item.B_soil / max(weights.beta, 1.0e-12)
        item.B_pollinator = weights.alpha * raw_pol / poll0
        item.B_soil = weights.beta * raw_soil / soil0
        item.B_eco = item.B_pollinator + item.B_soil
        item.net = item.B_eco - item.C_cost


def run_four_strategies(
    climate: str = "temperate",
    weights: EcoWeights | None = None,
    params: PhenologyParams | None = None,
) -> list[StrategyScore]:
    """积分四种治理策略。"""
    life = params if params is not None else PhenologyParams()
    eco = weights if weights is not None else EcoWeights()
    forcings = load_daily_forcing(climate, n_days=365)
    policies: list[tuple[str, str, ActionLedger, object]] = [
        ("wild", "S1 natural", *make_wild_policy()),
        ("weekly_mow", "S2 weekly mow", *make_weekly_mow_policy()),
        ("chemical", "S3 herbicide", *make_chemical_policy()),
        ("prisms", "S4 PRISMS window", *make_prisms_policy(forcings)),
    ]
    scores: list[StrategyScore] = []
    for name, label, ledger, intervene in policies:
        score, _trace = simulate_strategy(name, label, forcings, ledger, intervene, life, eco)
        scores.append(score)
    _rescale_benefits_to_wild(scores, eco)
    front = {(item.name, item.C_cost, item.B_eco) for item in nondominated_benefit_cost(scores)}
    for item in scores:
        item.on_front = (item.name, item.C_cost, item.B_eco) in front
    return scores


def export_strategy_artifacts(
    climate: str = "temperate",
    weights: EcoWeights | None = None,
    params: PhenologyParams | None = None,
    results_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """积分四策略，写得分、逐日轨迹与空间截面。返回（得分表，轨迹表）。"""
    life = params if params is not None else PhenologyParams()
    eco = weights if weights is not None else EcoWeights()
    out_dir = results_dir or (PACK_ROOT / "results")
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    forcings = load_daily_forcing(climate, n_days=365)
    policies: list[tuple[str, str, ActionLedger, object]] = [
        ("wild", "S1 natural", *make_wild_policy()),
        ("weekly_mow", "S2 weekly mow", *make_weekly_mow_policy()),
        ("chemical", "S3 herbicide", *make_chemical_policy()),
        ("prisms", "S4 PRISMS window", *make_prisms_policy(forcings)),
    ]
    scores: list[StrategyScore] = []
    traces: list[DailyTrace] = []
    for name, label, ledger, intervene in policies:
        score, trace = simulate_strategy(
            name,
            label,
            forcings,
            ledger,
            intervene,
            life,
            eco,
            snap_dir=snap_dir,
        )
        scores.append(score)
        traces.append(trace)
    _rescale_benefits_to_wild(scores, eco)
    front = {(item.name, item.C_cost, item.B_eco) for item in nondominated_benefit_cost(scores)}
    for item in scores:
        item.on_front = (item.name, item.C_cost, item.B_eco) in front
    score_frame = scores_to_frame(scores)
    rows: list[dict[str, object]] = []
    for score, trace in zip(scores, traces):
        for i in range(trace.cover.size):
            rows.append(
                {
                    "strategy": score.name,
                    "label": score.label,
                    "day": i + 1,
                    "cover": float(trace.cover[i]),
                    "flowers": float(trace.flowers[i]),
                    "biomass": float(trace.biomass[i]),
                    "n_plants": float(trace.n_plants[i]),
                }
            )
    trace_frame = pd.DataFrame(rows)
    score_frame.to_csv(out_dir / "bioeconomic_tradeoff.csv", index=False)
    trace_frame.to_csv(out_dir / "tradeoff_daily_traces.csv", index=False)
    return score_frame, trace_frame


def scores_to_frame(scores: list[StrategyScore]) -> pd.DataFrame:
    """结构化报表。"""
    return pd.DataFrame(
        [
            {
                "strategy": item.name,
                "label": item.label,
                "B_pollinator": item.B_pollinator,
                "B_soil": item.B_soil,
                "B_eco": item.B_eco,
                "C_turf": item.C_turf,
                "C_mow": item.C_mow,
                "C_herb": item.C_herb,
                "C_cost": item.C_cost,
                "net": item.net,
                "cover_365": item.cover_365,
                "n_plants_365": item.n_plants_365,
                "n_adult_365": item.n_adult_365,
                "n_mows": item.n_mows,
                "n_sprays": item.n_sprays,
                "on_front": item.on_front,
            }
            for item in scores
        ]
    )


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


def _minmax_col(values: NDArray[np.float64]) -> NDArray[np.float64]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if abs(hi - lo) < 1.0e-12:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def plot_tradeoff(table: pd.DataFrame, fig_dir: Path) -> Path:
    """左：帕累托散点；右：五轴雷达（高=更好）。"""
    fig, (ax_sc, ax_rd) = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    colors = {
        "wild": WONG_GREEN,
        "weekly_mow": WONG_ORANGE,
        "chemical": WONG_VERM,
        "prisms": WONG_BLUE,
    }
    off = table[~table["on_front"]]
    on = table[table["on_front"]]
    if not off.empty:
        ax_sc.scatter(off["C_cost"], off["B_eco"], c="#BBBBBB", s=70, zorder=2, label="dominated")
    for _, row in table.iterrows():
        ax_sc.scatter(
            row["C_cost"],
            row["B_eco"],
            c=colors.get(str(row["strategy"]), "k"),
            s=88,
            zorder=4,
            edgecolors="k",
            linewidths=0.4,
        )
        ax_sc.annotate(str(row["label"]), (row["C_cost"], row["B_eco"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    if len(on) >= 2:
        ordered = on.sort_values("C_cost")
        ax_sc.plot(ordered["C_cost"], ordered["B_eco"], color="#333333", lw=1.0, ls="--", zorder=1)
    ax_sc.set_xlabel("C_cost  (turf + management)")
    ax_sc.set_ylabel("B_eco  (pollinator + soil)")
    ax_sc.set_title("Benefit-cost Pareto")

    axes_names = ("pollinator", "soil", "turf health", "low mgmt", "net")
    poll = _minmax_col(table["B_pollinator"].to_numpy(dtype=np.float64))
    soil = _minmax_col(table["B_soil"].to_numpy(dtype=np.float64))
    turf_h = _minmax_col(-table["C_turf"].to_numpy(dtype=np.float64))
    mgmt = _minmax_col(-(table["C_mow"] + table["C_herb"]).to_numpy(dtype=np.float64))
    net = _minmax_col(table["net"].to_numpy(dtype=np.float64))
    radar = np.column_stack((poll, soil, turf_h, mgmt, net))
    ang = np.linspace(0.0, 2.0 * np.pi, len(axes_names), endpoint=False)
    ang = np.concatenate((ang, ang[:1]))
    ax_rd.remove()
    ax_rd = fig.add_subplot(1, 2, 2, polar=True)
    for i, row in table.iterrows():
        vals = np.concatenate((radar[int(i)], radar[int(i)][:1]))
        ax_rd.plot(ang, vals, color=colors.get(str(row["strategy"]), "k"), lw=1.6, label=str(row["label"]))
        ax_rd.fill(ang, vals, color=colors.get(str(row["strategy"]), "k"), alpha=0.08)
    ax_rd.set_xticks(ang[:-1])
    ax_rd.set_xticklabels(axes_names)
    ax_rd.set_yticklabels([])
    ax_rd.set_title("Normalized radar (higher better)")
    ax_rd.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), frameon=False, fontsize=8)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_bioeconomic_tradeoff.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    """跑四策略、写 CSV、只读 CSV 出图。"""
    scores = run_four_strategies("temperate")
    frame = scores_to_frame(scores)
    out_csv = PACK_ROOT / "results" / "bioeconomic_tradeoff.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    _style()
    fig_path = plot_tradeoff(pd.read_csv(out_csv), PACK_ROOT / "paper_figures")
    print(f"wrote {out_csv.relative_to(PACK_ROOT)}")
    print(f"wrote {fig_path.relative_to(PACK_ROOT)}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

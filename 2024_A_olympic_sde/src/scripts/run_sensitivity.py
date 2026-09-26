#!/usr/bin/env python3
"""布里斯班 2032 候选排序与准则权重单因子敏感性。

运行（题包根目录）::

    PYTHONPATH=. python -m src.scripts.run_sensitivity

主权重来自锁定 AHP 判断矩阵（与 ``run_eval`` 相同）。扰动后将向量重新归一化到
``sum(w)=1``，保证仍可送入 ``composite_scoring``。不使用伪随机数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from src.indicators.loader import (
    build_brisbane_candidates,
    build_sde_matrix,
    export_brisbane_candidates,
    export_sde_matrix,
)
from src.indicators.schema import INDICATORS, REQUIRED_INDICATOR_COLUMNS, validate_matrix
from src.mcda.ahp import compute_ahp_weights
from src.mcda.score import composite_scoring, normalize_indicators

CANDIDATE_CSV: Path = PACK_ROOT / "data" / "processed" / "brisbane_candidates.csv"
MATRIX_CSV: Path = PACK_ROOT / "data" / "processed" / "sde_matrix.csv"
AHP_CSV: Path = PACK_ROOT / "data" / "raw" / "ahp_judgment_locked.csv"
OUT_RANK: Path = PACK_ROOT / "results" / "brisbane_2032_ranking.csv"
OUT_SHOCK: Path = PACK_ROOT / "results" / "sensitivity_weight_shock.csv"
OUT_FIG: Path = PACK_ROOT / "paper_figures" / "fig_sensitivity_shock.png"

SHOCK_RELATIVE: float = 0.20
PLACE_BY_RANK: dict[int, str] = {1: "1st", 2: "2nd", 3: "3rd"}

# 主模型七个叶子权重合成为赛题六大准则（人气可达拆成覆盖/转播 vs 小项规模）。
IOC_CRITERIA: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Popularity_Accessibility", ("Appearances", "BROADCAST_VAL")),
    ("Programme_Footprint", ("Events_Count",)),
    ("Inclusivity", ("GLOBAL_REACH",)),
    ("Gender_Equity", ("GENDER_PARITY",)),
    ("Relevance_Innovation", ("YOUTH_APPEAL",)),
    ("Sustainability", ("INFRA_COST",)),
)

SPORT_COLOR: dict[str, str] = {
    "Cricket": "#0072B2",
    "Squash": "#E69F00",
    "Flag football": "#009E73",
}

SPORT_ROLE: dict[str, str] = {
    "CKT": "traditional_host",
    "SQU": "established_niche",
    "AFB": "emerging_youth",
}

INDICATOR_INDEX: dict[str, int] = {item.code: i for i, item in enumerate(INDICATORS)}


def load_locked_ahp_weights(path: Path = AHP_CSV) -> tuple[NDArray[np.float64], float, bool]:
    """读取锁定判断矩阵并求 Perron 权重。"""
    if not path.is_file():
        raise FileNotFoundError(f"missing locked AHP matrix: {path}")
    table = pd.read_csv(path, index_col=0)
    order = [item.code for item in INDICATORS]
    missing = [c for c in order if c not in table.columns or c not in table.index]
    if missing:
        raise ValueError(f"AHP matrix missing labels {missing}")
    aligned = table.loc[order, order].to_numpy(dtype=np.float64)
    return compute_ahp_weights(aligned)


def load_candidates(path: Path = CANDIDATE_CSV) -> pd.DataFrame:
    """读取三候选表；缺失则调用 loader 生成。"""
    if not path.is_file():
        export_brisbane_candidates(build_brisbane_candidates(), path)
    frame = pd.read_csv(path)
    return validate_matrix(frame)


def load_full_matrix(path: Path = MATRIX_CSV) -> pd.DataFrame:
    """读取含历史项的因子表，供青年权重归零时对照传统/新兴位次。"""
    if not path.is_file():
        export_sde_matrix(build_sde_matrix(), path)
    frame = pd.read_csv(path)
    return validate_matrix(frame)


def score_universe(
    raw: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    """在给定方案集合内 min-max 规范化后做加权和。"""
    labeled = raw.copy()
    if "SDE_Name" not in labeled.columns:
        labeled["SDE_Name"] = labeled["Discipline"].astype(str)
    if "Role" not in labeled.columns:
        labeled["Role"] = labeled["Code"].map(SPORT_ROLE).fillna("other")
    norm = normalize_indicators(labeled, list(INDICATORS))
    scored = composite_scoring(norm, weights, schema=INDICATORS)
    keep_raw = labeled.loc[:, ["Code", *REQUIRED_INDICATOR_COLUMNS]].copy()
    merged = scored.merge(keep_raw, on="Code", how="left", suffixes=("_norm", ""))
    merged["Place"] = merged["Rank"].map(lambda r: PLACE_BY_RANK.get(int(r), str(int(r))))
    cols = [
        "SDE_Name",
        "Code",
        "Role",
        "Place",
        "Rank",
        "Score",
        *list(REQUIRED_INDICATOR_COLUMNS),
    ]
    return merged.loc[:, cols].sort_values("Rank", kind="mergesort").reset_index(drop=True)


def perturb_group_weights(
    weights: np.ndarray,
    leaf_codes: tuple[str, ...],
    scale: float,
) -> NDArray[np.float64]:
    """把一组叶子权重乘以 ``scale``（``scale>=0``），再归一化和为 1。

    Parameters
    ----------
    weights :
        形状 ``(7,)``，与 ``INDICATORS`` 对齐，``sum=1``。
    leaf_codes :
        本准则对应的叶子因子代号。
    scale :
        例如 ``1.2`` 或 ``0.8``；``0`` 表示去掉该组权重。
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1).copy()
    n = len(INDICATORS)
    if int(w.size) != n:
        raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    if scale < 0.0:
        raise ValueError("scale must be non-negative")
    for code in leaf_codes:
        if code not in INDICATOR_INDEX:
            raise KeyError(f"unknown indicator {code}")
        idx = INDICATOR_INDEX[code]
        w[idx] = max(0.0, float(w[idx]) * float(scale))
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("perturbed weights have non-positive mass")
    return (w / total).astype(np.float64)


def zero_youth_weights(weights: np.ndarray) -> NDArray[np.float64]:
    """极端情景：``YOUTH_APPEAL`` 权重置 0，其余按比例重分配。"""
    return perturb_group_weights(weights, ("YOUTH_APPEAL",), 0.0)


def _shock_rows(
    universe: str,
    scenario: str,
    criterion: str,
    shock: str,
    ranking: pd.DataFrame,
    baseline_rank: dict[str, int],
) -> list[dict[str, object]]:
    """把一次打分展开为敏感性总表行。"""
    rows: list[dict[str, object]] = []
    for _, rec in ranking.iterrows():
        code = str(rec["Code"])
        rank = int(rec["Rank"])
        base = int(baseline_rank[code])
        rows.append(
            {
                "Universe": universe,
                "Scenario": scenario,
                "Criterion": criterion,
                "Shock": shock,
                "SDE_Name": str(rec["SDE_Name"]),
                "Code": code,
                "Role": str(rec["Role"]),
                "Score": float(rec["Score"]),
                "Rank": rank,
                "Place": str(rec["Place"]),
                "Baseline_Rank": base,
                "Delta_Rank": base - rank,
            }
        )
    return rows


def run_oat_on_candidates(
    raw: pd.DataFrame,
    base_weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """基线排名 + 六准则 ±20% + 青年权重归零。"""
    baseline = score_universe(raw, base_weights)
    base_rank = {str(r["Code"]): int(r["Rank"]) for _, r in baseline.iterrows()}
    records: list[dict[str, object]] = []
    records.extend(
        _shock_rows("brisbane_candidates", "baseline", "none", "0", baseline, base_rank)
    )
    for criterion, leaves in IOC_CRITERIA:
        for label, scale in (("plus20", 1.0 + SHOCK_RELATIVE), ("minus20", 1.0 - SHOCK_RELATIVE)):
            shocked = perturb_group_weights(base_weights, leaves, scale)
            ranked = score_universe(raw, shocked)
            scenario = f"{criterion}_{label}"
            records.extend(
                _shock_rows(
                    "brisbane_candidates",
                    scenario,
                    criterion,
                    label,
                    ranked,
                    base_rank,
                )
            )
    youth0 = score_universe(raw, zero_youth_weights(base_weights))
    records.extend(
        _shock_rows(
            "brisbane_candidates",
            "youth_weight_zero",
            "Relevance_Innovation",
            "set_zero",
            youth0,
            base_rank,
        )
    )
    shock_table = pd.DataFrame.from_records(records)
    return baseline, shock_table


def run_youth_zero_full_panel(
    full_raw: pd.DataFrame,
    base_weights: np.ndarray,
) -> pd.DataFrame:
    """在历史+候选全集上对比基线与青年权重=0（传统项 vs 近增/新兴项）。"""
    baseline = score_universe(full_raw, base_weights)
    youth0 = score_universe(full_raw, zero_youth_weights(base_weights))
    base_rank = {str(r["Code"]): int(r["Rank"]) for _, r in baseline.iterrows()}
    rows = _shock_rows("sde_matrix", "baseline", "none", "0", baseline, base_rank)
    rows.extend(
        _shock_rows(
            "sde_matrix",
            "youth_weight_zero",
            "Relevance_Innovation",
            "set_zero",
            youth0,
            base_rank,
        )
    )
    return pd.DataFrame.from_records(rows)


def robustness_summary(shock_table: pd.DataFrame) -> pd.DataFrame:
    """每个候选在布里斯班 OAT 情景中的名次极差（不含全集面板）。"""
    sub = shock_table.loc[shock_table["Universe"] == "brisbane_candidates"]
    rows: list[dict[str, object]] = []
    for code, grp in sub.groupby("Code", sort=False):
        ranks = grp["Rank"].to_numpy(dtype=int)
        name = str(grp["SDE_Name"].iloc[0])
        rows.append(
            {
                "Code": str(code),
                "SDE_Name": name,
                "Rank_min": int(np.min(ranks)),
                "Rank_max": int(np.max(ranks)),
                "Rank_range": int(np.max(ranks) - np.min(ranks)),
                "Times_1st": int(np.sum(ranks == 1)),
                "N_scenarios": int(len(ranks)),
            }
        )
    return pd.DataFrame(rows).sort_values("Rank_range", kind="mergesort").reset_index(drop=True)


def plot_sensitivity(
    shock_table: pd.DataFrame,
    dest: Path,
    robust_name: str,
    fragile_name: str,
) -> None:
    """平行坐标：左为综合分走线，右为名次走线（第 1 名在上）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    sub = shock_table.loc[shock_table["Universe"] == "brisbane_candidates"].copy()
    scenario_order = ["baseline"]
    for criterion, _leaves in IOC_CRITERIA:
        scenario_order.append(f"{criterion}_plus20")
        scenario_order.append(f"{criterion}_minus20")
    scenario_order.append("youth_weight_zero")
    short = {
        "baseline": "Base",
        "Popularity_Accessibility_plus20": "Pop+20",
        "Popularity_Accessibility_minus20": "Pop-20",
        "Programme_Footprint_plus20": "Evt+20",
        "Programme_Footprint_minus20": "Evt-20",
        "Inclusivity_plus20": "Inc+20",
        "Inclusivity_minus20": "Inc-20",
        "Gender_Equity_plus20": "Gen+20",
        "Gender_Equity_minus20": "Gen-20",
        "Relevance_Innovation_plus20": "Yth+20",
        "Relevance_Innovation_minus20": "Yth-20",
        "Sustainability_plus20": "Sus+20",
        "Sustainability_minus20": "Sus-20",
        "youth_weight_zero": "Yth=0",
    }
    names = [str(n) for n in sub["SDE_Name"].drop_duplicates().tolist()]
    x = np.arange(len(scenario_order), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    for ax, value_col, ylabel, invert_rank in (
        (axes[0], "Score", "Composite score", False),
        (axes[1], "Rank", "Rank (1 = best)", True),
    ):
        for name in names:
            ys: list[float] = []
            for sc in scenario_order:
                hit = sub.loc[(sub["SDE_Name"] == name) & (sub["Scenario"] == sc), value_col]
                if hit.empty:
                    raise ValueError(f"missing {value_col} for {name} / {sc}")
                ys.append(float(hit.iloc[0]))
            ax.plot(
                x,
                ys,
                color=SPORT_COLOR.get(name, "#333333"),
                marker="o",
                markersize=4.0,
                linewidth=1.8,
                label=name,
            )
        ax.set_xticks(x, [short[s] for s in scenario_order], rotation=55, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if invert_rank:
            ax.set_ylim(3.4, 0.6)
            ax.set_yticks([1, 2, 3])
    axes[0].set_title("Score walk (OAT ±20% and youth=0)")
    axes[1].set_title("Rank walk (lower is better)")
    axes[0].legend(loc="best", framealpha=0.92)
    fig.suptitle(
        f"Brisbane 2032 weight shocks: robust={robust_name}; most rank-sensitive={fragile_name}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_report(
    ranking: pd.DataFrame,
    robust: pd.DataFrame,
    shock_table: pd.DataFrame,
    weights: np.ndarray,
    cr: float,
    ok: bool,
) -> None:
    """终端：冠亚季、稳健性、青年归零对传统/新兴的影响。"""
    print("AHP locked weights (CR={:.4f}, consistent={})".format(cr, ok))
    for ind, w in zip(INDICATORS, weights, strict=True):
        print(f"  {ind.code:16} {w:.4f}")
    print()
    print("Brisbane 2032 ranking (locked AHP, 3-candidate min-max)")
    show = ranking.loc[:, ["Place", "Rank", "SDE_Name", "Role", "Score"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("OAT rank range on brisbane_candidates")
    print(robust.to_string(index=False))
    print()
    youth = shock_table.loc[
        (shock_table["Universe"] == "brisbane_candidates")
        & (shock_table["Scenario"] == "youth_weight_zero")
    ]
    print("Extreme: YOUTH_APPEAL weight = 0 (candidates only)")
    print(youth.loc[:, ["Place", "SDE_Name", "Role", "Score", "Baseline_Rank", "Delta_Rank"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))
    panel = shock_table.loc[shock_table["Universe"] == "sde_matrix"]
    if not panel.empty:
        print()
        print("Youth=0 on full sde_matrix (traditional core vs recent/emerging)")
        focus = {"ATH", "SWM", "SKB", "BKG", "CKT", "AFB"}
        both = panel.loc[panel["Code"].isin(focus)]
        pivot = both.pivot_table(
            index=["Code", "SDE_Name"],
            columns="Scenario",
            values="Rank",
            aggfunc="first",
        )
        print(pivot.to_string())
    print()
    print("解读提示")
    print(
        "  HOST_POPULARITY 已写入候选表，但不在锁定 AHP 七因子内，本脚本未把它计入综合分。"
    )
    print(
        "  青年权重归零后，YOUTH_APPEAL 较低的方案相对受益；高青年分的新兴项名次可能下滑。"
    )
    print(
        "  名次极差为 0 表示在 ±20% 与青年归零下排序不变（稳健）；"
        "极差大表示高度依赖某一准则的偏好。"
    )


def main() -> int:
    """候选打分、OAT 扰动、青年归零、CSV 与平行坐标图。"""
    weights, cr, ok = load_locked_ahp_weights()
    candidates = load_candidates()
    full = load_full_matrix()
    ranking, oat = run_oat_on_candidates(candidates, weights)
    panel = run_youth_zero_full_panel(full, weights)
    shock = pd.concat([oat, panel], ignore_index=True)
    robust = robustness_summary(shock)
    if robust.empty:
        raise RuntimeError("robustness table is empty")
    stable = robust.sort_values(
        ["Rank_range", "Times_1st"],
        ascending=[True, False],
        kind="mergesort",
    )
    fragile = robust.sort_values(
        ["Rank_range", "Times_1st"],
        ascending=[True, True],
        kind="mergesort",
    )
    robust_name = str(stable.iloc[0]["SDE_Name"])
    fragile_name = str(fragile.iloc[-1]["SDE_Name"])

    OUT_RANK.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUT_RANK, index=False, encoding="utf-8")
    shock.to_csv(OUT_SHOCK, index=False, encoding="utf-8")
    plot_sensitivity(shock, OUT_FIG, robust_name, fragile_name)
    print(f"wrote {OUT_RANK}")
    print(f"wrote {OUT_SHOCK}")
    print(f"wrote {OUT_FIG}")
    print()
    print_report(ranking, robust, shock, weights, cr, ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

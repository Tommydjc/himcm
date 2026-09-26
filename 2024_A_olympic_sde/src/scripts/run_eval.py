#!/usr/bin/env python3
"""历史 SDE 回测：锁定 AHP 权重 + 规范化加权分 + 条形图。

运行（题包根目录）::

    PYTHONPATH=. python -m src.scripts.run_eval
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from src.indicators.loader import build_sde_matrix, export_sde_matrix
from src.indicators.schema import INDICATORS, REQUIRED_INDICATOR_COLUMNS, validate_matrix
from src.mcda.ahp import compute_ahp_weights
from src.mcda.score import composite_scoring, normalize_indicators

MATRIX_CSV: Path = PACK_ROOT / "data" / "processed" / "sde_matrix.csv"
AHP_CSV: Path = PACK_ROOT / "data" / "raw" / "ahp_judgment_locked.csv"
OUT_CSV: Path = PACK_ROOT / "results" / "historical_ranking.csv"
OUT_FIG: Path = PACK_ROOT / "paper_figures" / "fig_historical_backtest.png"

CATEGORY_BY_CODE: dict[str, str] = {
    "ATH": "legacy_core",
    "SWM": "legacy_core",
    "SKB": "recent_add",
    "SRF": "recent_add",
    "BKG": "recent_add",
    "KTE": "marginal_exit",
    "CKT": "candidate_2032",
    "SQU": "candidate_2032",
    "AFB": "candidate_2032",
}

CATEGORY_LABEL_EN: dict[str, str] = {
    "legacy_core": "Legacy core",
    "recent_add": "Recent addition",
    "marginal_exit": "Marginal / dropped",
    "candidate_2032": "2032 candidate",
}

CATEGORY_COLOR: dict[str, str] = {
    "legacy_core": "#0072B2",
    "recent_add": "#009E73",
    "marginal_exit": "#D55E00",
    "candidate_2032": "#E69F00",
}


def load_factor_table(path: Path = MATRIX_CSV) -> pd.DataFrame:
    """读取 processed 因子表；缺失则现场跑 loader。"""
    if not path.is_file():
        export_sde_matrix(build_sde_matrix(), path)
    frame = pd.read_csv(path)
    return validate_matrix(frame)


def load_locked_ahp_matrix(path: Path = AHP_CSV) -> np.ndarray:
    """加载团队锁定的正互反判断矩阵，行/列顺序与 ``INDICATORS`` 一致。"""
    if not path.is_file():
        raise FileNotFoundError(f"missing locked AHP matrix: {path}")
    table = pd.read_csv(path, index_col=0)
    order = [item.code for item in INDICATORS]
    missing = [c for c in order if c not in table.columns or c not in table.index]
    if missing:
        raise ValueError(f"AHP matrix missing labels {missing}")
    aligned = table.loc[order, order]
    return aligned.to_numpy(dtype=np.float64)


def assign_category(frame: pd.DataFrame) -> pd.DataFrame:
    """按 IOC 近年增删角色打类别。"""
    out = frame.copy()
    out["Category"] = out["Code"].map(CATEGORY_BY_CODE).fillna("other")
    out["SDE_Name"] = out["Discipline"].astype(str)
    return out


def build_ranking(
    raw: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    """原始因子保留，Norm_Score / Final_Rank 来自加权规范化分。"""
    labeled = assign_category(raw)
    norm = normalize_indicators(labeled, list(INDICATORS))
    scored = composite_scoring(norm, weights, schema=INDICATORS)
    raw_metrics = labeled.loc[:, ["Code", *REQUIRED_INDICATOR_COLUMNS]].copy()
    ranked = scored.merge(raw_metrics, on="Code", how="left", suffixes=("_norm", ""))
    export_cols = [
        "SDE_Name",
        "Category",
        "Code",
        *list(REQUIRED_INDICATOR_COLUMNS),
        "Score",
        "Rank",
    ]
    out = ranked.loc[:, export_cols].copy()
    out = out.rename(columns={"Score": "Norm_Score", "Rank": "Final_Rank"})
    return out.sort_values("Final_Rank", kind="mergesort").reset_index(drop=True)


def plot_backtest(ranking: pd.DataFrame, dest: Path) -> None:
    """综合得分降序条形图，类别分色，300 DPI。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ordered = ranking.sort_values("Norm_Score", ascending=False, kind="mergesort")
    names = ordered["SDE_Name"].tolist()
    scores = ordered["Norm_Score"].to_numpy(dtype=float)
    colors = [CATEGORY_COLOR.get(str(c), "#7f7f7f") for c in ordered["Category"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(np.arange(len(names)), scores, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(np.arange(len(names)), names, rotation=20, ha="right")
    ax.set_ylabel("AHP-weighted composite score")
    ax.set_title("Historical SDE backtest (locked AHP weights)")
    ax.set_ylim(0.0, max(1.0, float(np.max(scores)) * 1.12))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        Patch(facecolor=CATEGORY_COLOR[k], edgecolor="black", label=CATEGORY_LABEL_EN[k])
        for k in ("legacy_core", "recent_add", "marginal_exit", "candidate_2032")
        if k in set(ordered["Category"])
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_report(ranking: pd.DataFrame, weights: np.ndarray, cr: float, ok: bool) -> None:
    """打印排名与回测解读。"""
    print("AHP locked weights (CR={:.4f}, consistent={})".format(cr, ok))
    for ind, w in zip(INDICATORS, weights, strict=True):
        print(f"  {ind.code:16} {w:.4f}  ({ind.sense})")
    print()
    show = ranking.copy()
    show["Category_CN"] = show["Category"].map(
        {
            "legacy_core": "长期稳固项",
            "recent_add": "近增项",
            "marginal_exit": "边缘淘汰项",
            "candidate_2032": "2032 候选",
        }
    )
    cols = ["Final_Rank", "SDE_Name", "Category_CN", "Norm_Score"]
    print(show.loc[:, cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    by_name = ranking.set_index("SDE_Name")
    karate = float(by_name.loc["Karate", "Norm_Score"]) if "Karate" in by_name.index else float("nan")
    recent = ranking.loc[ranking["Category"] == "recent_add", "Norm_Score"]
    legacy = ranking.loc[ranking["Category"] == "legacy_core", "Norm_Score"]
    print("回测解读提示")
    if not recent.empty:
        print(
            f"  近增项（滑板/冲浪/霹雳舞）均分 {float(recent.mean()):.3f}；"
            f"长期稳固项（田径/游泳）均分 {float(legacy.mean()):.3f}。"
        )
    if "Karate" in by_name.index:
        print(
            f"  空手道（东京后未进入 2024/2028 正式小项）综合分 {karate:.3f}，"
            "相对青年向近增项的名次可用于对照 IOC 近年『青年化、城市运动』增补与一次性项目退出。"
        )
    print(
        "  田径/游泳高 Appearances 与转播分，模型应仍给稳固核心较高分；"
        "这与它们长期留在奥运节目单一致，不等于它们也是 2032 最该『新增』的项目。"
    )
    print(
        "  板球/壁球/腰旗橄榄球标为 candidate_2032，其名次是前瞻对照，不是 2020–2028 已发生的增删事实。"
    )


def main() -> int:
    """端到端：AHP → 规范化加权 → CSV + 图 + 终端报告。"""
    raw = load_factor_table()
    matrix = load_locked_ahp_matrix()
    weights, cr, ok = compute_ahp_weights(matrix)
    ranking = build_ranking(raw, weights)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUT_CSV, index=False, encoding="utf-8")
    plot_backtest(ranking, OUT_FIG)
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_FIG}")
    print()
    print_report(ranking, weights, cr, ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

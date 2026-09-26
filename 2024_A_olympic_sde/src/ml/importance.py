#!/usr/bin/env python3
"""AHP 主观权与 Logistic 标准化 |β| 的六准则交叉印证。

全量标签样本上 ``StandardScaler`` + L2 logistic。丢掉截距与惯性两列后，
对 6 个 IOC 准则 ``|β_j|`` 做 L1 归一。AHP 取锁定 Perron 向量中与这 6 列
同名的分量并重新归一到和为 1（``Appearances`` 不在 dataset 六准则块中）。

``Δw_j = w_j^{ML} - w_j^{AHP}``。Spearman 报告双尾 p 值。

运行::

    PYTHONPATH=. python -m src.ml.importance
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from src.indicators.schema import INDICATORS
from src.mcda.ahp import compute_ahp_weights
from src.ml.classifier import _new_logit
from src.ml.dataset import FEATURE_COLS, INERTIA_COLS, IOC_FEATURE_COLS, build_inclusion_dataset

AHP_CSV: Path = PACK_ROOT / "data" / "raw" / "ahp_judgment_locked.csv"
OUT_CSV: Path = PACK_ROOT / "results" / "ahp_vs_ml_weights.csv"
OUT_FIG: Path = PACK_ROOT / "paper_figures" / "fig_ahp_vs_ml_weights.png"

WEIGHT_ATOL: float = 1.0e-12
ALIGN_TOL: float = 0.02


def load_ahp_weights(path: Path = AHP_CSV) -> tuple[NDArray[np.float64], list[str]]:
    """调用 ``compute_ahp_weights``，顺序与 ``INDICATORS`` 一致。"""
    if not path.is_file():
        raise FileNotFoundError(f"missing locked AHP matrix: {path}")
    table = pd.read_csv(path, index_col=0)
    order = [item.code for item in INDICATORS]
    missing = [c for c in order if c not in table.columns or c not in table.index]
    if missing:
        raise ValueError(f"AHP matrix missing labels {missing}")
    weights, _cr, ok = compute_ahp_weights(table.loc[order, order].to_numpy(dtype=np.float64))
    if not ok:
        raise ValueError("locked AHP matrix failed CR < 0.1")
    return weights, order


def ahp_weights_on_criteria(criteria: tuple[str, ...]) -> NDArray[np.float64]:
    """锁定 AHP 在指定准则上的质量并 L1 归一，便于与六列 ML 权比较。"""
    full, names = load_ahp_weights()
    by_name = {name: float(full[i]) for i, name in enumerate(names)}
    missing = [c for c in criteria if c not in by_name]
    if missing:
        raise KeyError(f"AHP vector missing {missing}")
    sliced = np.asarray([by_name[c] for c in criteria], dtype=np.float64)
    return l1_normalize(sliced)


def l1_normalize(values: np.ndarray) -> NDArray[np.float64]:
    """``w_j = |v_j| / sum|v_k|``。"""
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    if raw.size < 1:
        raise ValueError("values must be non-empty")
    if not np.all(np.isfinite(raw)):
        raise ValueError("values contain NaN or inf")
    mass = np.abs(raw)
    total = float(np.sum(mass))
    if total <= WEIGHT_ATOL:
        return np.full(raw.shape, 1.0 / float(raw.size), dtype=np.float64)
    return (mass / total).astype(np.float64)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """``(u·v)/(||u|| ||v||)``。"""
    u = np.asarray(left, dtype=np.float64).reshape(-1)
    v = np.asarray(right, dtype=np.float64).reshape(-1)
    if u.shape != v.shape:
        raise ValueError(f"shape mismatch {u.shape} vs {v.shape}")
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom <= WEIGHT_ATOL:
        return 0.0
    return float(np.dot(u, v) / denom)


def spearman_rho(left: np.ndarray, right: np.ndarray) -> float:
    """仅 ρ，供单测。"""
    rho, _p = spearman_with_pvalue(left, right)
    return rho


def spearman_with_pvalue(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Spearman ρ 与双尾 p 值；常数向量时 ρ=0、p=1。"""
    u = np.asarray(left, dtype=np.float64).reshape(-1)
    v = np.asarray(right, dtype=np.float64).reshape(-1)
    if u.shape != v.shape:
        raise ValueError(f"shape mismatch {u.shape} vs {v.shape}")
    if float(np.std(u)) <= WEIGHT_ATOL or float(np.std(v)) <= WEIGHT_ATOL:
        return 0.0, 1.0
    rho, p_val = spearmanr(u, v, alternative="two-sided")
    rho_f = 0.0 if rho is None or not np.isfinite(rho) else float(rho)
    p_f = 1.0 if p_val is None or not np.isfinite(p_val) else float(p_val)
    return rho_f, p_f


def logistic_criterion_weights(
    coef: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """丢掉惯性列后对 6 准则 ``|β|`` L1 归一。截距不在 ``coef_`` 中。"""
    beta = np.asarray(coef, dtype=np.float64).reshape(-1)
    if beta.size != len(feature_names):
        raise ValueError(f"coef length {beta.size} != {len(feature_names)} features")
    keep: list[int] = []
    for i, name in enumerate(feature_names):
        if name in INERTIA_COLS:
            continue
        if name not in IOC_FEATURE_COLS:
            raise ValueError(f"unexpected non-inertia feature {name}")
        keep.append(i)
    if len(keep) != 6:
        raise ValueError(f"expected 6 criterion coefficients, got {len(keep)}")
    beta_six = beta[np.asarray(keep, dtype=np.int64)]
    return l1_normalize(beta_six), beta_six


def bias_interpretation(delta: float) -> str:
    """``Δw = w_ML - w_AHP``。"""
    if abs(delta) < ALIGN_TOL:
        return "aligned"
    if delta > 0.0:
        return "ML_higher_than_AHP"
    return "AHP_higher_than_ML"


def build_comparison_table(
    criteria: tuple[str, ...],
    w_ahp: np.ndarray,
    w_ml: np.ndarray,
    cosine: float,
    rho: float,
    p_rho: float,
) -> pd.DataFrame:
    """导出列：Criterion, AHP_Subjective_Weight, ML_Objective_Weight, Difference, Bias_Interpretation。"""
    rows: list[dict[str, object]] = []
    for j, name in enumerate(criteria):
        delta = float(w_ml[j] - w_ahp[j])
        rows.append(
            {
                "Criterion": name,
                "AHP_Subjective_Weight": float(w_ahp[j]),
                "ML_Objective_Weight": float(w_ml[j]),
                "Difference": delta,
                "Bias_Interpretation": bias_interpretation(delta),
                "Cosine_Similarity": cosine,
                "Spearman_Rho": rho,
                "Spearman_pvalue_twosided": p_rho,
            }
        )
    table = pd.DataFrame.from_records(rows)
    np.testing.assert_allclose(float(table["AHP_Subjective_Weight"].sum()), 1.0, atol=1.0e-8)
    np.testing.assert_allclose(float(table["ML_Objective_Weight"].sum()), 1.0, atol=1.0e-8)
    return table


def plot_dumbbell(table: pd.DataFrame, dest: Path) -> None:
    """哑铃图：AHP vs 六准则 |β|。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values("AHP_Subjective_Weight", ascending=True, kind="mergesort")
    names = ordered["Criterion"].tolist()
    y = np.arange(len(names), dtype=float)
    ahp = ordered["AHP_Subjective_Weight"].to_numpy(dtype=float)
    ml = ordered["ML_Objective_Weight"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for yi, a, m in zip(y, ahp, ml, strict=True):
        ax.plot([a, m], [yi, yi], color="#4d4d4d", linewidth=1.6, zorder=1)
    ax.scatter(ahp, y, s=64, color="#0072B2", zorder=2, label="AHP (subjective)")
    ax.scatter(ml, y, s=64, color="#D55E00", zorder=2, label=r"ML $|\beta|$ (6 criteria)")
    ax.set_yticks(y, names)
    ax.set_xlabel("Normalized weight on the six-criterion simplex")
    ax.set_xlim(0.0, max(0.40, float(np.max(np.concatenate([ahp, ml]))) * 1.15))
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "AHP vs logistic |β|  (cosine={:.3f}, Spearman ρ={:.3f}, p={:.3f})".format(
            float(ordered["Cosine_Similarity"].iloc[0]),
            float(ordered["Spearman_Rho"].iloc[0]),
            float(ordered["Spearman_pvalue_twosided"].iloc[0]),
        )
    )
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_report(table: pd.DataFrame) -> None:
    """终端：收敛性与“历史决定性高于专家”的准则。"""
    show = table.loc[
        :,
        [
            "Criterion",
            "AHP_Subjective_Weight",
            "ML_Objective_Weight",
            "Difference",
            "Bias_Interpretation",
        ],
    ]
    print("AHP subjective vs ML objective (6 IOC criteria; inertia/intercept dropped)")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    cosine = float(table["Cosine_Similarity"].iloc[0])
    rho = float(table["Spearman_Rho"].iloc[0])
    p_rho = float(table["Spearman_pvalue_twosided"].iloc[0])
    print(f"cosine similarity = {cosine:.4f}")
    print(f"Spearman rho = {rho:.4f}  two-sided p = {p_rho:.4f}")
    if p_rho < 0.05:
        print("  rank agreement is statistically detectable at 5% (n=6 criteria, still a small test).")
    else:
        print("  rank agreement is NOT significant at 5%; do not claim AHP and ML tell the same story.")
    ml_hi = table.loc[table["Difference"] > ALIGN_TOL].sort_values(
        "Difference", ascending=False, kind="mergesort"
    )
    if ml_hi.empty:
        print("No criterion has ML weight more than 0.02 above AHP.")
    else:
        top = ml_hi.iloc[0]
        print(
            "Historical inclusion signal puts MORE weight than the expert AHP prior on: "
            f"{top['Criterion']} (Δw={float(top['Difference']):+.4f})."
        )
        others = [str(x) for x in ml_hi["Criterion"].tolist()]
        print(f"  All ML-higher criteria: {others}")
    print("N=6 labeled SDEs; this is a diagnostic against 'AHP is only a guess', not a replacement prior.")


def main() -> int:
    """标准化 logistic、六准则 |β|、对照锁定 AHP、写 CSV。"""
    data = build_inclusion_dataset()
    scaler = StandardScaler()
    x_std = np.asarray(scaler.fit_transform(data.X), dtype=np.float64)
    logit = _new_logit()
    logit.fit(x_std, data.y)
    coef = np.asarray(logit.coef_, dtype=np.float64).reshape(-1)
    assert coef.size == len(FEATURE_COLS)
    w_ml, _beta_six = logistic_criterion_weights(coef, FEATURE_COLS)
    w_ahp = ahp_weights_on_criteria(IOC_FEATURE_COLS)
    cosine = cosine_similarity(w_ahp, w_ml)
    rho, p_rho = spearman_with_pvalue(w_ahp, w_ml)
    table = build_comparison_table(IOC_FEATURE_COLS, w_ahp, w_ml, cosine, rho, p_rho)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    export = table.loc[
        :,
        [
            "Criterion",
            "AHP_Subjective_Weight",
            "ML_Objective_Weight",
            "Difference",
            "Bias_Interpretation",
        ],
    ]
    export.to_csv(OUT_CSV, index=False, encoding="utf-8")
    plot_dumbbell(table, OUT_FIG)
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_FIG}")
    print()
    print_report(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

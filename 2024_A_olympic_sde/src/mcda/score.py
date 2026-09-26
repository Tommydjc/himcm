"""规范化加权综合分与轻量 TOPSIS（方案层；权重由 AHP 等模块提供）。

效益/成本 min-max 使用 ``+ε`` 避免除零。不写入 ``results/``，不使用伪随机数。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.indicators.schema import Indicator

RANGE_EPSILON: float = 1.0e-12
WEIGHT_ATOL: float = 1.0e-8


def _indicator_block(
    data: pd.DataFrame,
    schema: Sequence[Indicator],
) -> tuple[pd.DataFrame, list[str]]:
    """取出按 ``schema`` 顺序排列的因子列，校验列齐全且为有限数值。"""
    if not schema:
        raise ValueError("schema must contain at least one Indicator")
    codes = [item.code for item in schema]
    missing = [c for c in codes if c not in data.columns]
    if missing:
        raise ValueError(f"missing indicator columns: {missing}")
    if data.shape[0] < 1:
        raise ValueError("data must have at least one SDE row")
    block = data.loc[:, codes].apply(pd.to_numeric, errors="coerce")
    if block.shape != (data.shape[0], len(codes)):
        raise ValueError(
            f"indicator block must be m×{len(codes)}, got {block.shape}"
        )
    if block.isna().any().any():
        bad = block.columns[block.isna().any()].tolist()
        raise ValueError(f"non-finite or empty values in {bad}")
    return block, codes


def _assert_weights(weights: np.ndarray, n_criteria: int) -> NDArray[np.float64]:
    """权重形状 ``(n,)``、非负、和为 1。"""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.ndim != 1 or int(w.size) != n_criteria:
        raise ValueError(
            f"weights must have shape ({n_criteria},), got {np.asarray(weights).shape}"
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contain NaN or inf")
    if np.any(w < -WEIGHT_ATOL):
        raise ValueError("weights must be non-negative")
    w = np.clip(w, 0.0, None)
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    if abs(total - 1.0) > 1.0e-6:
        raise ValueError(f"weights must sum to 1, got {total}")
    return (w / total).astype(np.float64)


def normalize_indicators(
    data: pd.DataFrame,
    schema: list[Indicator] | Sequence[Indicator],
) -> pd.DataFrame:
    """按列做效益/成本 min-max 规范化，结果落在 ``[0, 1]``。

    效益型::

        x̃ = (x - min) / (max - min + ε)

    成本型::

        x̃ = (max - x) / (max - min + ε)

    其中 min/max 对**同一因子、所有方案**（列方向）取值。``ε = RANGE_EPSILON``。
    非因子列原样保留。
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be DataFrame, got {type(data)!r}")
    schema_list = list(schema)
    block, codes = _indicator_block(data, schema_list)
    out = data.copy()
    for item in schema_list:
        col = block[item.code].to_numpy(dtype=np.float64)
        col_min = float(np.min(col))
        col_max = float(np.max(col))
        denom = (col_max - col_min) + RANGE_EPSILON
        if item.sense == "benefit":
            scaled = (col - col_min) / denom
        elif item.sense == "cost":
            scaled = (col_max - col) / denom
        else:
            raise ValueError(f"unknown sense {item.sense!r} for {item.code}")
        scaled = np.clip(scaled, 0.0, 1.0)
        out[item.code] = scaled
    return out


def composite_scoring(
    norm_data: pd.DataFrame,
    weights: np.ndarray,
    *,
    schema: Sequence[Indicator] | None = None,
) -> pd.DataFrame:
    """加权和 ``S_i = Σ_j w_j x̃_{ij}``，按得分降序并写入 ``Score``、``Rank``。

    Parameters
    ----------
    norm_data :
        ``normalize_indicators`` 的输出；因子列已在 ``[0, 1]``。
    weights :
        形状 ``(n,)``，与因子列顺序一致（``schema`` 顺序，或 ``norm_data`` 中
        数值因子列从左到右）。必须和为 1。
    schema :
        若给定，只用这些 ``code`` 列，避免把 ``Score`` 等辅助列当因子。
    """
    if not isinstance(norm_data, pd.DataFrame):
        raise TypeError("norm_data must be DataFrame")
    if schema is not None:
        codes = [item.code for item in schema]
        missing = [c for c in codes if c not in norm_data.columns]
        if missing:
            raise ValueError(f"missing normalized columns: {missing}")
        block = norm_data.loc[:, codes].to_numpy(dtype=np.float64)
    else:
        numeric = norm_data.select_dtypes(include=(np.number,))
        block = numeric.to_numpy(dtype=np.float64)
        codes = list(numeric.columns)
    if block.ndim != 2 or block.shape[1] < 1:
        raise ValueError("no numeric indicator columns to score")
    w = _assert_weights(weights, int(block.shape[1]))
    if np.any(block < -WEIGHT_ATOL) or np.any(block > 1.0 + WEIGHT_ATOL):
        raise ValueError("normalized indicators must lie in [0, 1]")
    scores = block @ w
    result = norm_data.copy()
    result["Score"] = scores
    result["Rank"] = (
        result["Score"].rank(ascending=False, method="min").astype(int)
    )
    return result.sort_values("Score", ascending=False, kind="mergesort").reset_index(
        drop=True
    )


def topsis_scoring(
    norm_data: pd.DataFrame,
    weights: np.ndarray,
    *,
    schema: Sequence[Indicator] | None = None,
) -> pd.DataFrame:
    """在已转为效益方向的 ``[0,1]`` 表上算加权 TOPSIS 贴近度。

    加权矩阵 ``v_{ij} = w_j x̃_{ij}``。正理想解为各列最大，负理想解为各列最小
    （成本型已在 ``normalize_indicators`` 中取补）。贴近度

    ``C_i = D_i^- / (D_i^+ + D_i^- + ε)``，``D`` 为欧氏距离。

    输出列 ``TopsisScore``、``Rank``，按贴近度降序。
    """
    if not isinstance(norm_data, pd.DataFrame):
        raise TypeError("norm_data must be DataFrame")
    if schema is not None:
        codes = [item.code for item in schema]
        missing = [c for c in codes if c not in norm_data.columns]
        if missing:
            raise ValueError(f"missing normalized columns: {missing}")
        block = norm_data.loc[:, codes].to_numpy(dtype=np.float64)
    else:
        numeric = norm_data.select_dtypes(include=(np.number,))
        block = numeric.to_numpy(dtype=np.float64)
    w = _assert_weights(weights, int(block.shape[1]))
    if np.any(block < -WEIGHT_ATOL) or np.any(block > 1.0 + WEIGHT_ATOL):
        raise ValueError("normalized indicators must lie in [0, 1]")
    weighted = block * w.reshape(1, -1)
    ideal_pos = np.max(weighted, axis=0)
    ideal_neg = np.min(weighted, axis=0)
    d_pos = np.sqrt(np.sum((weighted - ideal_pos) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weighted - ideal_neg) ** 2, axis=1))
    closeness = d_neg / (d_pos + d_neg + RANGE_EPSILON)
    closeness = np.clip(closeness, 0.0, 1.0)
    result = norm_data.copy()
    result["TopsisScore"] = closeness
    result["Rank"] = (
        result["TopsisScore"].rank(ascending=False, method="min").astype(int)
    )
    return result.sort_values(
        "TopsisScore", ascending=False, kind="mergesort"
    ).reset_index(drop=True)

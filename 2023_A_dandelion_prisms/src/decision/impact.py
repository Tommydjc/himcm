"""入侵种影响因子：锁定 SAW。

.. math::

    I=\\sum_{k\\in\\mathcal{B}} w_k\\,\\tilde x_k
    +\\sum_{k\\in\\mathcal{C}} w_k\\,(1-\\tilde x_k),

效益型 \(\\mathcal{B}=\) spread, competition, economic, health, ecology；
成本型 \(\\mathcal{C}=\) utility（食用/药用越高，净影响越低）。
\(\\tilde x\) 在**本表物种集合**上 min-max，禁止与其他宇宙横比。

蒲公英的 spread 可用温带 12 月覆盖率映射

.. math::

    x_{\\mathrm{spread}}=1+8\\bigl(1-e^{-\\mathrm{cover}/0.15}\\bigr)\\in[1,9].
"""

from __future__ import annotations

from typing import Any

import numpy as np


BENEFIT_KEYS: tuple[str, ...] = ("spread", "competition", "economic", "health", "ecology")
COST_KEYS: tuple[str, ...] = ("utility",)


def cover_to_spread_score(cover_frac: float) -> float:
    """把 [0,1] 覆盖率映到 1–9 主观尺。"""
    cover = min(max(float(cover_frac), 0.0), 1.0)
    return float(1.0 + 8.0 * (1.0 - np.exp(-cover / 0.15)))


def _minmax(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if abs(hi - lo) < 1.0e-12:
        return [0.5] * len(values)
    return [float((v - lo) / (hi - lo)) for v in values]


def score_species(
    rows: list[dict[str, Any]],
    weights: dict[str, float],
) -> list[dict[str, float]]:
    """对已填好 1–9 属性的物种行计算 \(I\)。

    Parameters
    ----------
    rows :
        每行含 code 与六列原始分。
    weights :
        六权，和须为 1。
    """
    wsum = sum(float(weights[k]) for k in (*BENEFIT_KEYS, *COST_KEYS))
    if abs(wsum - 1.0) > 1.0e-8:
        raise ValueError(f"impact weights must sum to 1, got {wsum}")
    columns = {key: [float(row[key]) for row in rows] for key in (*BENEFIT_KEYS, *COST_KEYS)}
    normed = {key: _minmax(vals) for key, vals in columns.items()}
    out: list[dict[str, float]] = []
    for i, row in enumerate(rows):
        score = 0.0
        for key in BENEFIT_KEYS:
            score += float(weights[key]) * normed[key][i]
        for key in COST_KEYS:
            score += float(weights[key]) * (1.0 - normed[key][i])
        record = {key: float(row[key]) for key in (*BENEFIT_KEYS, *COST_KEYS)}
        record["impact"] = float(score)
        out.append(record)
    return out

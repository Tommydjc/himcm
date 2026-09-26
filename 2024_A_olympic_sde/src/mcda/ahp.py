"""层次分析法：正互反判断矩阵的特征值权重与一致性比率。

采用 Perron 根（最大实特征对）求归一化权重，**不用**列和归一化的近似特征向量。
本模块只输出准则/因子权重，不读取 ``results/``、不生成伪随机判断矩阵。
"""

from __future__ import annotations

import warnings
from typing import Final

import numpy as np
from numpy.typing import NDArray

# Saaty (1980) n=1..10；n=11,12 取常用扩展表。下标与阶数一致：``SAATY_RI[n]``。
SAATY_RI: Final[tuple[float, ...]] = (
    float("nan"),
    0.0,
    0.0,
    0.58,
    0.90,
    1.12,
    1.24,
    1.32,
    1.41,
    1.45,
    1.49,
    1.51,
    1.48,
)

CR_THRESHOLD: Final[float] = 0.10
RECIPROCAL_RTOL: Final[float] = 1.0e-6
RECIPROCAL_ATOL: Final[float] = 1.0e-9


def _assert_positive_reciprocal(matrix: NDArray[np.float64]) -> int:
    """校验正互反：方阵、``a_ij>0``、``a_ii=1``、``a_ji=1/a_ij``。返回阶数 ``n``。"""
    if matrix.ndim != 2:
        raise ValueError(f"comparison matrix must be 2-D, got ndim={matrix.ndim}")
    n_row, n_col = int(matrix.shape[0]), int(matrix.shape[1])
    if n_row != n_col:
        raise ValueError(f"comparison matrix must be square, got {matrix.shape}")
    if n_row < 1:
        raise ValueError("comparison matrix order n must be >= 1")
    if n_row > 12:
        raise ValueError("RI table covers n=1..12 only; group criteria into a hierarchy")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("comparison matrix contains NaN or inf")
    if np.any(matrix <= 0.0):
        raise ValueError("comparison matrix must be strictly positive (AHP scale 1/9..9)")
    diag = np.diag(matrix)
    if not np.allclose(diag, 1.0, rtol=RECIPROCAL_RTOL, atol=RECIPROCAL_ATOL):
        raise ValueError("diagonal entries must be 1")
    reciprocal = np.reciprocal(matrix.T)
    if not np.allclose(matrix, reciprocal, rtol=RECIPROCAL_RTOL, atol=RECIPROCAL_ATOL):
        raise ValueError("matrix is not reciprocal: require a_ji = 1/a_ij")
    return n_row


def saaty_ri(n: int) -> float:
    """返回阶数 ``n`` 的随机一致性指标 ``RI``。"""
    if n < 1 or n > 12:
        raise ValueError("n must be in 1..12")
    return float(SAATY_RI[n])


def compute_ahp_weights(
    comparison_matrix: np.ndarray,
) -> tuple[NDArray[np.float64], float, bool]:
    """由正互反判断矩阵求 AHP 权重、一致性比率与是否通过。

    Parameters
    ----------
    comparison_matrix :
        形状 ``(n, n)``，``n∈[1,12]``。须满足 ``a_ij>0``，``a_ii=1``，
        ``a_ji=1/a_ij``。

    Returns
    -------
    weights :
        归一化主特征向量 ``w``，形状 ``(n,)``，``sum(w)=1``，``w_i>0``。
    cr :
        一致性比率 ``CR = CI / RI``。``n≤2`` 时恒为 0。
    is_consistent :
        ``CR < 0.1`` 为 True；否则 False，并 ``warnings.warn``。

    Notes
    -----
    主特征对取模最大的特征值及其特征向量实部绝对值后归一化。
    ``CI = (λ_max - n) / (n - 1)``（``n=1`` 时 ``CI=0``）。
    """
    matrix = np.asarray(comparison_matrix, dtype=np.float64)
    n = _assert_positive_reciprocal(matrix)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    moduli = np.abs(eigenvalues)
    principal = int(np.argmax(moduli))
    lambda_max = float(np.real(eigenvalues[principal]))
    raw = np.real(eigenvectors[:, principal])
    raw = np.abs(raw)
    total = float(np.sum(raw))
    if total <= 0.0:
        raise ValueError("principal eigenvector has non-positive mass")
    weights = (raw / total).astype(np.float64)

    if n <= 2:
        cr = 0.0
    else:
        ci = (lambda_max - float(n)) / float(n - 1)
        ri = saaty_ri(n)
        if ri <= 0.0:
            cr = 0.0 if ci <= 1.0e-12 else float("inf")
        else:
            cr = float(max(0.0, ci / ri))

    is_consistent = bool(cr < CR_THRESHOLD)
    if not is_consistent:
        warnings.warn(
            f"AHP consistency failed: n={n}, CR={cr:.4f} >= {CR_THRESHOLD}",
            UserWarning,
            stacklevel=2,
        )
    return weights, cr, is_consistent

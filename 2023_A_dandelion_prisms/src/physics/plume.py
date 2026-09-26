"""2D 各向异性羽流：顺风 WALD × 侧向高斯，并做离散归一化。

风坐标系：\(x'\) 沿下风向，\(y'\) 左侧向。气象来向 \(\theta\)（0°=北，顺时针）对应
下风单位向量 \((-\\sin\\theta, -\\cos\\theta)\)（东、北分量）。

.. math::

    K(x',y') = f_{\\mathrm{WALD}}(x')\\,
    \\frac{1}{\\sqrt{2\\pi}\\,\\sigma_c(x')}
    \\exp\\left(-\\frac{y'^2}{2\\sigma_c(x')^2}\\right),
    \\quad
    \\sigma_c(x')=\\sigma_0+\\gamma\\sqrt{x'_+}.

离散核 \(K_{ij}=K(x_i,y_j)\\Delta x^2\) 后强制 \(\\sum_{ij}K_{ij}=1\)。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.params import Biophysics
from src.physics.wald import wald_mean_shape, wald_pdf


def rotate_offsets(
    x_east_m: NDArray[np.float64],
    y_north_m: NDArray[np.float64],
    wind_from_deg: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """把东-北偏移转到风坐标 \((x',y')\)。

    Parameters
    ----------
    x_east_m, y_north_m :
        相对源点的东、北位移（m）。
    wind_from_deg :
        气象来向（度）。
    """
    theta = np.deg2rad(wind_from_deg)
    down_e = -np.sin(theta)
    down_n = -np.cos(theta)
    x_prime = x_east_m * down_e + y_north_m * down_n
    y_prime = -x_east_m * down_n + y_north_m * down_e
    return x_prime, y_prime


def _continuous_density(
    x_prime: NDArray[np.float64],
    y_prime: NDArray[np.float64],
    mu: float,
    lam: float,
    sigma0_m: float,
    gamma_plume: float,
) -> NDArray[np.float64]:
    """连续 2D 密度（1/m²）。"""
    along = wald_pdf(x_prime, mu, lam)
    x_plus = np.maximum(x_prime, 0.0)
    sigma_c = np.maximum(sigma0_m + gamma_plume * np.sqrt(x_plus), 1.0e-3)
    cross = np.exp(-0.5 * (y_prime / sigma_c) ** 2) / (np.sqrt(2.0 * np.pi) * sigma_c)
    return along * cross


def build_plume_kernel(
    params: Biophysics,
    wind_mps: float,
    wind_from_deg: float,
) -> NDArray[np.float64]:
    """构造以源为中心的奇数边长离散核，元素和为 1。

    Parameters
    ----------
    params :
        生物物理契约。
    wind_mps :
        小时或周平均风速（m/s）。
    wind_from_deg :
        气象来向（度）。

    Returns
    -------
    ndarray
        形状 ``(2R+1, 2R+1)``，``R=kernel_half_m``（在 \(\\Delta x=1\) 时）。
    """
    u_bar = max(float(wind_mps), params.u_floor_mps)
    mu, lam = wald_mean_shape(params.H_m, u_bar, params.v_t_mps, params.kappa_turb)
    half = int(params.kernel_half_m)
    dx = float(params.dx_m)
    n = 2 * half + 1
    axis = (np.arange(n, dtype=np.float64) - half) * dx
    east, north = np.meshgrid(axis, axis, indexing="xy")
    x_p, y_p = rotate_offsets(east, north, wind_from_deg)
    dens = _continuous_density(x_p, y_p, mu, lam, params.sigma0_m, params.gamma_plume)
    kernel = dens * (dx * dx)
    total = float(np.sum(kernel))
    if total <= 0.0:
        raise ValueError("plume kernel mass vanished; check wind/parameters")
    return (kernel / total).astype(np.float64)


def deposit_from_point(
    n_seeds: float,
    x_src_m: float,
    y_src_m: float,
    x_centers: NDArray[np.float64],
    y_centers: NDArray[np.float64],
    params: Biophysics,
    wind_mps: float,
    wind_from_deg: float,
) -> tuple[NDArray[np.float64], float]:
    """把已归一化羽流核盖到格网上，返回（格内籽数场，落到域外的籽数）。

    质量账：``n_seeds = sum(field) + mass_outside``。核中心取距源点最近的格心
    （源在域外时中心索引可为负，窗外质量计入 ``mass_outside``）。

    Parameters
    ----------
    n_seeds :
        释放粒数。
    x_src_m, y_src_m :
        源点（m）。
    x_centers, y_centers :
        1D 格心坐标（m），长度均为 ``grid_n``。
    """
    if n_seeds < 0.0:
        raise ValueError("n_seeds must be nonnegative")
    kernel = build_plume_kernel(params, wind_mps, wind_from_deg)
    dx = float(params.dx_m)
    n = int(x_centers.size)
    field = np.zeros((n, n), dtype=np.float64)
    col0 = int(np.rint((x_src_m - float(x_centers[0])) / dx))
    row0 = int(np.rint((y_src_m - float(y_centers[0])) / dx))
    half = kernel.shape[0] // 2
    k_n = kernel.shape[0]
    row_start = row0 - half
    col_start = col0 - half
    src_r0 = max(0, -row_start)
    src_c0 = max(0, -col_start)
    dst_r0 = max(0, row_start)
    dst_c0 = max(0, col_start)
    src_r1 = min(k_n, n - row_start)
    src_c1 = min(k_n, n - col_start)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    if src_r1 > src_r0 and src_c1 > src_c0:
        field[dst_r0:dst_r1, dst_c0:dst_c1] = kernel[src_r0:src_r1, src_c0:src_c1] * n_seeds
    inside = float(np.sum(field))
    mass_outside = n_seeds - inside
    return field, mass_outside


def kernel_second_moment(kernel: NDArray[np.float64], dx_m: float) -> float:
    """离散核的二阶矩 \(M_2=\\sum (x^2+y^2)K_{ij}\)（m²）。"""
    ny, nx = kernel.shape
    if ny != nx or ny % 2 == 0:
        raise ValueError("kernel must be square with odd side")
    half = ny // 2
    axis = (np.arange(ny, dtype=np.float64) - half) * dx_m
    east, north = np.meshgrid(axis, axis, indexing="xy")
    return float(np.sum((east**2 + north**2) * kernel))

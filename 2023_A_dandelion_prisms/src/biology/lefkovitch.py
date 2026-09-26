"""四阶段 Lefkovitch 周步：种子库 \(S\)、幼苗 \(L\)、莲座 \(R\)、成株 \(A\)。

气候乘子

.. math::

    \\varphi_T=\\mathbf{1}_{T\\ge T_{\\min}}
    \\exp\\bigl(-(T-T_{\\mathrm{opt}})^2/(2\\sigma_T^2)\\bigr),
    \\qquad
    \\varphi_\\theta=\\frac{\\theta^{n}}{\\theta^{n}+\\theta_h^{n}},
    \\qquad
    \\varphi=\\varphi_T\\varphi_\\theta.

转移（已含周存活 \(s_\\bullet\) 与基线 \(g_\\bullet\\varphi\)）

.. math::

    \\begin{aligned}
    S' &= s_S(1-g_S)S,\\\\
    L' &= s_S g_S S + s_L(1-g_L)L,\\\\
    R' &= s_L g_L L + s_R(1-g_R)R,\\\\
    A' &= s_R g_R R + s_A A.
    \\end{aligned}

新籽不在本函数内写入 \(S\)：空间羽流与就近滞留由仿真层注入。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.params import Biophysics


@dataclass(frozen=True, slots=True)
class LefkovitchRates:
    """单周有效转移率（已乘 \(\\varphi\)）。"""

    g_seed: float
    g_seedling: float
    g_rosette: float
    s_seed: float
    s_seedling: float
    s_rosette: float
    s_adult: float


def climate_multiplier(temp_c: float, smi: float, params: Biophysics) -> float:
    """计算 \(\\varphi(T,\\theta)\\in[0,1]\)。"""
    if temp_c < params.T_min_c:
        return 0.0
    dtemp = temp_c - params.T_opt_c
    phi_t = float(np.exp(-0.5 * (dtemp / params.T_sigma_c) ** 2))
    theta = max(float(smi), 0.0)
    hill = theta ** params.smi_hill_n
    denom = hill + params.smi_hill ** params.smi_hill_n
    phi_m = hill / denom if denom > 0.0 else 0.0
    return float(np.clip(phi_t * phi_m, 0.0, 1.0))


def build_rates(temp_c: float, smi: float, params: Biophysics) -> LefkovitchRates:
    """由气候构造本周 Lefkovitch 率。"""
    phi = climate_multiplier(temp_c, smi, params)
    g_s = min(params.g_seed_base * phi, 0.95)
    g_l = min(params.g_seedling_base * phi, 0.95)
    g_r = min(params.g_rosette_base * phi, 0.95)
    return LefkovitchRates(
        g_seed=g_s,
        g_seedling=g_l,
        g_rosette=g_r,
        s_seed=params.surv_seed_week,
        s_seedling=params.surv_seedling_week,
        s_rosette=params.surv_rosette_week,
        s_adult=params.surv_adult_week,
    )


def lefkovitch_step(
    seed: NDArray[np.float64],
    seedling: NDArray[np.float64],
    rosette: NDArray[np.float64],
    adult: NDArray[np.float64],
    rates: LefkovitchRates,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """对四个密度场做一次线性阶段转移（不产籽、不自疏）。

    Parameters
    ----------
    seed, seedling, rosette, adult :
        同形状非负数组，单位：粒或株 / 格（1 m²）。
    """
    stay_s = rates.s_seed * (1.0 - rates.g_seed)
    to_l = rates.s_seed * rates.g_seed
    stay_l = rates.s_seedling * (1.0 - rates.g_seedling)
    to_r = rates.s_seedling * rates.g_seedling
    stay_r = rates.s_rosette * (1.0 - rates.g_rosette)
    to_a = rates.s_rosette * rates.g_rosette
    seed_n = stay_s * seed
    seedling_n = to_l * seed + stay_l * seedling
    rosette_n = to_r * seedling + stay_r * rosette
    adult_n = to_a * rosette + rates.s_adult * adult
    return seed_n, seedling_n, rosette_n, adult_n


def fecundity_seeds(
    adult: NDArray[np.float64],
    blooming: bool,
    temp_c: float,
    smi: float,
    params: Biophysics,
) -> NDArray[np.float64]:
    """花期成株产籽场（粒/格）。非花期为 0。

    .. math::

        F = A\\, n_{\\mathrm{cap}}\\, n_{\\mathrm{head}}\\, \\varphi \\, \\mathbf{1}_{\\mathrm{bloom}}.
    """
    if not blooming:
        return np.zeros_like(adult)
    phi = climate_multiplier(temp_c, smi, params)
    per_adult = params.n_seed_per_capitulum * params.n_capitula_per_adult_week_bloom * phi
    return adult * per_adult


def spectral_radius(rates: LefkovitchRates, seeds_per_adult: float) -> float:
    """含本地产籽闭环的 4×4 Lefkovitch 谱半径（无空间损失）。

    本地闭环把 \(F\) 全部写入 \(S\)，仅供 Fisher–KPP 的 \(r=\\ln\\rho/\\Delta t\) 对照。
    """
    matrix = np.array(
        [
            [rates.s_seed * (1.0 - rates.g_seed), 0.0, 0.0, seeds_per_adult],
            [rates.s_seed * rates.g_seed, rates.s_seedling * (1.0 - rates.g_seedling), 0.0, 0.0],
            [0.0, rates.s_seedling * rates.g_seedling, rates.s_rosette * (1.0 - rates.g_rosette), 0.0],
            [0.0, 0.0, rates.s_rosette * rates.g_rosette, rates.s_adult],
        ],
        dtype=np.float64,
    )
    eig = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eig)))

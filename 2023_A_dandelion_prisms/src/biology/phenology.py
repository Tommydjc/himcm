r"""日步四阶段 Lefkovitch 状态机（无空间卷积）。

状态：种子 \(S\)、幼苗 \(L\)（25 日龄结构）、莲座 \(R\)、开花成株 \(A\)。
环境修正 \(\gamma(T,M)\in[0,1]\) 只调萌发与莲座→成株；莲座抗旱存活不随 \(\gamma\) 坍缩。
成株硬上限 \(K_{\mathrm{cell}}=15\)；幼苗存活随 \(A/K_{\mathrm{cell}}\) 密度依赖下降。

本模块不导入 ``physics`` / ``simulation``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PhenologyParams:
    r"""日步生活史参数。

    Parameters
    ----------
    seedling_days :
        幼苗建成天数，默认 25。
    n_seed_per_adult :
        每轮每成株落籽数，默认 180。
    K_cell :
        单格成株硬上限（株/m²）。
    g_seed, g_rosette :
        基线日萌发率、莲座→成株率（再乘 \(\gamma\)）。
    s_seed, s_seedling, s_rosette, s_adult :
        日存活（幼苗再乘密度因子）。
    mu_dd :
        密度依赖指数：\(s_L\leftarrow s_L(1-(A/K)^{\mu_{dd}})\)。
    T_base, T_max, k_T, k_ceil, K_m, M_waterlog :
        \(\gamma(T,M)\) 的热力逻辑斯蒂与水分 Michaelis–Menten。
    """

    seedling_days: int = 25
    n_seed_per_adult: float = 180.0
    K_cell: float = 15.0
    g_seed: float = 0.040
    g_rosette: float = 0.030
    s_seed: float = 0.995
    s_seedling: float = 0.980
    s_rosette: float = 0.997
    s_adult: float = 0.998
    mu_dd: float = 2.0
    T_base: float = 10.0
    T_max: float = 35.0
    k_T: float = 0.5
    k_ceil: float = 0.8
    K_m: float = 0.12
    M_waterlog: float = 0.45

    def __post_init__(self) -> None:
        if self.seedling_days < 1:
            raise ValueError("seedling_days must be >= 1")
        if self.n_seed_per_adult < 0.0 or self.K_cell <= 0.0:
            raise ValueError("n_seed_per_adult nonnegative and K_cell positive")


@dataclass
class StageState:
    """四阶段密度场。

    Attributes
    ----------
    seed :
        种子库，形状 ``(n, n)``，粒/格。
    seedling :
        幼苗龄结构，形状 ``(seedling_days, n, n)``，龄 0 为当日萌发。
    rosette, adult :
        莲座与开花成株，形状 ``(n, n)``，株/格。
    """

    seed: NDArray[np.float64]
    seedling: NDArray[np.float64]
    rosette: NDArray[np.float64]
    adult: NDArray[np.float64]

    @property
    def seedling_total(self) -> NDArray[np.float64]:
        """对龄轴求和的幼苗密度。"""
        return np.sum(self.seedling, axis=0)


def empty_stages(n: int, params: PhenologyParams) -> StageState:
    """全零阶段场。"""
    if n <= 0:
        raise ValueError("n must be positive")
    z = np.zeros((n, n), dtype=np.float64)
    seedling = np.zeros((params.seedling_days, n, n), dtype=np.float64)
    return StageState(seed=z.copy(), seedling=seedling, rosette=z.copy(), adult=z.copy())


def initial_center_adult(
    n: int,
    params: PhenologyParams,
    row: int = 50,
    col: int = 50,
) -> StageState:
    r"""\(t=0\)：仅 ``(row, col)`` 一株开花成株。默认格心 (50, 50)（0 起算）。"""
    state = empty_stages(n, params)
    if not (0 <= row < n and 0 <= col < n):
        raise IndexError(f"center ({row}, {col}) outside {n}x{n}")
    state.adult[row, col] = 1.0
    return state


def gamma_environment(
    temp_c: float,
    moisture: float,
    params: PhenologyParams,
) -> float:
    r"""\(\gamma(T,M)=f_T(T)g_M(M)\in[0,1]\)。

    \(M\) 为体积含水量（m³/m³）。\(M>M_{\mathrm{waterlog}}\) 时 \(g_M=0\)。
    """
    f_warm = 1.0 / (1.0 + np.exp(-params.k_T * (temp_c - params.T_base)))
    f_hot = 1.0 / (1.0 + np.exp(-params.k_ceil * (temp_c - params.T_max)))
    f_t = f_warm * (1.0 - f_hot)
    theta = max(float(moisture), 0.0)
    g_m = 0.0 if theta > params.M_waterlog else theta / (theta + params.K_m)
    return float(np.clip(f_t * g_m, 0.0, 1.0))


def seedling_survival(adult: NDArray[np.float64], params: PhenologyParams) -> NDArray[np.float64]:
    r"""密度依赖幼苗日存活 \(s_L(1-(A/K)^{\mu})\)，成株满格时幼苗存活→0。"""
    load = np.clip(adult / params.K_cell, 0.0, 1.0)
    return params.s_seedling * (1.0 - np.power(load, params.mu_dd))


def apply_adult_cap(adult: NDArray[np.float64], params: PhenologyParams) -> NDArray[np.float64]:
    r"""成株硬盖帽 \(A\leftarrow\min(A,K_{\mathrm{cell}})\)。"""
    return np.minimum(adult, params.K_cell)


def step_phenology(
    state: StageState,
    temp_c: float,
    moisture: float,
    params: PhenologyParams,
) -> StageState:
    """推进一日阶段转移：萌发、25 日龄推移、莲座→成株、自疏。不产籽。"""
    gamma = gamma_environment(temp_c, moisture, params)
    germ_rate = min(params.g_seed * gamma, 0.95)
    to_adult = min(params.g_rosette * gamma, 0.95)

    germinants = params.s_seed * germ_rate * state.seed
    seed_n = params.s_seed * (1.0 - germ_rate) * state.seed

    surv_l = seedling_survival(state.adult, params)
    aged = state.seedling * surv_l[np.newaxis, :, :]
    graduates = aged[-1]
    seedling_n = np.empty_like(state.seedling)
    seedling_n[0] = germinants
    if params.seedling_days > 1:
        seedling_n[1:] = aged[:-1]

    stay_r = params.s_rosette * (1.0 - to_adult)
    move_a = params.s_rosette * to_adult
    rosette_n = stay_r * state.rosette + graduates
    adult_n = apply_adult_cap(move_a * state.rosette + params.s_adult * state.adult, params)
    return StageState(seed=seed_n, seedling=seedling_n, rosette=rosette_n, adult=adult_n)


def seed_release_field(
    adult: NDArray[np.float64],
    releasing: bool,
    params: PhenologyParams,
) -> NDArray[np.float64]:
    """成株产籽场。非释放日为 0。

    .. math::

        F = 180\\, A\\, \\mathbf{1}_{\\mathrm{release}}.
    """
    if not releasing:
        return np.zeros_like(adult)
    return adult * params.n_seed_per_adult

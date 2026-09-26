"""Fisher–KPP 反应扩散与离散核的矩匹配。

连续方程

.. math::

    \\partial_t u = D\\nabla^2 u + r u\\bigl(1-u/K\\bigr),
    \\qquad
    c^*=2\\sqrt{rD}.

二维核的有效扩散与内禀增长

.. math::

    D=\\frac{M_2}{4\\tau},\\qquad
    r=\\frac{\\ln\\rho(L)}{\\tau},

其中 \(M_2\) 为羽流核二阶矩，\(\\rho(L)\) 为常气候 Lefkovitch 谱半径，\(\\tau=7\) 天。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.biology.lefkovitch import LefkovitchRates, climate_multiplier, spectral_radius
from src.biology.phenology import PhenologyParams
from src.params import Biophysics
from src.physics.dispersal_kernel import PappusAeroParams, build_hybrid_kernel
from src.physics.plume import build_plume_kernel, kernel_second_moment


def traveling_wave_speed(r: float, D: float) -> float:
    """解析最小波速 \(c^*=2\\sqrt{rD}\)（m / 时间单位）。"""
    if r < 0.0 or D < 0.0:
        raise ValueError("r and D must be nonnegative")
    return float(2.0 * np.sqrt(r * D))


def diffusion_from_kernel(params: Biophysics, wind_mps: float, wind_from_deg: float, tau: float) -> float:
    """由核二阶矩估 \(D=M_2/(4\\tau)\)。

    Parameters
    ----------
    tau :
        繁殖步长，与 \(r\) 同一时间单位（本包取 1 周）。
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    kernel = build_plume_kernel(params, wind_mps, wind_from_deg)
    moment = kernel_second_moment(kernel, params.dx_m)
    return float(moment / (4.0 * tau))


def intrinsic_rate(params: Biophysics, temp_c: float, smi: float, blooming: bool, tau: float = 1.0) -> float:
    """\(r=\\ln\\rho/\\tau\)。非花期 \(\\rho\) 不含产籽。"""
    from src.biology.lefkovitch import build_rates

    rates: LefkovitchRates = build_rates(temp_c, smi, params)
    phi = climate_multiplier(temp_c, smi, params)
    fec = 0.0
    if blooming:
        fec = params.n_seed_per_capitulum * params.n_capitula_per_adult_week_bloom * phi
    rho = spectral_radius(rates, fec)
    rho = max(rho, 1.0e-12)
    return float(np.log(rho) / tau)


def solve_fisher_1d(
    D: float,
    r: float,
    K: float,
    length_m: float,
    t_end: float,
    nx: int = 401,
    n_steps: int = 4000,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """一维 Fisher–KPP 显式步进，返回 \((x,u(t_{\\mathrm{end}}),c_{\\mathrm{num}})\)。

    左端小扰动初值；波前定义为 \(u=0.1K\) 的最右位置。
    CFL：自动把 ``n_steps`` 抬到 \(D\\Delta t/\\Delta x^2\\le 0.2\)。
    """
    if D <= 0.0 or r <= 0.0 or K <= 0.0:
        raise ValueError("D, r, K must be positive")
    x = np.linspace(0.0, length_m, nx, dtype=np.float64)
    dx = float(x[1] - x[0])
    dt_cfl = 0.20 * dx * dx / D
    dt = min(t_end / n_steps, dt_cfl)
    steps = int(np.ceil(t_end / dt))
    u = np.zeros(nx, dtype=np.float64)
    u[0:5] = 0.05 * K
    front_times: list[float] = []
    front_pos: list[float] = []
    sample_every = max(steps // 20, 1)
    for k in range(steps):
        lap = np.zeros_like(u)
        lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
        lap[0] = 0.0
        lap[-1] = (u[-2] - u[-1]) / (dx * dx)
        u = u + dt * (D * lap + r * u * (1.0 - u / K))
        u = np.clip(u, 0.0, K)
        if k % sample_every == 0:
            idx = np.where(u >= 0.1 * K)[0]
            if idx.size:
                front_times.append((k + 1) * dt)
                front_pos.append(float(x[int(idx[-1])]))
    c_num = 0.0
    if len(front_pos) >= 2:
        c_num = float((front_pos[-1] - front_pos[0]) / (front_times[-1] - front_times[0]))
    return x, u, c_num


TAU_DAY: float = 1.0


def diffusion_from_second_moment(m2_m2: float, tau: float) -> float:
    r"""二维核矩匹配 \(D=M_2/(4\tau)\)（m² / 时间单位）。"""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if m2_m2 < 0.0:
        raise ValueError("M2 must be nonnegative")
    return float(m2_m2 / (4.0 * tau))


def diffusion_from_wind_sigma(sigma_wind_sq: float, tau: float = TAU_DAY) -> float:
    r"""用户式 \(D\approx\frac12\sigma_{\mathrm{wind}}^2/\tau\)。

    若 \(\sigma_{\mathrm{wind}}^2\) 取单轴等价方差 \(M_2/2\)，则与 \(M_2/(4\tau)\) 恒等。
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if sigma_wind_sq < 0.0:
        raise ValueError("sigma_wind_sq must be nonnegative")
    return float(0.5 * sigma_wind_sq / tau)


def hybrid_kernel_second_moment(
    wind_mps: float,
    wind_from_deg: float,
    dx_m: float = 1.0,
    half_m: int = 40,
    aero: PappusAeroParams | None = None,
) -> float:
    """混合核离散二阶矩 \(M_2=\sum(x^2+y^2)K_{ij}\)（m²）。"""
    kernel = build_hybrid_kernel(
        wind_mps,
        wind_from_deg,
        dx_m=dx_m,
        half_m=half_m,
        aero=aero,
    )
    return kernel_second_moment(kernel, dx_m)


def diffusion_from_hybrid_kernel(
    wind_mps: float,
    wind_from_deg: float,
    tau: float = TAU_DAY,
    dx_m: float = 1.0,
    half_m: int = 40,
    aero: PappusAeroParams | None = None,
) -> tuple[float, float, float]:
    r"""由当日混合核估 \((D,M_2,\sigma_{\mathrm{wind}}^2)\)，\(\sigma_{\mathrm{wind}}^2=M_2/2\)。"""
    moment = hybrid_kernel_second_moment(wind_mps, wind_from_deg, dx_m, half_m, aero)
    sigma_sq = 0.5 * moment
    diff_d = diffusion_from_second_moment(moment, tau)
    return diff_d, moment, sigma_sq


def mean_diffusion_from_winds(
    wind_mps: NDArray[np.float64],
    wind_from_deg: NDArray[np.float64],
    tau: float = TAU_DAY,
    dx_m: float = 1.0,
    half_m: int = 40,
    aero: PappusAeroParams | None = None,
) -> tuple[float, float, float]:
    """对一组风样本平均 \(M_2\) 后再匹配 \(D\)。返回 \((D,\bar M_2,\bar\sigma^2)\)。"""
    if wind_mps.size == 0:
        raise ValueError("wind samples must be nonempty")
    cache: dict[tuple[int, int], float] = {}
    moments = np.empty(wind_mps.size, dtype=np.float64)
    for i, (speed, heading) in enumerate(zip(wind_mps, wind_from_deg)):
        key = (int(np.rint(max(float(speed), 0.0) * 2.0)), int(np.rint(float(heading) / 22.5)) % 16)
        hit = cache.get(key)
        if hit is None:
            hit = hybrid_kernel_second_moment(float(speed), float(heading), dx_m, half_m, aero)
            cache[key] = hit
        moments[i] = hit
    moment = float(np.mean(moments))
    sigma_sq = 0.5 * moment
    return diffusion_from_second_moment(moment, tau), moment, sigma_sq


def daily_projection_matrix(
    params: PhenologyParams,
    gamma: float,
    release_prob: float,
) -> NDArray[np.float64]:
    r"""低密度日步投影矩阵（无自疏）：\(S,L_0,\ldots,L_{d-1},R,A\)。

    成株列对种子库的贡献为 \(180\,p_{\mathrm{release}}\)。
    """
    n_l = int(params.seedling_days)
    n = 1 + n_l + 2
    i_s = 0
    i_l0 = 1
    i_r = 1 + n_l
    i_a = i_r + 1
    mat = np.zeros((n, n), dtype=np.float64)
    germ = min(params.g_seed * gamma, 0.95)
    to_adult = min(params.g_rosette * gamma, 0.95)
    mat[i_s, i_s] = params.s_seed * (1.0 - germ)
    mat[i_l0, i_s] = params.s_seed * germ
    for age in range(n_l - 1):
        mat[i_l0 + age + 1, i_l0 + age] = params.s_seedling
    mat[i_r, i_l0 + n_l - 1] = params.s_seedling
    mat[i_r, i_r] = params.s_rosette * (1.0 - to_adult)
    mat[i_a, i_r] = params.s_rosette * to_adult
    mat[i_a, i_a] = params.s_adult
    mat[i_s, i_a] = params.n_seed_per_adult * max(float(release_prob), 0.0)
    return mat


def intrinsic_rate_daily(
    params: PhenologyParams,
    gamma: float,
    release_prob: float,
    tau: float = TAU_DAY,
) -> float:
    r"""日步 \(r=\ln\rho/\tau\)，\(\rho\) 为 ``daily_projection_matrix`` 谱半径。"""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    mat = daily_projection_matrix(params, gamma, release_prob)
    rho = float(np.max(np.abs(np.linalg.eigvals(mat))))
    rho = max(rho, 1.0e-12)
    return float(np.log(rho) / tau)


def regress_front_speed(
    times: NDArray[np.float64],
    radii_m: NDArray[np.float64],
    r_cap_m: float = 42.0,
    n_late: int = 30,
) -> tuple[float, int, int]:
    """对未触边界的后期 \(R(t)\) 做直线回归，斜率即 \(c_{\mathrm{num}}\)（m/day）。

    返回 ``(c_num, i0, i1)``，区间为闭下标。点数不足时 \(c_{\mathrm{num}}=0\)。
    """
    if times.size != radii_m.size or times.size < 2:
        return 0.0, 0, 0
    expanding = radii_m < r_cap_m
    if not np.any(expanding):
        return 0.0, 0, 0
    last = int(np.flatnonzero(expanding)[-1])
    first = int(np.flatnonzero(expanding)[0])
    start = max(first, last - int(n_late) + 1)
    t_win = times[start : last + 1]
    r_win = radii_m[start : last + 1]
    if t_win.size < 2 or float(t_win[-1] - t_win[0]) <= 0.0:
        return 0.0, start, last
    slope = float(np.polyfit(t_win.astype(np.float64), r_win.astype(np.float64), 1)[0])
    return max(slope, 0.0), start, last


def relative_wave_error(c_num: float, c_star: float) -> float:
    r"""\(\lvert c_{\mathrm{num}}-c^*\rvert/\max(c^*,\varepsilon)\)。"""
    return float(abs(c_num - c_star) / max(c_star, 1.0e-9))

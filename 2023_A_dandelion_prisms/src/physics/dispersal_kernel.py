"""混合扩散核：近场二元高斯 + 热上升 WALD 长尾，再乘萌发抑制 \(\Phi(T,\\theta)\)。

列名锚定（禁止另造）：

- Open-Meteo 日表：``wind_speed``, ``wind_direction``, ``temperature``,
  ``soil_moisture``（0–7 cm VWC，m³/m³）。
- 情景小时表：``wind_speed_mps``, ``wind_from_deg``, ``air_temp_c``。
- 情景土壤：``smi``（无量纲，**不是** \(\\theta\)；本模块的 \(g_\\theta\) 要 VWC）。

气象来向下风向量与 ``plume.py`` 一致：\((-\\sin\\theta,-\\cos\\theta)\)。

.. math::

    K_{\\mathrm{disp}}=(1-p_{\\mathrm{up}})K_{\\mathrm{base}}+p_{\\mathrm{up}}K_{\\mathrm{up}},
    \\qquad
    \\Phi(T,\\theta)=f_T(T)\\,g_\\theta(\\theta).

\(K_{\\mathrm{base}}\) 为各向异性二元高斯，漂移 \(\\boldsymbol{\\mu}=\\mathbf{u}\\tau_{\\mathrm{fall}}\)。
\(K_{\\mathrm{up}}\) 为 \(H_{\\mathrm{up}}\\gg H_0\) 的 WALD 羽流，侧向方差再乘 \(\\beta_{\\mathrm{up}}\)。
离散核在衰减前强制 \(\\sum K\\,\\Delta x\\Delta y=1\)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.physics.plume import rotate_offsets
from src.physics.wald import wald_mean_shape, wald_pdf

# 与 data/raw 表头一致，禁止在调用侧发明别名。
OPEN_METEO_WIND_SPEED: str = "wind_speed"
OPEN_METEO_WIND_FROM: str = "wind_direction"
OPEN_METEO_TEMPERATURE: str = "temperature"
OPEN_METEO_VWC: str = "soil_moisture"
SCENARIO_WIND_SPEED: str = "wind_speed_mps"
SCENARIO_WIND_FROM: str = "wind_from_deg"
SCENARIO_TEMPERATURE: str = "air_temp_c"
SCENARIO_SMI: str = "smi"

U_FLOOR_MPS: float = 0.50
SIGMA_FLOOR_M: float = 1.0e-3


@dataclass(frozen=True, slots=True)
class PappusAeroParams:
    """分离涡环沉降与上升气流混合权重。

    Parameters
    ----------
    v_t :
        终端沉降速度 \(v_t\)（m/s），默认 0.28。
    H_0 :
        名义释放高度 \(H_0\)（m），默认 0.25。
    p_up :
        热上升卷吸概率 \(p_{\\mathrm{up}}\\in[0.01,0.05]\)。
    H_up :
        上升有效释放高度 \(H_{\\mathrm{up}}\\gg H_0\)（m）。
    beta_up :
        上升侧向方差放大 \(\\beta_{\\mathrm{up}}\\in[50,150]\)。
    alpha_u :
        风切变方差系数 \(\\alpha_u\)（s²），使 \(\\sigma^2=2D_T\\tau+\\alpha_u u^2\)。
    rho :
        近场高斯相关 \(\\rho\\in(-1,1)\)。
    kappa_up :
        上升 WALD 的 \(\\kappa=\\sigma_w/\\bar U\)。
    """

    v_t: float = 0.28
    H_0: float = 0.25
    p_up: float = 0.03
    H_up: float = 12.5
    beta_up: float = 80.0
    alpha_u: float = 0.40
    rho: float = 0.0
    kappa_up: float = 0.40

    def __post_init__(self) -> None:
        if self.v_t <= 0.0 or self.H_0 <= 0.0 or self.H_up <= 0.0:
            raise ValueError("v_t, H_0, H_up must be positive")
        if not 0.0 <= self.p_up <= 1.0:
            raise ValueError("p_up must lie in [0, 1]")
        if self.beta_up <= 0.0 or self.alpha_u < 0.0:
            raise ValueError("beta_up must be positive and alpha_u nonnegative")
        if abs(self.rho) >= 1.0:
            raise ValueError("rho must lie in (-1, 1)")


@dataclass(frozen=True, slots=True)
class GridConfig:
    """正交均匀格网。

    Parameters
    ----------
    x_min, x_max, y_min, y_max :
        域边界（m）。
    dx, dy :
        步长（m）。
    """

    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = 0.0
    y_max: float = 100.0
    dx: float = 1.0
    dy: float = 1.0

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("grid bounds must be ordered")
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError("dx and dy must be positive")


@dataclass(frozen=True, slots=True)
class GerminationParams:
    """萌发抑制 \(\\Phi(T,\\theta)\) 的热力与水分参数。

    Parameters
    ----------
    T_base, T_max :
        逻辑斯蒂升温阈与高温衰减中心（℃）。
    k_T, k_ceil :
        升温 / 热抑制斜率（1/℃）。
    K_theta :
        水分半饱和 \(K_\\theta\)（m³/m³）。
    theta_waterlog :
        渍水截断 \(\\theta_{\\mathrm{waterlog}}\)（m³/m³）。
    """

    T_base: float = 10.0
    T_max: float = 35.0
    k_T: float = 0.5
    k_ceil: float = 0.8
    K_theta: float = 0.12
    theta_waterlog: float = 0.45

    def __post_init__(self) -> None:
        if self.K_theta <= 0.0:
            raise ValueError("K_theta must be positive")
        if self.theta_waterlog <= 0.0:
            raise ValueError("theta_waterlog must be positive")


def compute_fall_time(height: float, vt: float = 0.28) -> float:
    """分离涡环终端速度下的沉降时间 \(\\tau=H/v_t\)（s）。"""
    if height <= 0.0 or vt <= 0.0:
        raise ValueError("height and vt must be positive")
    return float(height / vt)


def advective_wind_from_meteo(speed_mps: float, wind_from_deg: float) -> tuple[float, float]:
    """气象来向 → 东、北平流分量 \((u_x,u_y)\)（m/s）。

    与 ``plume.rotate_offsets`` 同一约定：下风单位向量
    \((-\\sin\\theta,-\\cos\\theta)\)。
    """
    speed = max(float(speed_mps), 0.0)
    theta = np.deg2rad(float(wind_from_deg))
    u_x = -speed * np.sin(theta)
    u_y = -speed * np.cos(theta)
    return float(u_x), float(u_y)


def wind_vector_from_record(row: Mapping[str, object]) -> tuple[float, float]:
    """从原始行读风速风向，只认已存在列名。"""
    if OPEN_METEO_WIND_SPEED in row and OPEN_METEO_WIND_FROM in row:
        speed = float(row[OPEN_METEO_WIND_SPEED])
        direction = float(row[OPEN_METEO_WIND_FROM])
    elif SCENARIO_WIND_SPEED in row and SCENARIO_WIND_FROM in row:
        speed = float(row[SCENARIO_WIND_SPEED])
        direction = float(row[SCENARIO_WIND_FROM])
    else:
        raise KeyError(
            "wind columns not found; expected "
            f"({OPEN_METEO_WIND_SPEED}, {OPEN_METEO_WIND_FROM}) or "
            f"({SCENARIO_WIND_SPEED}, {SCENARIO_WIND_FROM})"
        )
    return advective_wind_from_meteo(speed, direction)


def volumetric_moisture_from_record(row: Mapping[str, object]) -> float:
    """只接受 Open-Meteo ``soil_moisture``（m³/m³）。``smi`` 不是 \(\\theta\)。"""
    if OPEN_METEO_VWC in row:
        return float(row[OPEN_METEO_VWC])
    if SCENARIO_SMI in row:
        raise TypeError(
            f"{SCENARIO_SMI} is a dimensionless scenario index, not VWC; "
            f"pass {OPEN_METEO_VWC} from data/raw/soil/*_soil.csv"
        )
    raise KeyError(f"missing {OPEN_METEO_VWC}")


def temperature_from_record(row: Mapping[str, object]) -> float:
    """读气温（℃）：``temperature`` 或 ``air_temp_c``。"""
    if OPEN_METEO_TEMPERATURE in row:
        return float(row[OPEN_METEO_TEMPERATURE])
    if SCENARIO_TEMPERATURE in row:
        return float(row[SCENARIO_TEMPERATURE])
    raise KeyError(
        f"missing {OPEN_METEO_TEMPERATURE} or {SCENARIO_TEMPERATURE}"
    )


def _mesh_spacing(grid_x: NDArray[np.float64], grid_y: NDArray[np.float64]) -> tuple[float, float]:
    """由 meshgrid 估计 \(\\Delta x,\\Delta y\)。"""
    xs = np.unique(np.asarray(grid_x, dtype=np.float64))
    ys = np.unique(np.asarray(grid_y, dtype=np.float64))
    if xs.size < 2 or ys.size < 2:
        raise ValueError("grid_x and grid_y must contain at least two distinct nodes")
    dx = float(np.median(np.diff(xs)))
    dy = float(np.median(np.diff(ys)))
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("inferred dx, dy must be positive")
    return dx, dy


def _bivariate_gaussian(
    grid_x: NDArray[np.float64],
    grid_y: NDArray[np.float64],
    mu_x: float,
    mu_y: float,
    sigma_x: float,
    sigma_y: float,
    rho: float,
) -> NDArray[np.float64]:
    """向量化二元正态密度（1/m²）。"""
    sx = max(float(sigma_x), SIGMA_FLOOR_M)
    sy = max(float(sigma_y), SIGMA_FLOOR_M)
    corr = float(np.clip(rho, -0.95, 0.95))
    omr = 1.0 - corr * corr
    dx = grid_x - mu_x
    dy = grid_y - mu_y
    quad = (dx * dx) / (sx * sx) + (dy * dy) / (sy * sy) - (2.0 * corr * dx * dy) / (sx * sy)
    norm = 2.0 * np.pi * sx * sy * np.sqrt(omr)
    return np.exp(-0.5 * quad / omr) / norm


def _wald_updraft_density(
    grid_x: NDArray[np.float64],
    grid_y: NDArray[np.float64],
    wind_vector: tuple[float, float],
    params: PappusAeroParams,
) -> NDArray[np.float64]:
    """\(H_{\\mathrm{up}}\) WALD × 加宽侧向高斯（1/m²）。"""
    u_x, u_y = wind_vector
    speed = float(np.hypot(u_x, u_y))
    speed = max(speed, U_FLOOR_MPS)
    from_deg = float(np.rad2deg(np.arctan2(-u_x, -u_y))) % 360.0
    mu, lam = wald_mean_shape(params.H_up, speed, params.v_t, params.kappa_up)
    x_p, y_p = rotate_offsets(grid_x, grid_y, from_deg)
    along = wald_pdf(x_p, mu, lam)
    sigma0 = 0.80 * np.sqrt(params.beta_up)
    gamma = 0.35 * np.sqrt(params.beta_up)
    x_plus = np.maximum(x_p, 0.0)
    sigma_c = np.maximum(sigma0 + gamma * np.sqrt(x_plus), SIGMA_FLOOR_M)
    cross = np.exp(-0.5 * (y_p / sigma_c) ** 2) / (np.sqrt(2.0 * np.pi) * sigma_c)
    return along * cross


def _normalize_density(
    density: NDArray[np.float64],
    dx: float,
    dy: float,
) -> NDArray[np.float64]:
    """强制 \(\\sum K\\,\\Delta x\\Delta y=1\)。"""
    mass = float(np.sum(density) * dx * dy)
    if mass <= 0.0:
        raise ValueError("kernel mass vanished")
    return (density / mass).astype(np.float64)


def evaluate_bivariate_kernel(
    grid_x: NDArray[np.float64],
    grid_y: NDArray[np.float64],
    wind_vector: tuple[float, float],
    turbulent_diffusivity: tuple[float, float] = (0.15, 0.15),
    params: PappusAeroParams | None = None,
) -> NDArray[np.float64]:
    """在 meshgrid 上计算归一化 \(K_{\\mathrm{disp}}\)（1/m²）。

    Parameters
    ----------
    grid_x, grid_y :
        同形 meshgrid（m），东、北。
    wind_vector :
        平流 \((u_x,u_y)\)（m/s）。
    turbulent_diffusivity :
        \((D_{Tx},D_{Ty})\)（m²/s）。
    params :
        冠毛气动参数；默认 ``PappusAeroParams()``。
    """
    aero = params if params is not None else PappusAeroParams()
    d_tx, d_ty = turbulent_diffusivity
    if d_tx < 0.0 or d_ty < 0.0:
        raise ValueError("turbulent diffusivity must be nonnegative")
    tau = compute_fall_time(aero.H_0, aero.v_t)
    u_x, u_y = float(wind_vector[0]), float(wind_vector[1])
    mu_x = u_x * tau
    mu_y = u_y * tau
    sigma_x = float(np.sqrt(max(2.0 * d_tx * tau + aero.alpha_u * u_x * u_x, SIGMA_FLOOR_M**2)))
    sigma_y = float(np.sqrt(max(2.0 * d_ty * tau + aero.alpha_u * u_y * u_y, SIGMA_FLOOR_M**2)))
    base = _bivariate_gaussian(grid_x, grid_y, mu_x, mu_y, sigma_x, sigma_y, aero.rho)
    updraft = _wald_updraft_density(grid_x, grid_y, (u_x, u_y), aero)
    mixture = (1.0 - aero.p_up) * base + aero.p_up * updraft
    dx, dy = _mesh_spacing(grid_x, grid_y)
    return _normalize_density(np.maximum(mixture, 0.0), dx, dy)


def calculate_germination_modifier(
    temperature: float | np.ndarray,
    soil_moisture: float | np.ndarray,
    params: GerminationParams | None = None,
) -> float | NDArray[np.float64]:
    """环境萌发因子 \(\\Phi(T,\\theta)\\in[0,1]\)，对数组向量化。

    \(T\) 为 ℃；\(\\theta\) 为 ``soil_moisture`` 的 m³/m³，不是 ``smi``。
    """
    germ = params if params is not None else GerminationParams()
    temp = np.asarray(temperature, dtype=np.float64)
    theta = np.asarray(soil_moisture, dtype=np.float64)
    f_warm = 1.0 / (1.0 + np.exp(-germ.k_T * (temp - germ.T_base)))
    f_hot = 1.0 / (1.0 + np.exp(-germ.k_ceil * (temp - germ.T_max)))
    f_t = f_warm * (1.0 - f_hot)
    moisture = np.maximum(theta, 0.0)
    g_theta = moisture / (moisture + germ.K_theta)
    g_theta = np.where(theta <= germ.theta_waterlog, g_theta, 0.0)
    phi = np.clip(f_t * g_theta, 0.0, 1.0)
    if np.ndim(temperature) == 0 and np.ndim(soil_moisture) == 0:
        return float(phi)
    return phi.astype(np.float64)


def cell_center_mesh(grid_cfg: GridConfig) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """由 ``GridConfig`` 生成格心 meshgrid ``(X,Y)``。"""
    n_x = int(np.round((grid_cfg.x_max - grid_cfg.x_min) / grid_cfg.dx))
    n_y = int(np.round((grid_cfg.y_max - grid_cfg.y_min) / grid_cfg.dy))
    if n_x < 2 or n_y < 2:
        raise ValueError("grid must have at least 2 cells on each axis")
    x = grid_cfg.x_min + (np.arange(n_x, dtype=np.float64) + 0.5) * grid_cfg.dx
    y = grid_cfg.y_min + (np.arange(n_y, dtype=np.float64) + 0.5) * grid_cfg.dy
    return np.meshgrid(x, y, indexing="xy")


def compute_effective_seed_flux(
    seed_count: int,
    wind_vector: tuple[float, float],
    temperature: float,
    soil_moisture: float,
    grid_cfg: GridConfig,
    turbulent_diffusivity: tuple[float, float] = (0.15, 0.15),
    aero: PappusAeroParams | None = None,
    germ: GerminationParams | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """返回 \((X,Y,\\rho_{\\mathrm{eff}})\)，\(\\rho_{\\mathrm{eff}}=N K_{\\mathrm{disp}}\\Phi\)。

    环境衰减前 \(\\sum (N K)\\Delta x\\Delta y=N\)。\(\\Phi\) 之后不再守恒。

    Parameters
    ----------
    seed_count :
        释放粒数 \(N\\ge 0\)。
    wind_vector :
        \((u_x,u_y)\)（m/s）。
    temperature :
        ℃，对应 ``temperature`` / ``air_temp_c``。
    soil_moisture :
        m³/m³，对应 ``soil_moisture``。
    grid_cfg :
        正交格网。
    """
    if seed_count < 0:
        raise ValueError("seed_count must be nonnegative")
    grid_x, grid_y = cell_center_mesh(grid_cfg)
    kernel = evaluate_bivariate_kernel(
        grid_x,
        grid_y,
        wind_vector,
        turbulent_diffusivity=turbulent_diffusivity,
        params=aero,
    )
    landing = float(seed_count) * kernel
    phi = calculate_germination_modifier(temperature, soil_moisture, germ)
    effective = landing * float(phi)
    return grid_x, grid_y, effective.astype(np.float64)


def kernel_area_mass(density: NDArray[np.float64], grid_cfg: GridConfig) -> float:
    """离散积分 \(\\sum K\\Delta x\\Delta y\)。"""
    return float(np.sum(density) * grid_cfg.dx * grid_cfg.dy)


def wind_vector_from_open_meteo_arrays(
    wind_speed: Sequence[float],
    wind_direction: Sequence[float],
) -> NDArray[np.float64]:
    """批量把 Open-Meteo 列转为 \((u_x,u_y)\)，形状 ``(n, 2)``。"""
    speed = np.asarray(wind_speed, dtype=np.float64)
    direction = np.asarray(wind_direction, dtype=np.float64)
    theta = np.deg2rad(direction)
    u_x = -speed * np.sin(theta)
    u_y = -speed * np.cos(theta)
    return np.column_stack((u_x, u_y))


def hybrid_kernel_half_m(
    wind_mps: float,
    dx_m: float,
    min_half_m: int,
    aero: PappusAeroParams | None = None,
    max_half_m: int = 120,
) -> int:
    """卷积核半宽（m）：至少包住上升 WALD 均值加一段尾。"""
    params = aero if aero is not None else PappusAeroParams()
    speed = max(float(wind_mps), U_FLOOR_MPS)
    mu_up = params.H_up * speed / params.v_t
    needed = int(np.ceil(mu_up + 4.0 * np.sqrt(params.beta_up)))
    half = max(int(min_half_m), needed)
    half = min(half, int(max_half_m))
    if dx_m != 1.0:
        half = int(np.ceil(half / dx_m))
    return max(half, 1)


def build_hybrid_kernel(
    wind_mps: float,
    wind_from_deg: float,
    dx_m: float,
    half_m: int,
    aero: PappusAeroParams | None = None,
    turbulent_diffusivity: tuple[float, float] = (0.15, 0.15),
) -> NDArray[np.float64]:
    """以源为中心的奇数边长离散核，元素和为 1，供 ``fftconvolve``。

    Parameters
    ----------
    wind_mps, wind_from_deg :
        气象风速（m/s）与来向（°）。
    dx_m :
        格距（m），须与仿真格网一致。
    half_m :
        半宽（m）；边长 ``2*half_m/dx+1``。
    """
    if dx_m <= 0.0 or half_m <= 0:
        raise ValueError("dx_m and half_m must be positive")
    n = 2 * int(round(half_m / dx_m)) + 1
    axis = (np.arange(n, dtype=np.float64) - (n // 2)) * dx_m
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    wind = advective_wind_from_meteo(wind_mps, wind_from_deg)
    density = evaluate_bivariate_kernel(
        grid_x,
        grid_y,
        wind,
        turbulent_diffusivity=turbulent_diffusivity,
        params=aero,
    )
    kernel = density * (dx_m * dx_m)
    total = float(np.sum(kernel))
    if total <= 0.0:
        raise ValueError("hybrid kernel mass vanished")
    return (kernel / total).astype(np.float64)


def deposit_hybrid_from_point(
    n_seeds: float,
    x_src_m: float,
    y_src_m: float,
    x_centers: NDArray[np.float64],
    y_centers: NDArray[np.float64],
    wind_mps: float,
    wind_from_deg: float,
    dx_m: float,
    half_m: int,
    aero: PappusAeroParams | None = None,
) -> tuple[NDArray[np.float64], float]:
    """把混合核盖到格网上。质量账：``n_seeds = sum(field) + mass_outside``。"""
    if n_seeds < 0.0:
        raise ValueError("n_seeds must be nonnegative")
    kernel = build_hybrid_kernel(wind_mps, wind_from_deg, dx_m, half_m, aero=aero)
    n = int(x_centers.size)
    field = np.zeros((n, n), dtype=np.float64)
    col0 = int(np.rint((x_src_m - float(x_centers[0])) / dx_m))
    row0 = int(np.rint((y_src_m - float(y_centers[0])) / dx_m))
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
    if src_r1 > src_r0 and src_c1 > src_c0:
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        field[dst_r0:dst_r1, dst_c0:dst_c1] = kernel[src_r0:src_r1, src_c0:src_c1] * n_seeds
    inside = float(np.sum(field))
    return field, n_seeds - inside

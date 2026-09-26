"""周步空间引擎：Lefkovitch → 产籽 → 混合核卷积 → 自疏 → 可选割草。

籽雨分解

.. math::

    F=(1-\\varepsilon)F_{\\mathrm{disp}}+\\varepsilon F,\\qquad
    S\\leftarrow S+\\Phi\\bigl(\\varepsilon F+\\mathcal{K}*F_{\\mathrm{disp}}+F_{\\mathrm{src}}\\bigr).

\(\\mathcal{K}=K_{\\mathrm{disp}}\) 为 ``dispersal_kernel`` 的高斯+WALD 混合核。
\(\\Phi(T,\\theta)\) 只在强迫带有 Open-Meteo ``soil_moisture``（m³/m³）时作用；
情景 ``smi`` 只进入 Lefkovitch，不当作 \(\\theta\)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from src.biology.lefkovitch import build_rates, fecundity_seeds, lefkovitch_step
from src.biology.self_thinning import apply_self_thinning
from src.params import Biophysics
from src.physics.dispersal_kernel import (
    GerminationParams,
    PappusAeroParams,
    build_hybrid_kernel,
    calculate_germination_modifier,
    deposit_hybrid_from_point,
    hybrid_kernel_half_m,
)
from src.simulation.grid import GridState, cell_centers, empty_grid, front_extent_m, occupied_metrics


@dataclass(frozen=True, slots=True)
class WeekForcing:
    """单周强迫。

    Parameters
    ----------
    week :
        年周编号，1–52。
    month :
        日历月 1–12。
    temp_c, wind_mps, wind_from_deg, smi :
        周平均气温、风速、来向、情景土壤指数（Lefkovitch）。
    blooming :
        当月是否允许开花结籽。
    mow_eta :
        割草强度 \(\\eta\\in[0,1]\)；0 表示本周不割。
    soil_moisture :
        Open-Meteo 0–7 cm VWC（m³/m³）。``None`` 时 \(\\Phi=1\)。
    """

    week: int
    month: int
    temp_c: float
    wind_mps: float
    wind_from_deg: float
    smi: float
    blooming: bool
    mow_eta: float = 0.0
    soil_moisture: float | None = None


_TINY: float = 1.0e-12


def _floor_tiny(field: NDArray[np.float64]) -> NDArray[np.float64]:
    """把低于 \(10^{-12}\) 的密度置零，避免离岸风情景放大舍入误差。"""
    cleaned = field.copy()
    cleaned[cleaned < _TINY] = 0.0
    return cleaned


def _fence_local_drop(n_seeds: float, params: Biophysics) -> NDArray[np.float64]:
    """源株就近滞留：质量落在西缘、最靠近 \(y_{\\mathrm{src}}\) 的三格，权和为 ``n_seeds``。"""
    n = int(params.grid_n)
    field = np.zeros((n, n), dtype=np.float64)
    if n_seeds <= 0.0:
        return field
    y_c = (np.arange(n, dtype=np.float64) + 0.5) * float(params.dx_m)
    row = int(np.argmin(np.abs(y_c - params.source_y_m)))
    weights = np.array([0.25, 0.50, 0.25], dtype=np.float64)
    for offset, weight in zip((-1, 0, 1), weights):
        rr = row + offset
        if 0 <= rr < n:
            field[rr, 0] += n_seeds * weight
    leftover = n_seeds - float(np.sum(field))
    if leftover > 0.0:
        field[row, 0] += leftover
    return field


def _convolve_seed_rain(field: NDArray[np.float64], kernel: NDArray[np.float64]) -> NDArray[np.float64]:
    """周期外零填充的 FFT 卷积，输出与 ``field`` 同形。"""
    rain = fftconvolve(field, kernel, mode="same")
    rain = np.maximum(rain, 0.0)
    return rain.astype(np.float64)


def step_week(
    state: GridState,
    forcing: WeekForcing,
    params: Biophysics,
    inject_source: bool,
) -> tuple[GridState, dict[str, float]]:
    """推进一周，返回新状态与质量账。

    Parameters
    ----------
    inject_source :
        是否投放地块外绒球源（第 0 周必投；其后仅当 ``source_persists`` 且花期）。
    """
    rates = build_rates(forcing.temp_c, forcing.smi, params)
    seed, seedling, rosette, adult = lefkovitch_step(
        state.seed, state.seedling, state.rosette, state.adult, rates
    )
    seeds_aborted = 0.0
    if forcing.mow_eta > 0.0:
        keep = 1.0 - min(max(forcing.mow_eta, 0.0), 1.0)
        seeds_aborted = float(np.sum(fecundity_seeds(adult, forcing.blooming, forcing.temp_c, forcing.smi, params))) * (
            1.0 - keep
        )
        rosette = rosette * keep
        adult = adult * keep

    produced = fecundity_seeds(adult, forcing.blooming, forcing.temp_c, forcing.smi, params)
    local = params.local_retention * produced
    dispersed = (1.0 - params.local_retention) * produced
    aero = PappusAeroParams()
    germ = GerminationParams()
    half_m = hybrid_kernel_half_m(forcing.wind_mps, params.dx_m, params.kernel_half_m, aero)
    kernel = build_hybrid_kernel(
        forcing.wind_mps,
        forcing.wind_from_deg,
        params.dx_m,
        half_m,
        aero=aero,
    )
    rain = _convolve_seed_rain(dispersed, kernel)
    x_c, y_c = cell_centers(params)
    source_field = np.zeros_like(seed)
    source_local = np.zeros_like(seed)
    source_outside = 0.0
    source_released = 0.0
    if inject_source:
        source_released = float(params.n_seed_initial_puffball)
        if params.source_persists and forcing.week > 1:
            if forcing.blooming:
                source_released = (
                    params.n_seed_per_capitulum * params.n_capitula_per_adult_week_bloom
                )
            else:
                source_released = 0.0
        if source_released > 0.0:
            n_local = params.local_retention * source_released
            n_far = (1.0 - params.local_retention) * source_released
            source_local = _fence_local_drop(n_local, params)
            source_field, source_outside = deposit_hybrid_from_point(
                n_far,
                params.source_x_m,
                params.source_y_m,
                x_c,
                y_c,
                forcing.wind_mps,
                forcing.wind_from_deg,
                params.dx_m,
                half_m,
                aero=aero,
            )

    phi = 1.0
    if forcing.soil_moisture is not None:
        phi = float(calculate_germination_modifier(forcing.temp_c, forcing.soil_moisture, germ))
    local = local * phi
    rain = rain * phi
    source_field = source_field * phi
    source_local = source_local * phi

    seed = seed + local + rain + source_field + source_local
    seed = _floor_tiny(seed)
    seedling = _floor_tiny(seedling)
    rosette = _floor_tiny(rosette)
    adult = _floor_tiny(adult)
    rosette, adult = apply_self_thinning(rosette, adult, forcing.smi, params)
    nxt = GridState(seed=seed, seedling=seedling, rosette=rosette, adult=adult)
    ledger = {
        "produced_seeds": float(np.sum(produced)),
        "local_seeds": float(np.sum(local)),
        "dispersed_seeds": float(np.sum(dispersed)),
        "rain_on_grid": float(np.sum(rain)),
        "source_released": source_released,
        "source_on_grid": float(np.sum(source_field) + np.sum(source_local)),
        "source_outside": source_outside,
        "seeds_aborted": seeds_aborted,
        "phi": phi,
        "kernel_half_m": float(half_m),
    }
    return nxt, ledger


def run_year(
    forcings: list[WeekForcing],
    params: Biophysics,
    snapshot_weeks: tuple[int, ...] = (4, 8, 13, 26, 52),
    on_week: Callable[[int, GridState], None] | None = None,
) -> tuple[GridState, list[dict[str, float]], dict[int, NDArray[np.float64]]]:
    """从空格网跑完整年（或给定周列），记录月末指标与快照。

    Parameters
    ----------
    forcings :
        按周排序的强迫，长度通常为 52。
    snapshot_weeks :
        保存建成株热图的周号。
    on_week :
        可选回调。

    Returns
    -------
    GridState
        年末状态。
    list[dict]
        每周指标行。
    dict[int, ndarray]
        周号 → \(L+R+A\) 热图。
    """
    state = empty_grid(params)
    rows: list[dict[str, float]] = []
    snaps: dict[int, NDArray[np.float64]] = {}
    for idx, forcing in enumerate(forcings):
        inject = idx == 0 or (params.source_persists and forcing.blooming)
        if idx == 0:
            inject = True
        state, ledger = step_week(state, forcing, params, inject_source=inject)
        metrics = occupied_metrics(state)
        row: dict[str, float] = {
            "week": float(forcing.week),
            "month": float(forcing.month),
            "temp_c": forcing.temp_c,
            "wind_mps": forcing.wind_mps,
            "smi": forcing.smi,
            "soil_moisture": float("nan") if forcing.soil_moisture is None else forcing.soil_moisture,
            "blooming": 1.0 if forcing.blooming else 0.0,
            "mow_eta": forcing.mow_eta,
            "front_m": front_extent_m(state, params.dx_m),
            **metrics,
            **ledger,
        }
        rows.append(row)
        if forcing.week in snapshot_weeks:
            snaps[forcing.week] = state.plants.copy()
        if on_week is not None:
            on_week(forcing.week, state)
    return state, rows, snaps

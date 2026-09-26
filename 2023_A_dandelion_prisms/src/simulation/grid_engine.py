r"""1 公顷日步空间引擎：成株场 × 180 与当日风散核做吸收边界卷积。

.. math::

    \mathbf{S}_{\mathrm{new}}=
    \mathrm{convolve2d}(180\,\mathbf{A},\,\mathcal{K}_t,\ \mathrm{mode=same}).

``mode='same'`` 零填充即吸收边界：飞出 \(100\times 100\) 的籽质量丢失。
禁止逐籽粒子循环。生活史只调用 ``phenology``，核只调用 ``dispersal_kernel``。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import convolve2d

from src.biology.phenology import (
    PhenologyParams,
    StageState,
    initial_center_adult,
    seed_release_field,
    step_phenology,
)
from src.physics.dispersal_kernel import (
    OPEN_METEO_TEMPERATURE,
    OPEN_METEO_VWC,
    OPEN_METEO_WIND_FROM,
    OPEN_METEO_WIND_SPEED,
    PappusAeroParams,
    build_hybrid_kernel,
)

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
GRID_N: int = 100
DX_M: float = 1.0
KERNEL_HALF_M: int = 40
CENTER_ROW: int = 50
CENTER_COL: int = 50
DEFAULT_BLOOM_MONTHS: tuple[int, ...] = (4, 5, 6, 9, 10)


@dataclass(frozen=True, slots=True)
class DailyForcing:
    r"""单日强迫。

    Parameters
    ----------
    day :
        年积日 0–364。
    temp_c, wind_mps, wind_from_deg, moisture :
        气温（℃）、风速（m/s）、气象来向（°）、VWC（m³/m³）。
    releasing :
        本日是否按 \(180 A\) 释放。
    """

    day: int
    temp_c: float
    wind_mps: float
    wind_from_deg: float
    moisture: float
    releasing: bool


@dataclass
class HectareResult:
    """365 日积分摘要。"""

    state: StageState
    n_adult: NDArray[np.float64]
    n_seed_rain: NDArray[np.float64]
    mass_lost: float
    elapsed_s: float


def _month_of_doy(doy: int) -> int:
    """非闰年年积日 0–364 → 月 1–12。"""
    month_ends = (31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)
    day = doy + 1
    for month, end in enumerate(month_ends, start=1):
        if day <= end:
            return month
    return 12


def load_daily_forcing(
    climate: str = "temperate",
    n_days: int = 365,
    bloom_months: tuple[int, ...] = DEFAULT_BLOOM_MONTHS,
    weather_dir: Path | None = None,
    soil_dir: Path | None = None,
    force_day0_release: bool = True,
) -> list[DailyForcing]:
    """优先读 Open-Meteo ``*_weather.csv`` / ``*_soil.csv``；缺文件则用常气候。"""
    wdir = weather_dir or (PACK_ROOT / "data" / "raw" / "weather")
    sdir = soil_dir or (PACK_ROOT / "data" / "raw" / "soil")
    weather_path = wdir / f"{climate}_weather.csv"
    soil_path = sdir / f"{climate}_soil.csv"
    bloom = set(bloom_months)
    if weather_path.is_file() and soil_path.is_file():
        weather = pd.read_csv(weather_path)
        soil = pd.read_csv(soil_path)
        for col in (OPEN_METEO_WIND_SPEED, OPEN_METEO_WIND_FROM, OPEN_METEO_TEMPERATURE):
            if col not in weather.columns:
                raise KeyError(f"{weather_path.name} missing {col}")
        if OPEN_METEO_VWC not in soil.columns:
            raise KeyError(f"{soil_path.name} missing {OPEN_METEO_VWC}")
        n = min(n_days, len(weather), len(soil))
        rows: list[DailyForcing] = []
        for i in range(n):
            month = _month_of_doy(i)
            releasing = month in bloom or (force_day0_release and i == 0)
            rows.append(
                DailyForcing(
                    day=i,
                    temp_c=float(weather.iloc[i][OPEN_METEO_TEMPERATURE]),
                    wind_mps=float(weather.iloc[i][OPEN_METEO_WIND_SPEED]),
                    wind_from_deg=float(weather.iloc[i][OPEN_METEO_WIND_FROM]),
                    moisture=float(soil.iloc[i][OPEN_METEO_VWC]),
                    releasing=releasing,
                )
            )
        return rows
    return [
        DailyForcing(
            day=i,
            temp_c=18.0,
            wind_mps=3.5,
            wind_from_deg=270.0,
            moisture=0.20,
            releasing=( _month_of_doy(i) in bloom) or (force_day0_release and i == 0),
        )
        for i in range(n_days)
    ]


def disperse_seeds(
    adult: NDArray[np.float64],
    kernel: NDArray[np.float64],
    params: PhenologyParams,
    releasing: bool,
) -> tuple[NDArray[np.float64], float]:
    r"""吸收边界卷积。返回（格内新籽，飞出域的粒数）。

    \(\sum F - \sum S_{\mathrm{new}}\) 为边界吸收量。
    """
    source = seed_release_field(adult, releasing, params)
    released = float(np.sum(source))
    if released <= 0.0:
        return np.zeros_like(adult), 0.0
    landed = convolve2d(source, kernel, mode="same", boundary="fill", fillvalue=0.0)
    landed = np.maximum(landed, 0.0)
    lost = released - float(np.sum(landed))
    return landed.astype(np.float64), max(lost, 0.0)


def _kernel_cache_key(wind_mps: float, wind_from_deg: float) -> tuple[int, int]:
    """把风量化到 (0.5 m/s, 22.5°) 以便复用核。"""
    speed_bin = int(np.rint(max(wind_mps, 0.0) * 2.0))
    dir_bin = int(np.rint(wind_from_deg / 22.5)) % 16
    return speed_bin, dir_bin


def kernel_for_wind(
    wind_mps: float,
    wind_from_deg: float,
    cache: dict[tuple[int, int], NDArray[np.float64]],
    aero: PappusAeroParams,
) -> NDArray[np.float64]:
    """取或建当日离散核，元素和为 1。"""
    key = _kernel_cache_key(wind_mps, wind_from_deg)
    hit = cache.get(key)
    if hit is not None:
        return hit
    kernel = build_hybrid_kernel(
        wind_mps,
        wind_from_deg,
        dx_m=DX_M,
        half_m=KERNEL_HALF_M,
        aero=aero,
    )
    cache[key] = kernel
    return kernel


def run_hectare_days(
    forcings: list[DailyForcing],
    params: PhenologyParams | None = None,
    aero: PappusAeroParams | None = None,
    n: int = GRID_N,
    on_day: Callable[[int, StageState], None] | None = None,
    intervene: Callable[[DailyForcing, StageState], StageState] | None = None,
) -> HectareResult:
    """从中央 1 株成株积分 ``len(forcings)`` 日（默认 365）。

    ``intervene`` 在当日卷积产籽之前作用（先修剪再飞散）。
    """
    import time

    life = params if params is not None else PhenologyParams()
    aero_p = aero if aero is not None else PappusAeroParams()
    state = initial_center_adult(n, life, CENTER_ROW, CENTER_COL)
    cache: dict[tuple[int, int], NDArray[np.float64]] = {}
    n_adult = np.zeros(len(forcings), dtype=np.float64)
    n_rain = np.zeros(len(forcings), dtype=np.float64)
    lost_total = 0.0
    t0 = time.perf_counter()
    for i, forcing in enumerate(forcings):
        if intervene is not None:
            state = intervene(forcing, state)
        kernel = kernel_for_wind(forcing.wind_mps, forcing.wind_from_deg, cache, aero_p)
        rain, lost = disperse_seeds(state.adult, kernel, life, forcing.releasing)
        state.seed = state.seed + rain
        lost_total += lost
        state = step_phenology(state, forcing.temp_c, forcing.moisture, life)
        n_adult[i] = float(np.sum(state.adult))
        n_rain[i] = float(np.sum(rain))
        if on_day is not None:
            on_day(forcing.day, state)
    elapsed = time.perf_counter() - t0
    return HectareResult(
        state=state,
        n_adult=n_adult,
        n_seed_rain=n_rain,
        mass_lost=lost_total,
        elapsed_s=elapsed,
    )


def run_temperate_365() -> HectareResult:
    """温带 Open-Meteo 2022（若存在）跑满 365 日。"""
    return run_hectare_days(load_daily_forcing("temperate", n_days=365))

"""100 m × 100 m 格网状态与空间统计。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.params import Biophysics


@dataclass
class GridState:
    """四阶段密度场，单位：粒或株 / 格（默认 1 m²）。

    Attributes
    ----------
    seed, seedling, rosette, adult :
        形状 ``(n, n)`` 的 float64 数组。
    """

    seed: NDArray[np.float64]
    seedling: NDArray[np.float64]
    rosette: NDArray[np.float64]
    adult: NDArray[np.float64]

    @property
    def plants(self) -> NDArray[np.float64]:
        """建成株密度 \(L+R+A\)。"""
        return self.seedling + self.rosette + self.adult

    @property
    def canopy(self) -> NDArray[np.float64]:
        """自疏对象 \(R+A\)。"""
        return self.rosette + self.adult


def empty_grid(params: Biophysics) -> GridState:
    """全零 1 公顷格网。"""
    n = int(params.grid_n)
    z = np.zeros((n, n), dtype=np.float64)
    return GridState(seed=z.copy(), seedling=z.copy(), rosette=z.copy(), adult=z.copy())


def cell_centers(params: Biophysics) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """东向、北向格心坐标（m），原点在地块西南角。"""
    n = int(params.grid_n)
    dx = float(params.dx_m)
    axis = (np.arange(n, dtype=np.float64) + 0.5) * dx
    return axis, axis


def occupied_metrics(state: GridState, plant_threshold: float = 0.05) -> dict[str, float]:
    """覆盖与种群合计。

    Parameters
    ----------
    plant_threshold :
        格被占的阈值（株/m²）。
    """
    plants = state.plants
    cover = float(np.mean(plants >= plant_threshold))
    return {
        "n_seed": float(np.sum(state.seed)),
        "n_seedling": float(np.sum(state.seedling)),
        "n_rosette": float(np.sum(state.rosette)),
        "n_adult": float(np.sum(state.adult)),
        "n_plants": float(np.sum(plants)),
        "cover_frac": cover,
        "max_canopy": float(np.max(state.canopy)),
    }


def front_extent_m(state: GridState, dx_m: float, plant_threshold: float = 0.05) -> float:
    """从西缘量起，存在建成株的最大东向距离（m）。"""
    occupied = state.plants >= plant_threshold
    if not np.any(occupied):
        return 0.0
    cols = np.where(occupied.any(axis=0))[0]
    east_index = int(cols.max())
    return float((east_index + 0.5) * dx_m)

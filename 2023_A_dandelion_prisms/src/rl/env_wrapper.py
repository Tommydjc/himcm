"""Gymnasium 封装：复用 ``phenology`` 与日强迫，不用不存在的引擎类名。

观测 8 维；动作 Discrete(4)。空间用 10×10 格（10 m/格），外圈 1 格 = 10 m 缓冲带，
以便 200+ 回合在约 2 分钟内训完。生活史仍调用 ``step_phenology``。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from src.biology.phenology import (
    PhenologyParams,
    StageState,
    initial_center_adult,
    seed_release_field,
    step_phenology,
)
from src.decision.tradeoff_model import apply_chemical, apply_mechanical, month_of_doy
from src.simulation.grid_engine import DailyForcing, load_daily_forcing

OBS_DIM: int = 8
N_ACTIONS: int = 4
GRID_N: int = 10
CENTER: int = 5
OCCUPY: float = 0.05
SPRING_MONTHS: frozenset[int] = frozenset({3, 4})
ACTION_COST: tuple[float, ...] = (0.00, 0.35, 0.65, 1.00)
ACTION_NAMES: tuple[str, ...] = ("NO_OP", "BUFFER_MOW", "CORE_MOW", "PRECISION_SPRAY")
SPRING_PENALTY: float = 15.0


def _buffer_mask(n: int) -> NDArray[np.bool_]:
    """最外圈 True（10 m 缓冲带）。"""
    mask = np.ones((n, n), dtype=np.bool_)
    mask[1:-1, 1:-1] = False
    return mask


def _absorbing_leak(field: NDArray[np.float64], stay: float = 0.72) -> NDArray[np.float64]:
    """四邻域吸收扩散：出界质量丢失，禁止逐籽循环。"""
    hop = (1.0 - stay) * 0.25
    acc = stay * field
    up = np.zeros_like(field)
    down = np.zeros_like(field)
    left = np.zeros_like(field)
    right = np.zeros_like(field)
    up[:-1] = field[1:]
    down[1:] = field[:-1]
    left[:, :-1] = field[:, 1:]
    right[:, 1:] = field[:, :-1]
    return acc + hop * (up + down + left + right)


class DandelionEcoEnv(gym.Env):
    """一年 365 步的离散管理环境。

    底层是 ``PhenologyParams`` / ``step_phenology`` / ``load_daily_forcing``，
    不是虚构的 ``DandelionGridEngine``。
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        climate: str = "temperate",
        weather_seed: int | None = None,
        params: PhenologyParams | None = None,
    ) -> None:
        super().__init__()
        self.params = params if params is not None else PhenologyParams()
        self.climate = climate
        self.weather_seed = weather_seed
        self.base_forcings = load_daily_forcing(climate, n_days=365)
        if len(self.base_forcings) < 365:
            raise RuntimeError("need 365 daily forcing rows")
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.buffer = _buffer_mask(GRID_N)
        self.forcings: list[DailyForcing] = list(self.base_forcings)
        self.state: StageState = initial_center_adult(GRID_N, self.params, CENTER, CENTER)
        self.day_index: int = 0
        self.cum_cost: float = 0.0
        self.n_mows: int = 0
        self.n_sprays: int = 0
        self.bee_sum: float = 0.0

    def _perturbed_forcings(self, seed: int | None) -> list[DailyForcing]:
        """同一气象种子下对 Open-Meteo 加可复现微扰。"""
        if seed is None:
            return list(self.base_forcings)
        rng = np.random.default_rng(int(seed))
        rows: list[DailyForcing] = []
        for row in self.base_forcings:
            rows.append(
                DailyForcing(
                    day=row.day,
                    temp_c=float(row.temp_c + rng.normal(0.0, 0.4)),
                    wind_mps=max(0.3, float(row.wind_mps + rng.normal(0.0, 0.15))),
                    wind_from_deg=float(row.wind_from_deg),
                    moisture=float(np.clip(row.moisture + rng.normal(0.0, 0.01), 0.02, 0.55)),
                    releasing=row.releasing,
                )
            )
        return rows

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        weather = self.weather_seed if options is None else options.get("weather_seed", self.weather_seed)
        if seed is not None and weather is None:
            weather = seed
        self.forcings = self._perturbed_forcings(weather)
        self.state = initial_center_adult(GRID_N, self.params, CENTER, CENTER)
        self.day_index = 0
        self.cum_cost = 0.0
        self.n_mows = 0
        self.n_sprays = 0
        self.bee_sum = 0.0
        return self._obs(), {"day": 0}

    def _apply_action(self, action: int) -> float:
        """执行动作并返回当日成本。"""
        cost = float(ACTION_COST[int(action)])
        if action == 0:
            return cost
        if action == 1:
            keep = np.where(self.buffer, 0.15, 1.0)
            self.state = StageState(
                seed=self.state.seed,
                seedling=self.state.seedling * np.where(self.buffer, 0.80, 1.0)[np.newaxis, :, :],
                rosette=self.state.rosette * np.where(self.buffer, 0.55, 1.0),
                adult=self.state.adult * keep,
            )
            self.n_mows += 1
            return cost
        if action == 2:
            self.state = apply_mechanical(self.state, 0.90)
            self.n_mows += 1
            return cost
        density = self.state.adult + self.state.rosette
        hot = density >= max(0.35 * self.params.K_cell, float(np.quantile(density, 0.80)))
        if not np.any(hot):
            hot = density >= float(np.max(density)) * 0.5
            if not np.any(hot):
                return cost
        sprayed = apply_chemical(self.state, 0.88)
        self.state = StageState(
            seed=self.state.seed,
            seedling=np.where(hot[np.newaxis, :, :], sprayed.seedling, self.state.seedling),
            rosette=np.where(hot, sprayed.rosette, self.state.rosette),
            adult=np.where(hot, sprayed.adult, self.state.adult),
        )
        self.n_sprays += 1
        return cost + 0.15

    def _bee_utility(self, forcing: DailyForcing) -> float:
        """早春开花日接近 1，其它花期打折，非花期为 0。"""
        flowers = float(np.sum(self.state.adult))
        if flowers <= 0.0 or not forcing.releasing:
            return 0.0
        month = month_of_doy(forcing.day)
        richness = float(np.clip(flowers / (GRID_N * GRID_N * 0.25), 0.0, 1.0))
        if month in SPRING_MONTHS:
            season = 1.00
        elif month in {5, 6, 9, 10}:
            season = 0.35
        else:
            season = 0.05
        return float(np.clip(season * richness, 0.0, 1.0))

    def _turf_damage(self) -> float:
        """占用率 ∈[0,1]。"""
        plants = self.state.seedling_total + self.state.rosette + self.state.adult
        return float(np.mean(plants >= OCCUPY))

    def _obs(self) -> NDArray[np.float32]:
        forcing = self.forcings[min(self.day_index, 364)]
        ncell = float(GRID_N * GRID_N)
        kcap = self.params.K_cell
        doy = float(forcing.day)
        season = 0.5 + 0.5 * np.sin(2.0 * np.pi * doy / 365.0)
        vec = np.array(
            [
                float(np.clip((forcing.temp_c + 10.0) / 50.0, 0.0, 1.0)),
                float(np.clip(forcing.moisture, 0.0, 1.0)),
                float(np.clip(np.sum(self.state.adult) / (ncell * kcap), 0.0, 1.0)),
                float(np.clip(np.sum(self.state.rosette) / (ncell * kcap), 0.0, 1.0)),
                float(np.clip(np.sum(self.state.seed) / (ncell * 180.0 * kcap), 0.0, 1.0)),
                self._bee_utility(forcing),
                float(np.clip(self.cum_cost / 80.0, 0.0, 1.0)),
                float(season),
            ],
            dtype=np.float32,
        )
        return vec

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """先作业，再产籽泄漏，再物候一日。"""
        if self.day_index >= 365:
            raise RuntimeError("episode already done; call reset")
        forcing = self.forcings[self.day_index]
        month = month_of_doy(forcing.day)
        act = int(action)
        cost = self._apply_action(act)
        self.cum_cost += cost
        rain = seed_release_field(self.state.adult, forcing.releasing, self.params)
        self.state.seed = self.state.seed + _absorbing_leak(rain)
        self.state = step_phenology(self.state, forcing.temp_c, forcing.moisture, self.params)
        bee = self._bee_utility(forcing)
        self.bee_sum += bee
        turf = self._turf_damage()
        reward = 2.0 * bee - 1.5 * turf - 0.5 * cost
        if month in SPRING_MONTHS and act in (2, 3):
            reward -= SPRING_PENALTY
        self.day_index += 1
        terminated = self.day_index >= 365
        info = {
            "day": self.day_index,
            "month": month,
            "bee": bee,
            "turf": turf,
            "cost": cost,
            "action": act,
            "n_mows": self.n_mows,
            "n_sprays": self.n_sprays,
            "bee_sum": self.bee_sum,
        }
        return self._obs(), float(reward), terminated, False, info

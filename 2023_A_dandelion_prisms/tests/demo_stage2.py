"""阶段 2 演示：90 日春季强迫下的 1 ha 卷积推演。

对接 ``DailyForcing`` / ``run_hectare_days``，不假设 ``DandelionGridEngine``。
风温为确定性斜坡（禁止 ``np.random`` 填表）。覆盖阈值与周步引擎一致：0.05 株/m²。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.biology.phenology import PhenologyParams, StageState
from src.simulation.grid_engine import DailyForcing, run_hectare_days


def clone_state(state: StageState) -> StageState:
    """深拷贝阶段场，避免后续 ``state.seed += rain`` 改写快照。"""
    return StageState(
        seed=state.seed.copy(),
        seedling=state.seedling.copy(),
        rosette=state.rosette.copy(),
        adult=state.adult.copy(),
    )

SNAPSHOT_DAYS: tuple[int, ...] = (1, 10, 30, 60, 90)
OCCUPY_THRESH: float = 0.05
CENTER_I: int = 50
CENTER_J: int = 50


def spring_forcing_90() -> list[DailyForcing]:
    """日均温 12→18 ℃、西风 3.5 m/s、VWC=0.35；花期全日释放。"""
    rows: list[DailyForcing] = []
    for i in range(90):
        day = i + 1
        temp_c = 12.0 + (day / 90.0) * 6.0
        rows.append(
            DailyForcing(
                day=day,
                temp_c=temp_c,
                wind_mps=3.5,
                wind_from_deg=270.0,
                moisture=0.35,
                releasing=True,
            )
        )
    return rows


def front_radius_m(adult: NDArray[np.float64], threshold: float = OCCUPY_THRESH) -> float:
    """相对 (50, 50) 的 95% 占用半径（m）；空场为 0。"""
    rows, cols = np.where(adult >= threshold)
    if rows.size == 0:
        return 0.0
    dist = np.hypot(rows.astype(np.float64) - CENTER_I, cols.astype(np.float64) - CENTER_J)
    return float(np.percentile(dist, 95))


def print_snapshot(day: int, state: StageState) -> None:
    """打印一日截面：成株合计、种子库、前锋半径、单格峰值。"""
    n_adult = float(np.sum(state.adult))
    n_seed = float(np.sum(state.seed))
    peak = float(np.max(state.adult))
    radius = front_radius_m(state.adult)
    print(
        f"Day {day:02d} ({day / 30.0:.1f} 个月)  | "
        f"{n_adult:<10.2f} | {n_seed:<12.1f} | {radius:<15.2f} | {peak:.2f} 株/m²"
    )


def main() -> None:
    """跑 90 步并在指定节点打印。"""
    params = PhenologyParams()
    snapshots: dict[int, StageState] = {}

    def on_day(day: int, state: StageState) -> None:
        if day in SNAPSHOT_DAYS:
            snapshots[day] = clone_state(state)

    print("=" * 70)
    print("[阶段 2 验证] 蒲公英生活史与 2D 卷积网格 90 天推演")
    print("=" * 70)
    print(f"{'时间节点':<15} | {'成株总数':<10} | {'地表种子数':<12} | {'扩散前锋半径 (m)':<15} | {'单格最高密度'}")
    print("-" * 70)

    t0 = time.perf_counter()
    result = run_hectare_days(spring_forcing_90(), params=params, on_day=on_day)
    wall = time.perf_counter() - t0

    for day in SNAPSHOT_DAYS:
        print_snapshot(day, snapshots[day])

    print("-" * 70)
    print(
        f"90 步仿真总耗时: {result.elapsed_s:.3f} s "
        f"(含核缓存；墙钟 {wall:.3f} s，平均单日 {result.elapsed_s / 90.0 * 1000.0:.2f} ms)"
    )
    print(f"90 日结束：max A = {float(np.max(result.state.adult)):.3f} ≤ K_cell = {params.K_cell}")
    print("=" * 70)


if __name__ == "__main__":
    main()

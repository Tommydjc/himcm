#!/usr/bin/env python3
"""由月气候契约展开逐小时风速与逐日 SMI。

运行::

    PYTHONPATH=. python -m src.scripts.generate_forcing
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from src.params import ClimateSpec, load_climates

DAYS_IN_MONTH: tuple[int, ...] = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
WEATHER_DIR: Path = PACK_ROOT / "data" / "raw" / "weather"
SOIL_DIR: Path = PACK_ROOT / "data" / "raw" / "soil"
YEAR: int = 2023


def _month_of(day_of_year: int) -> int:
    """1-index 年积日 → 月（非闰年）。"""
    acc = 0
    for month, ndays in enumerate(DAYS_IN_MONTH, start=1):
        acc += ndays
        if day_of_year <= acc:
            return month
    return 12


def expand_climate(spec: ClimateSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """确定性日变化：气温下午峰值，风速弱日循环，SMI 月内常数。"""
    by_month = {row.month: row for row in spec.months}
    weather_rows: list[dict[str, object]] = []
    soil_rows: list[dict[str, object]] = []
    cursor = date(YEAR, 1, 1)
    doy = 1
    while cursor.year == YEAR:
        month = _month_of(doy)
        forcing = by_month[month]
        soil_rows.append({"date": cursor.isoformat(), "smi": forcing.smi})
        for hour in range(24):
            stamp = datetime(YEAR, cursor.month, cursor.day, hour, 0, 0)
            temp = forcing.T_c + 4.0 * np.sin(2.0 * np.pi * (hour - 9) / 24.0)
            wind = forcing.wind_mps * (1.0 + 0.12 * np.sin(2.0 * np.pi * hour / 24.0))
            weather_rows.append(
                {
                    "timestamp": stamp.isoformat(sep=" "),
                    "wind_speed_mps": float(max(wind, 0.0)),
                    "wind_from_deg": float(forcing.wind_from_deg),
                    "air_temp_c": float(temp),
                }
            )
        cursor = cursor + timedelta(days=1)
        doy += 1
        if doy > 365:
            break
    weather = pd.DataFrame(weather_rows)
    soil = pd.DataFrame(soil_rows)
    return weather, soil


def write_all_climates(climates: dict[str, ClimateSpec] | None = None) -> list[Path]:
    """写出三套 weather/soil CSV，返回路径列表。"""
    specs = climates or load_climates()
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    SOIL_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, spec in specs.items():
        weather, soil = expand_climate(spec)
        w_path = WEATHER_DIR / f"{name}_hourly.csv"
        s_path = SOIL_DIR / f"{name}_smi.csv"
        weather.to_csv(w_path, index=False)
        soil.to_csv(s_path, index=False)
        written.extend([w_path, s_path])
    return written


def main() -> None:
    paths = write_all_climates()
    for path in paths:
        print(f"wrote {path.relative_to(PACK_ROOT)}")


if __name__ == "__main__":
    main()

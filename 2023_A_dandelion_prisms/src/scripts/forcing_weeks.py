"""把逐小时/逐日 CSV 聚成 52 周强迫。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.params import ClimateSpec
from src.simulation.engine import WeekForcing

PACK_ROOT: Path = Path(__file__).resolve().parents[2]


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """角度圆均值（度）。"""
    rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    mean = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    deg = float(np.rad2deg(mean)) % 360.0
    return deg


def week_to_month(week: int) -> int:
    """52 周均匀映射到 12 月（第 52 周归 12 月）。"""
    if week < 1:
        raise ValueError("week must be >= 1")
    return min(12, 1 + (week - 1) * 12 // 52)


def load_weekly_forcing(
    climate: str,
    spec: ClimateSpec,
    smi_scale: float = 1.0,
    mow_interval: int = 0,
    mow_eta: float = 0.0,
    weather_dir: Path | None = None,
    soil_dir: Path | None = None,
) -> list[WeekForcing]:
    """读情景 CSV，按 ISO 周近似：每 7 日一块，最后一块吃剩余日。

    Parameters
    ----------
    smi_scale :
        干旱扰动乘子，默认 1。
    mow_interval :
        割草间隔（周）；0 表示不割。
    mow_eta :
        割草强度。
    """
    wdir = weather_dir or (PACK_ROOT / "data" / "raw" / "weather")
    sdir = soil_dir or (PACK_ROOT / "data" / "raw" / "soil")
    weather = pd.read_csv(wdir / f"{climate}_hourly.csv", parse_dates=["timestamp"])
    soil = pd.read_csv(sdir / f"{climate}_smi.csv", parse_dates=["date"])
    weather["date"] = weather["timestamp"].dt.normalize()
    daily_wind = weather.groupby("date", as_index=False).agg(
        wind_speed_mps=("wind_speed_mps", "mean"),
        air_temp_c=("air_temp_c", "mean"),
    )
    daily_dir = weather.groupby("date")["wind_from_deg"].apply(lambda s: circular_mean_deg(s.to_numpy()))
    daily = daily_wind.merge(soil, left_on="date", right_on="date", how="left")
    daily["wind_from_deg"] = daily_dir.to_numpy()
    daily = daily.sort_values("date").reset_index(drop=True)
    n_days = len(daily)
    if n_days < 52:
        raise ValueError(f"{climate} soil/weather shorter than 52 days")
    vwc_weekly = _weekly_open_meteo_vwc(climate, sdir, n_days)
    bloom = set(spec.bloom_months)
    out: list[WeekForcing] = []
    for week in range(1, 53):
        start = (week - 1) * 7
        stop = n_days if week == 52 else min(week * 7, n_days)
        block = daily.iloc[start:stop]
        month = week_to_month(week)
        eta = 0.0
        if mow_interval > 0 and week % mow_interval == 0:
            eta = mow_eta
        vwc = None if vwc_weekly is None else vwc_weekly[week - 1]
        out.append(
            WeekForcing(
                week=week,
                month=month,
                temp_c=float(block["air_temp_c"].mean()),
                wind_mps=float(block["wind_speed_mps"].mean()),
                wind_from_deg=circular_mean_deg(block["wind_from_deg"].to_numpy()),
                smi=float(block["smi"].mean()) * smi_scale,
                blooming=month in bloom,
                mow_eta=eta,
                soil_moisture=vwc,
            )
        )
    return out


def _weekly_open_meteo_vwc(climate: str, soil_dir: Path, n_days: int) -> list[float] | None:
    """若存在 ``{climate}_soil.csv``，按与情景相同的 7 日切块取周均 VWC。"""
    path = soil_dir / f"{climate}_soil.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    if "soil_moisture" not in table.columns:
        return None
    values = pd.to_numeric(table["soil_moisture"], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{path.name} has null soil_moisture")
    series = values.to_numpy(dtype=np.float64)
    if series.size < 52:
        return None
    span = min(int(n_days), int(series.size))
    weekly: list[float] = []
    for week in range(1, 53):
        start = (week - 1) * 7
        stop = span if week == 52 else min(week * 7, span)
        weekly.append(float(np.mean(series[start:stop])))
    return weekly

#!/usr/bin/env python3
"""从 Open-Meteo Archive 拉取 2022 年三地逐日风、温、表层土壤湿度。

运行（题包根目录）::

    PYTHONPATH=. python scripts/download_real_climate.py

写出 ``data/raw/weather/{climate}_weather.csv`` 与
``data/raw/soil/{climate}_soil.csv``。不覆盖情景文件 ``*_hourly.csv`` / ``*_smi.csv``。
不编造缺测：任一日缺字段则失败退出。

数据源：https://archive-api.open-meteo.com/v1/archive （无需注册）。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PACK_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
START_DATE: str = "2022-01-01"
END_DATE: str = "2022-12-31"
EXPECTED_DAYS: int = 365
DAILY_VARS: tuple[str, ...] = (
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "temperature_2m_mean",
    "soil_moisture_0_to_7cm_mean",
)
WEATHER_COLUMNS: tuple[str, ...] = ("date", "wind_speed", "wind_direction", "temperature")
SOIL_COLUMNS: tuple[str, ...] = ("date", "soil_moisture")
MAX_RETRIES: int = 4
RETRY_SECONDS: float = 2.0


@dataclass(frozen=True, slots=True)
class Station:
    """一个气候代表站。

    Parameters
    ----------
    climate :
        题包气候键：``temperate`` / ``arid`` / ``tropical``。
    name :
        人类可读地名。
    latitude, longitude :
        WGS84 请求坐标（度）。
    timezone :
        IANA 时区，保证逐日聚合落在当地日历日。
    """

    climate: str
    name: str
    latitude: float
    longitude: float
    timezone: str


STATIONS: tuple[Station, ...] = (
    Station("temperate", "Princeton, NJ", 40.35, -74.66, "America/New_York"),
    Station("arid", "Phoenix, AZ", 33.45, -112.07, "America/Phoenix"),
    Station("tropical", "Miami, FL", 25.76, -80.19, "America/New_York"),
)


def build_archive_url(station: Station) -> str:
    """构造 Archive GET URL；风速单位强制为 m/s。"""
    query = urllib.parse.urlencode(
        {
            "latitude": f"{station.latitude:.4f}",
            "longitude": f"{station.longitude:.4f}",
            "start_date": START_DATE,
            "end_date": END_DATE,
            "daily": ",".join(DAILY_VARS),
            "wind_speed_unit": "ms",
            "timezone": station.timezone,
        }
    )
    return f"{ARCHIVE_URL}?{query}"


def fetch_archive_json(station: Station, timeout_s: float = 60.0) -> dict[str, Any]:
    """请求 Archive JSON，短暂重试网络错误。

    Returns
    -------
    dict
        Open-Meteo 成功对象，须含 ``daily``。

    Raises
    ------
    RuntimeError
        HTTP/解析失败或 API 返回 ``error``。
    """
    url = build_archive_url(station)
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "GoHiMCM-2023A/1.0"})
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("error"):
                raise RuntimeError(f"Open-Meteo error for {station.climate}: {payload.get('reason')}")
            if "daily" not in payload:
                raise RuntimeError(f"Open-Meteo response missing daily for {station.climate}")
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            time.sleep(RETRY_SECONDS * attempt)
    raise RuntimeError(f"failed to fetch {station.climate} after {MAX_RETRIES} tries: {last_error}")


def daily_to_frames(payload: dict[str, Any], station: Station) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把 ``daily`` 块拆成天气表与土壤表，并校验 365 行、无缺测。

    Parameters
    ----------
    payload :
        ``fetch_archive_json`` 的返回值。
    station :
        仅用于报错定位。

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        天气列 ``date, wind_speed, wind_direction, temperature``；
        土壤列 ``date, soil_moisture``。风速为 m/s，风向为气象来向度，
        温度为 ℃，土壤水分为 m³/m³。
    """
    daily = payload["daily"]
    missing = [key for key in ("time", *DAILY_VARS) if key not in daily]
    if missing:
        raise RuntimeError(f"{station.climate} daily missing keys: {missing}")
    weather = pd.DataFrame(
        {
            "date": daily["time"],
            "wind_speed": daily["wind_speed_10m_max"],
            "wind_direction": daily["wind_direction_10m_dominant"],
            "temperature": daily["temperature_2m_mean"],
        }
    )
    soil = pd.DataFrame(
        {
            "date": daily["time"],
            "soil_moisture": daily["soil_moisture_0_to_7cm_mean"],
        }
    )
    weather = weather.loc[:, list(WEATHER_COLUMNS)]
    soil = soil.loc[:, list(SOIL_COLUMNS)]
    if len(weather) != EXPECTED_DAYS or len(soil) != EXPECTED_DAYS:
        raise RuntimeError(
            f"{station.climate} expected {EXPECTED_DAYS} days, "
            f"got weather={len(weather)} soil={len(soil)}"
        )
    if weather[list(WEATHER_COLUMNS[1:])].isna().any().any():
        bad = weather.columns[weather.isna().any()].tolist()
        raise RuntimeError(f"{station.climate} weather has nulls in {bad}")
    if soil["soil_moisture"].isna().any():
        raise RuntimeError(f"{station.climate} soil_moisture has nulls")
    if weather["date"].iloc[0] != START_DATE or weather["date"].iloc[-1] != END_DATE:
        raise RuntimeError(
            f"{station.climate} date span {weather['date'].iloc[0]}–{weather['date'].iloc[-1]} "
            f"!= {START_DATE}–{END_DATE}"
        )
    return weather, soil


def write_station_csvs(
    station: Station,
    weather: pd.DataFrame,
    soil: pd.DataFrame,
    weather_dir: Path | None = None,
    soil_dir: Path | None = None,
) -> tuple[Path, Path]:
    """写出一对 CSV，返回路径。"""
    wdir = weather_dir or (PACK_ROOT / "data" / "raw" / "weather")
    sdir = soil_dir or (PACK_ROOT / "data" / "raw" / "soil")
    wdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    weather_path = wdir / f"{station.climate}_weather.csv"
    soil_path = sdir / f"{station.climate}_soil.csv"
    weather.to_csv(weather_path, index=False)
    soil.to_csv(soil_path, index=False)
    return weather_path, soil_path


def write_download_manifest(rows: list[dict[str, Any]], path: Path | None = None) -> Path:
    """记录请求坐标、API 回落格点与检索时刻，供论文引用。"""
    target = path or (PACK_ROOT / "data" / "raw" / "open_meteo_2022_manifest.json")
    target.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def download_all() -> list[Path]:
    """拉取三站并写盘，返回全部写出路径。"""
    written: list[Path] = []
    manifest: list[dict[str, Any]] = []
    retrieved = datetime.now(timezone.utc).isoformat()
    for station in STATIONS:
        payload = fetch_archive_json(station)
        weather, soil = daily_to_frames(payload, station)
        paths = write_station_csvs(station, weather, soil)
        written.extend(paths)
        manifest.append(
            {
                "climate": station.climate,
                "name": station.name,
                "requested_latitude": station.latitude,
                "requested_longitude": station.longitude,
                "api_latitude": payload.get("latitude"),
                "api_longitude": payload.get("longitude"),
                "elevation_m": payload.get("elevation"),
                "timezone": payload.get("timezone"),
                "url": build_archive_url(station),
                "retrieved_utc": retrieved,
                "n_days": int(len(weather)),
                "daily_units": payload.get("daily_units"),
            }
        )
    written.append(write_download_manifest(manifest))
    return written


def main() -> None:
    paths = download_all()
    for path in paths:
        print(f"wrote {path.relative_to(PACK_ROOT)}")


if __name__ == "__main__":
    main()

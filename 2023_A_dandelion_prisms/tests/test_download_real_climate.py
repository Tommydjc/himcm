"""Open-Meteo 日表拆分契约：365 行、列名、缺测即失败。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_real_climate import STATIONS, Station, daily_to_frames


def _payload(n_days: int = 365, start: str = "2022-01-01") -> dict:
    """构造合法 Archive daily 块（仅供单测，不是观测）。"""
    import pandas as pd

    dates = pd.date_range(start, periods=n_days, freq="D").strftime("%Y-%m-%d").tolist()
    ones = [1.0] * n_days
    return {
        "daily": {
            "time": dates,
            "wind_speed_10m_max": ones,
            "wind_direction_10m_dominant": [180.0] * n_days,
            "temperature_2m_mean": [10.0] * n_days,
            "soil_moisture_0_to_7cm_mean": [0.2] * n_days,
        }
    }


class TestDailyToFrames(unittest.TestCase):
    """拆表校验。"""

    station: Station = STATIONS[0]

    def test_365_days_and_columns(self) -> None:
        weather, soil = daily_to_frames(_payload(365), self.station)
        self.assertEqual(len(weather), 365)
        self.assertEqual(len(soil), 365)
        self.assertEqual(list(weather.columns), ["date", "wind_speed", "wind_direction", "temperature"])
        self.assertEqual(list(soil.columns), ["date", "soil_moisture"])
        self.assertEqual(weather["date"].iloc[0], "2022-01-01")
        self.assertEqual(weather["date"].iloc[-1], "2022-12-31")

    def test_null_wind_raises(self) -> None:
        payload = _payload(365)
        payload["daily"]["wind_speed_10m_max"][0] = None
        with self.assertRaises(RuntimeError):
            daily_to_frames(payload, self.station)

    def test_wrong_length_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            daily_to_frames(_payload(10), self.station)


if __name__ == "__main__":
    unittest.main()

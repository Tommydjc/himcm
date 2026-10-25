"""Phase-3 Bang-Bang summer-day tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / "src"))

from baseline_thermostat.bang_bang import (
    MINUTES_PER_DAY,
    BangBangController,
    count_short_cycles,
    run_bang_bang_day,
    synthesize_summer_day_1min,
    write_weather_csv,
)


def test_weather_shape_and_peak() -> None:
    df = synthesize_summer_day_1min()
    assert len(df) == MINUTES_PER_DAY
    assert float(df["T_out_C"].max()) == pytest_approx_max_near_38(df)
    # Peak near 15:00
    idx = int(df["T_out_C"].idxmax())
    assert 14.0 <= float(df.loc[idx, "hour_frac"]) <= 16.0
    assert float(df["T_out_C"].min()) < 24.0


def pytest_approx_max_near_38(df: pd.DataFrame) -> float:
    mx = float(df["T_out_C"].max())
    assert 37.5 <= mx <= 38.5
    return mx


def test_hysteresis_logic() -> None:
    c = BangBangController()
    q, on = c.step(24.5)
    assert on and q < 0.0
    q2, on2 = c.step(23.0)  # inside band — hold
    assert on2 and q2 < 0.0
    q3, on3 = c.step(21.5)
    assert (not on3) and q3 == 0.0


def test_short_cycle_counter() -> None:
    flags = np.array([False, True, True, False, True], dtype=bool)
    assert count_short_cycles(flags) == 2


def test_run_day_export_length() -> None:
    write_weather_csv()
    out = run_bang_bang_day()
    assert len(out["trajectory"]) == MINUTES_PER_DAY
    m = out["metrics"]
    assert m["energy_kwh"] >= 0.0
    assert m["cost_usd"] >= 0.0
    assert 0.0 <= m["peak_cost_share"] <= 1.0
    assert m["short_cycle_starts"] >= 1

"""Tests for CapEx–wait Pareto extraction helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization.pareto_sizing import (
    extract_pareto_front_2d,
    select_knee_near_budget,
)


def test_pareto_2d_basic() -> None:
    rows = [
        {"c_fast": 1, "total_cost": 10_000, "mean_wait_time": 10.0},
        {"c_fast": 2, "total_cost": 20_000, "mean_wait_time": 2.0},
        {"c_fast": 3, "total_cost": 15_000, "mean_wait_time": 12.0},
    ]
    front = extract_pareto_front_2d(rows)
    costs = {r["total_cost"] for r in front}
    assert 10_000 in costs
    assert 20_000 in costs
    assert 15_000 not in costs


def test_knee_prefers_under_budget() -> None:
    pareto = [
        {
            "c_fast": 1,
            "c_slow": 0,
            "c_bank": 1,
            "total_cost": 40_000,
            "mean_wait_time": 4.0,
            "reneging_rate": 0.02,
        },
        {
            "c_fast": 2,
            "c_slow": 0,
            "c_bank": 1,
            "total_cost": 70_000,
            "mean_wait_time": 1.0,
            "reneging_rate": 0.01,
        },
        {
            "c_fast": 3,
            "c_slow": 0,
            "c_bank": 1,
            "total_cost": 90_000,
            "mean_wait_time": 0.5,
            "reneging_rate": 0.01,
        },
    ]
    knee = select_knee_near_budget(pareto, 80_000)
    assert knee["total_cost"] == 70_000


def test_knee_skips_high_reneging_cheap_pack() -> None:
    pareto = [
        {
            "c_fast": 4,
            "c_slow": 0,
            "c_bank": 0,
            "total_cost": 8_000,
            "mean_wait_time": 0.0,
            "reneging_rate": 0.60,
        },
        {
            "c_fast": 12,
            "c_slow": 8,
            "c_bank": 2,
            "total_cost": 47_800,
            "mean_wait_time": 1.2,
            "reneging_rate": 0.02,
        },
    ]
    knee = select_knee_near_budget(pareto, 80_000, max_reneging=0.05)
    assert knee["total_cost"] == 47_800

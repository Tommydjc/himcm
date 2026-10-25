"""Phase-2 PMV discomfort + TOU v2 tests."""

from __future__ import annotations

import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / "src"))

from comfort_tariffs.models import (
    PMV_COMFORT_BAND,
    TOUTariffSchedule,
    calculate_pmv_ppd,
    discomfort_cost,
    discomfort_cost_from_air,
)


def test_pmv_near_neutral_at_24c() -> None:
    pmv, ppd = calculate_pmv_ppd(24.0, rh=50.0, met=1.2, clo=0.5)
    assert abs(pmv) < 0.35
    assert 5.0 <= ppd <= 25.0
    assert discomfort_cost(pmv) == 0.0 or abs(pmv) <= PMV_COMFORT_BAND


def test_discomfort_zero_inside_band() -> None:
    assert discomfort_cost(0.0) == 0.0
    assert discomfort_cost(0.5) == 0.0
    assert discomfort_cost(-0.5) == 0.0
    assert discomfort_cost(1.0) == (0.5) ** 2


def test_hot_room_positive_penalty() -> None:
    out = discomfort_cost_from_air(30.0, rh=50.0)
    assert out["pmv"] > 0.5
    assert out["discomfort_cost"] > 0.0


def test_tou_v2_rates() -> None:
    tou = TOUTariffSchedule.from_json()
    assert tou.rate_usd_per_kwh(3) == 0.12
    assert tou.rate_usd_per_kwh(23) == 0.12
    assert tou.rate_usd_per_kwh(10) == 0.22
    assert tou.rate_usd_per_kwh(20) == 0.22
    assert tou.rate_usd_per_kwh(15) == 0.48
    assert tou.period_name(15) == "on_peak"
    assert tou.is_on_peak(16)
    assert not tou.is_on_peak(10)


def test_tou_peak_to_off_ratio_is_four() -> None:
    tou = TOUTariffSchedule.from_json()
    assert abs(tou.rate_usd_per_kwh(15) / tou.rate_usd_per_kwh(2) - 4.0) < 1e-12


def test_tou_bill_one_peak_kwh() -> None:
    tou = TOUTariffSchedule.from_json()
    bill = tou.bill_from_power_W([1000.0], [15], dt_hours=1.0)
    assert abs(bill["cost_usd"] - 0.48) < 1e-12
    assert abs(bill["peak_cost_share"] - 1.0) < 1e-12


def test_all_hours_covered() -> None:
    tou = TOUTariffSchedule.from_json()
    table = tou.hourly_rate_table()
    assert set(table.keys()) == set(range(24))
    assert set(table.values()) == {0.12, 0.22, 0.48}

"""SimPy airport DES sandbox tests (physical conservation + KPI bounds)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from des_env.airport_sim import (
    HORIZON_MIN,
    arrival_intensity,
    run_airport_simulation,
)


def test_peak_intensity_is_triple_base() -> None:
    lam0 = 0.4
    for peak in (120.0, 420.0, 720.0):
        assert arrival_intensity(peak, lam0) == pytest.approx(3.0 * lam0, rel=1e-9)
        assert arrival_intensity(
            peak, lam0, peak_multiplier=5.0
        ) == pytest.approx(5.0 * lam0, rel=1e-9)
    # Mid-valley between 08:00 and 13:00 should be near base.
    mid = 270.0
    assert arrival_intensity(mid, lam0) < 1.2 * lam0


def test_timeline_has_system_load() -> None:
    m = run_airport_simulation(4, 8, 2, seed=3)
    tl = m["queue_timeline"]
    assert len(tl) > 100
    assert "system_load" in tl[0]
    assert "lambda_t" in tl[0]
    assert "in_service" in tl[0]
    # Over a full day with seekers, peak system load should exceed waiting-only max
    # or at least be non-negative and finite.
    assert m["max_system_load"] >= m["max_queue_len"]
    loads = [row["system_load"] for row in tl]
    assert max(loads) == m["max_system_load"]


def test_run_metrics_bounds() -> None:
    m = run_airport_simulation(4, 8, 2, pricing_mode=1.0, seed=7)
    assert m["n_arrivals"] > 50
    assert m["max_queue_len"] >= 0
    assert 0.0 <= m["reneging_rate"] <= 1.0
    assert 0.0 <= m["utilization_fast"] <= 1.0
    assert 0.0 <= m["utilization_slow"] <= 1.0
    assert m["total_powerbank_rentals"] >= 0
    assert m["mean_wait_time"] >= 0.0
    assert m["horizon_min"] == HORIZON_MIN


def test_scarce_chargers_raise_reneging() -> None:
    tight = run_airport_simulation(1, 1, 1, pricing_mode=1.0, seed=11, lambda_base=0.5)
    loose = run_airport_simulation(10, 16, 4, pricing_mode=1.0, seed=11, lambda_base=0.5)
    assert tight["n_seekers"] > 0
    assert tight["reneging_rate"] > 0.0
    assert tight["reneging_rate"] >= loose["reneging_rate"]
    assert tight["max_queue_len"] >= loose["max_queue_len"]
    assert tight["total_powerbank_rentals"] >= loose["total_powerbank_rentals"]


def test_clock_monotone_no_negative_queue() -> None:
    # Smoke: multiple seeds must not raise DES invariant errors.
    for seed in (0, 1, 2):
        m = run_airport_simulation(3, 6, 2, seed=seed, lambda_base=0.25)
        assert m["max_queue_len"] >= 0
        assert np.isfinite(m["mean_wait_time"])

"""Unit tests for BDI passenger intention transitions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bdi_agents.passenger_bdi import (
    BATTERY_ANXIETY_THRESHOLD,
    BDIPassenger,
    IntentionState,
    PassengerBeliefs,
    load_or_sample_passenger_features,
    sample_battery_levels,
    sample_boarding_countdowns,
)


def test_anxiety_mass_near_25_percent() -> None:
    levels = sample_battery_levels(20_000, rng=np.random.default_rng(0))
    share = float(np.mean(levels < BATTERY_ANXIETY_THRESHOLD))
    assert 0.24 <= share <= 0.26


def test_boarding_uniform_bounds() -> None:
    t = sample_boarding_countdowns(5_000, rng=np.random.default_rng(1))
    assert float(t.min()) >= 45.0
    assert float(t.max()) <= 120.0


def test_idle_to_seek_when_anxious() -> None:
    agent = BDIPassenger(
        passenger_id=0,
        beliefs=PassengerBeliefs(10.0, 90.0, 1.0),
    )
    assert agent.evaluate_intentions(0, 20.0) == IntentionState.SEEK_POWER


def test_queue_to_reneging_when_tight_boarding() -> None:
    agent = BDIPassenger(
        passenger_id=1,
        beliefs=PassengerBeliefs(8.0, 10.0, 1.2),
        intention=IntentionState.QUEUE_AT_KIOSK,
    )
    final = agent.evaluate_intentions(observed_queue_len=6, powerbank_price=10.0)
    assert agent.did_renege()
    assert final in (
        IntentionState.RENT_POWERBANK,
        IntentionState.ABORT_CHARGING,
    )


def test_hundred_percent_reneging_batch() -> None:
    n = 100
    for i in range(n):
        agent = BDIPassenger(
            passenger_id=i,
            beliefs=PassengerBeliefs(
                battery_level=5.0 + 0.01 * i,
                time_to_boarding=10.0,
                price_sensitivity=0.5 + (i % 15) * 0.1,
            ),
            intention=IntentionState.QUEUE_AT_KIOSK,
        )
        agent.evaluate_intentions(6, powerbank_price=30.0)
        assert agent.did_renege()


def test_feature_loader_samples_without_csv() -> None:
    frame = load_or_sample_passenger_features(32, seed=3)
    assert list(frame.columns) == [
        "battery_level",
        "time_to_boarding",
        "price_sensitivity",
    ]
    assert len(frame) == 32


def test_negative_queue_rejected() -> None:
    agent = BDIPassenger(
        passenger_id=2,
        beliefs=PassengerBeliefs(10.0, 60.0, 1.0),
        intention=IntentionState.SEEK_POWER,
    )
    with pytest.raises(ValueError, match="non-negative"):
        agent.evaluate_intentions(-1, 20.0)

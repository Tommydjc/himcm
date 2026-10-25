"""Time-of-use electricity tariff lookup and bill accumulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

PACK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOU = PACK_ROOT / "data" / "raw" / "tou_tariffs.json"


def load_tou_tariffs(path: Path = DEFAULT_TOU) -> Dict[str, Any]:
    """Load peak / mid / off-peak rate dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def rate_usd_per_kwh(
    hour_of_day: int,
    tariffs: Mapping[str, Any] | None = None,
    *,
    surge_multiplier: float = 1.0,
) -> float:
    """
    Return TOU rate for civil hour in [0, 23].

    Raises
    ------
    ValueError
        If hour is missing from all periods.
    """
    tariffs = tariffs or load_tou_tariffs()
    hod = int(hour_of_day) % 24
    for period in tariffs["periods"]:
        if hod in period["hours"]:
            return float(period["rate"]) * float(surge_multiplier)
    raise ValueError(f"hour {hod} not covered by TOU periods")


def bill_from_power_W(
    power_W: Sequence[float],
    hour_of_day: Sequence[int],
    *,
    tariffs: Mapping[str, Any] | None = None,
    surge_multiplier: float = 1.0,
    dt_hours: float = 1.0,
) -> Dict[str, float]:
    """
    Accumulate energy [kWh] and cost [USD] from hourly site power.

    Parameters
    ----------
    power_W :
        Average electric power over each interval [W].
    hour_of_day :
        Civil hour labels aligned with ``power_W``.
    """
    tariffs = tariffs or load_tou_tariffs()
    p = np.asarray(power_W, dtype=float)
    h = np.asarray(hour_of_day, dtype=int)
    if p.shape != h.shape:
        raise ValueError("power_W and hour_of_day shape mismatch")
    energy_kwh = p * dt_hours / 1000.0
    rates = np.asarray(
        [rate_usd_per_kwh(int(hh), tariffs, surge_multiplier=surge_multiplier) for hh in h],
        dtype=float,
    )
    cost = energy_kwh * rates
    return {
        "energy_kwh": float(np.sum(energy_kwh)),
        "cost_usd": float(np.sum(cost)),
        "peak_energy_kwh": float(
            np.sum(energy_kwh[[rate_usd_per_kwh(int(hh), tariffs) >= 0.30 for hh in h]])
        ),
    }

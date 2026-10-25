"""Tests for teaching-track black-box evaluation harness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eval_harness.benchmark_harness import (
    SCENARIO_BUILDERS,
    attach_M4_robustness,
    discomfort_degree_hours,
    run_controller_on_scenario,
    scenario_A_standard,
)


def test_discomfort_integral() -> None:
    """M2 integrates only positive excess over T_comf."""
    T = np.array([23.0, 24.0, 25.0, 22.0])
    # excess: 0,1,2,0 → sum 3 * (1/60) °C·h
    assert abs(discomfort_degree_hours(T) - 3.0 / 60.0) < 1e-12


def test_scenario_builders_shapes() -> None:
    """A/C/D/E are 1440 min; B is 3×1440."""
    a = scenario_A_standard()
    assert len(a) == 1440
    b = SCENARIO_BUILDERS["B"]()
    assert len(b) == 3 * 1440
    c = SCENARIO_BUILDERS["C"]()
    # After 14:00, outdoor should be ~15°C colder than unmodified peak shape.
    assert float(c.loc[c["hour_frac"] >= 14.0, "T_out_C"].mean()) < 30.0


def test_bang_scenario_A_metrics() -> None:
    """Black-box Bang on Scenario A returns finite M1–M3."""
    weather = scenario_A_standard()
    out = run_controller_on_scenario("bang_bang", "A", weather)
    m = out["metrics"]
    assert m["M1_cost_usd"] > 0.0
    assert m["M2_discomfort_Ch"] >= 0.0
    assert m["M3_mechanical"] >= 1.0
    assert len(out["trajectory"]) == 1440


def test_m4_variance_attached() -> None:
    """M4 is constant within a controller block."""
    rows = []
    for sid, cost in [("A", 1.0), ("B", 1.2), ("C", 0.9)]:
        rows.append(
            {
                "controller": "bang_bang",
                "scenario": sid,
                "M1_cost_usd": cost,
                "M2_discomfort_Ch": 0.1,
                "M3_mechanical": 10.0,
            }
        )
    df = attach_M4_robustness(pd.DataFrame(rows))
    assert df["M4_cost_variance"].nunique() == 1
    assert float(df["M4_cost_variance"].iloc[0]) > 0.0

"""FastMCP contract tests for airport charging DES tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_server.server import (
    AREA_CAP_M2,
    BUDGET_CAP_USD,
    TARGET_WAIT_MIN,
    get_airport_constraints,
    get_in_process_tools,
    mcp,
    run_charging_simulation,
    validate_infrastructure_budget,
)


def test_budget_valid_feasible_pack() -> None:
    # 12/14/2 → area 38.6 m² ≤ 40; CapEx $47,800 ≤ $80k.
    out = validate_infrastructure_budget(12, 14, 2)
    assert out["is_valid"] is True
    assert out["total_cost"] == 12 * 2000 + 14 * 1200 + 2 * 3500
    assert out["total_area"] == pytest.approx(12 * 0.8 + 14 * 2.0 + 2 * 0.5)
    assert out["violation_reason"] == ""


def test_budget_stage2_baseline_fails_area() -> None:
    # Stage-2 default 12/15/2 is 40.6 m² → area violation (caught by MCP).
    out = validate_infrastructure_budget(12, 15, 2)
    assert out["is_valid"] is False
    assert out["total_area"] == pytest.approx(40.6)
    assert "floor area" in out["violation_reason"]


def test_budget_rejects_over_capex_or_area() -> None:
    out = validate_infrastructure_budget(40, 30, 8)
    assert out["is_valid"] is False
    assert out["total_cost"] > BUDGET_CAP_USD
    assert "CapEx" in out["violation_reason"] or "floor area" in out["violation_reason"]


def test_constraints_resource_json() -> None:
    payload = json.loads(get_airport_constraints())
    assert payload["budget_cap_usd"] == BUDGET_CAP_USD
    assert payload["area_cap_m2"] == AREA_CAP_M2
    assert payload["target_mean_wait_min"] == TARGET_WAIT_MIN


def test_run_charging_simulation_strips_timeline() -> None:
    metrics = run_charging_simulation(3, 5, 1, pricing=1.0)
    assert "queue_timeline" not in metrics
    assert metrics["n_arrivals"] > 0
    assert 0.0 <= metrics["reneging_rate"] <= 1.0


def test_in_process_tools_dict() -> None:
    tools = get_in_process_tools()
    assert set(tools) >= {
        "run_charging_simulation",
        "validate_infrastructure_budget",
        "get_airport_constraints",
    }
    assert tools["validate_infrastructure_budget"](1, 1, 1)["is_valid"] is True


def test_fastmcp_server_name() -> None:
    assert mcp.name == "Airport-Charging-DES-Server"

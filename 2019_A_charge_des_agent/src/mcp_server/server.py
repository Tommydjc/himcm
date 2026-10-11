"""
FastMCP server wrapping the SimPy airport charging DES.

Protocol layer (Tools / Resources) stays separate from the physical DES in
``des_env.airport_sim`` and the BDI cognitive layer in ``bdi_agents``.

In-process agents should call ``get_in_process_tools()`` instead of spawning
a second MCP stdio process on macOS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Final

from fastmcp import FastMCP

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from des_env.airport_sim import run_airport_simulation

# ---------------------------------------------------------------------------
# Airport authority hard constraints (also exposed as an MCP Resource)
# ---------------------------------------------------------------------------

COST_FAST_USD: Final[float] = 2_000.0
COST_SLOW_USD: Final[float] = 1_200.0
COST_BANK_USD: Final[float] = 3_500.0
AREA_FAST_M2: Final[float] = 0.8
AREA_SLOW_M2: Final[float] = 2.0
AREA_BANK_M2: Final[float] = 0.5
BUDGET_CAP_USD: Final[float] = 80_000.0
AREA_CAP_M2: Final[float] = 40.0
TARGET_WAIT_MIN: Final[float] = 5.0

mcp = FastMCP("Airport-Charging-DES-Server")


def _strip_timeline(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return scalar KPIs only (drop minute-level timeline for MCP payloads)."""
    out = dict(metrics)
    out.pop("queue_timeline", None)
    return out


@mcp.tool()
def run_charging_simulation(
    c_fast: int,
    c_slow: int,
    c_bank: int,
    pricing: float = 1.0,
) -> dict:
    """
    Run one 16-hour airport charging DES and return scalar telemetry.

    Parameters
    ----------
    c_fast :
        Number of 65W fast chargers (``C_fast``).
    c_slow :
        Number of 15W slow lockers (``C_slow``).
    c_bank :
        Number of power-bank kiosk groups (``C_bank``).
    pricing :
        Power-bank price multiplier (maps to ``pricing_mode``).

    Returns
    -------
    dict
        KPI dictionary from ``run_airport_simulation`` without ``queue_timeline``.
    """
    if min(c_fast, c_slow, c_bank) < 0:
        raise ValueError("capacities must be non-negative integers")
    if pricing < 0.0:
        raise ValueError(f"pricing must be non-negative: {pricing}")

    metrics = run_airport_simulation(
        c_fast=int(c_fast),
        c_slow=int(c_slow),
        c_bank=int(c_bank),
        pricing_mode=float(pricing),
    )
    return _strip_timeline(metrics)


@mcp.tool()
def validate_infrastructure_budget(
    c_fast: int,
    c_slow: int,
    c_bank: int,
    budget_cap_usd: float = BUDGET_CAP_USD,
    area_cap_m2: float = AREA_CAP_M2,
) -> dict:
    """
    Hard-constraint check for CapEx and floor area.

    Costs: fast $2,000 / slow $1,200 / bank group $3,500.
    Areas: fast 0.8 m² / slow 2.0 m² / bank 0.5 m².
    Default caps: budget ≤ $80,000, area ≤ 40 m² (overridable).

    Returns
    -------
    dict
        ``is_valid``, ``total_cost``, ``total_area``, ``violation_reason``.
    """
    if min(c_fast, c_slow, c_bank) < 0:
        return {
            "is_valid": False,
            "total_cost": 0.0,
            "total_area": 0.0,
            "violation_reason": "capacities must be non-negative",
        }

    cap_budget = float(budget_cap_usd)
    cap_area = float(area_cap_m2)
    total_cost = (
        float(c_fast) * COST_FAST_USD
        + float(c_slow) * COST_SLOW_USD
        + float(c_bank) * COST_BANK_USD
    )
    total_area = (
        float(c_fast) * AREA_FAST_M2
        + float(c_slow) * AREA_SLOW_M2
        + float(c_bank) * AREA_BANK_M2
    )

    reasons: list[str] = []
    if total_cost > cap_budget + 1e-9:
        reasons.append(
            f"CapEx ${total_cost:,.0f} exceeds budget ${cap_budget:,.0f}"
        )
    if total_area > cap_area + 1e-9:
        reasons.append(
            f"floor area {total_area:.1f} m² exceeds cap {cap_area:.1f} m²"
        )

    is_valid = len(reasons) == 0
    return {
        "is_valid": bool(is_valid),
        "total_cost": float(total_cost),
        "total_area": float(total_area),
        "violation_reason": "" if is_valid else "; ".join(reasons),
        "budget_cap_usd": float(cap_budget),
        "area_cap_m2": float(cap_area),
        "cost_breakdown_usd": {
            "fast": float(c_fast) * COST_FAST_USD,
            "slow": float(c_slow) * COST_SLOW_USD,
            "bank": float(c_bank) * COST_BANK_USD,
        },
        "area_breakdown_m2": {
            "fast": float(c_fast) * AREA_FAST_M2,
            "slow": float(c_slow) * AREA_SLOW_M2,
            "bank": float(c_bank) * AREA_BANK_M2,
        },
    }


def get_airport_constraints() -> str:
    """
    Airport authority hard targets as a JSON document.

    Returns
    -------
    str
        JSON string for MCP Resource ``config://airport_constraints``.
    """
    payload = {
        "budget_cap_usd": BUDGET_CAP_USD,
        "area_cap_m2": AREA_CAP_M2,
        "target_mean_wait_min": TARGET_WAIT_MIN,
        "unit_cost_usd": {
            "fast_charger": COST_FAST_USD,
            "slow_locker": COST_SLOW_USD,
            "powerbank_kiosk_group": COST_BANK_USD,
        },
        "unit_area_m2": {
            "fast_charger": AREA_FAST_M2,
            "slow_locker": AREA_SLOW_M2,
            "powerbank_kiosk_group": AREA_BANK_M2,
        },
        "notes": (
            "Valid sizing must satisfy CapEx and area caps. "
            "Service quality target: mean wait < 5.0 minutes."
        ),
    }
    return json.dumps(payload, indent=2)


@mcp.resource("config://airport_constraints")
def airport_constraints_resource() -> str:
    """MCP Resource: airport authority budget / area / wait targets."""
    return get_airport_constraints()


def get_in_process_tools() -> Dict[str, Callable[..., Any]]:
    """
    Native callables for local agents (no MCP stdio / subprocess).

    Returns
    -------
    dict
        Mapping tool name → Python function. Keys match MCP tool names.
    """
    return {
        "run_charging_simulation": run_charging_simulation,
        "validate_infrastructure_budget": validate_infrastructure_budget,
        "get_airport_constraints": get_airport_constraints,
    }


def _run_smoke_tests() -> None:
    """Local self-check (stdout OK here; never use before MCP handshake)."""
    tools = get_in_process_tools()
    # 12/15/2 uses 40.6 m² (>40); use a feasible pack for the smoke test.
    budget = tools["validate_infrastructure_budget"](12, 14, 2)
    assert budget["is_valid"] is True, budget
    over = tools["validate_infrastructure_budget"](40, 30, 8)
    assert over["is_valid"] is False, over
    edge = tools["validate_infrastructure_budget"](12, 15, 2)
    assert edge["is_valid"] is False
    assert "floor area" in edge["violation_reason"]
    constraints = json.loads(tools["get_airport_constraints"]())
    assert constraints["budget_cap_usd"] == BUDGET_CAP_USD

    slim = tools["run_charging_simulation"](4, 6, 2, 1.0)
    assert "queue_timeline" not in slim
    assert slim["n_arrivals"] > 0
    print("[OK] in-process MCP tool smoke test passed")
    print(
        f"  budget valid={budget['is_valid']} "
        f"cost=${budget['total_cost']:,.0f} area={budget['total_area']:.1f}m²"
    )
    print(
        f"  sim reneging_rate={slim['reneging_rate']:.3f} "
        f"mean_wait={slim['mean_wait_time']:.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Airport Charging DES MCP server")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in-process tool smoke tests (do not use for Cursor MCP)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start FastMCP stdio server (prefer: python -m mcp_server)",
    )
    args = parser.parse_args()
    if args.smoke:
        _run_smoke_tests()
    if args.serve:
        mcp.run()
    if not args.smoke and not args.serve:
        parser.print_help()

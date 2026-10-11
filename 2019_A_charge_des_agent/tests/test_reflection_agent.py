"""Tests for LangGraph reflection agent (rule fallback path)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from des_agent.reflection_agent import (
    BASELINE_PARAMS,
    _force_feasible,
    _rule_propose,
    build_reflection_graph,
    run_reflection_agent,
)
from mcp_server.server import validate_infrastructure_budget


def test_force_feasible_repairs_baseline_area() -> None:
    # 12/15/2 is over area; repair must become valid.
    fixed = _force_feasible(dict(BASELINE_PARAMS))
    info = validate_infrastructure_budget(
        fixed["c_fast"], fixed["c_slow"], fixed["c_bank"]
    )
    assert info["is_valid"] is True


def test_rule_propose_from_invalid_budget() -> None:
    state = {
        "current_params": dict(BASELINE_PARAMS),
        "simulation_results": {},
        "budget_valid": False,
        "budget_info": {
            "violation_reason": "floor area 40.6 m² exceeds cap 40.0 m²"
        },
        "reflection_trace": [],
        "iteration": 0,
        "converged": False,
        "error_message": None,
        "budget_reject_count": 1,
        "llm_mode": "rules",
    }
    nxt = _rule_propose(state)  # type: ignore[arg-type]
    info = validate_infrastructure_budget(nxt["c_fast"], nxt["c_slow"], nxt["c_bank"])
    assert info["is_valid"] is True


def test_graph_compiles() -> None:
    app = build_reflection_graph()
    assert app is not None


def test_run_reflection_agent_exports_log(tmp_path: Path) -> None:
    out = tmp_path / "des_agent_reflection_log.json"
    final = run_reflection_agent(export_path=out)
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "reflection_trace" in payload
    assert payload["iteration"] >= 1
    assert final["converged"] is True
    assert "c_fast" in final["current_params"]

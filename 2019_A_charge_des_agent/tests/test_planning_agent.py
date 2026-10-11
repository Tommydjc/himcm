"""Tests for autonomous experiment planner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from des_agent.planning_agent import (
    _candidate_from_reflection,
    _default_subtasks,
    extract_pareto_front,
    plan_subtasks,
    run_planning_agent,
)


def test_default_subtasks_cover_four_themes() -> None:
    subs = _default_subtasks()
    assert len(subs) == 4
    budgets = {s["budget_cap_usd"] for s in subs}
    assert 40_000.0 in budgets
    assert 80_000.0 in budgets
    prices = {s["pricing_mode"] for s in subs}
    assert min(prices) < 1.0
    assert max(prices) > 1.0


def test_plan_subtasks_fallback_rules() -> None:
    subs, source = plan_subtasks()
    assert source in {"rules", "ollama"}
    assert len(subs) >= 3


def test_pareto_extraction() -> None:
    cands = [
        {
            "subtask_id": "a",
            "total_cost": 100.0,
            "mean_wait_time": 10.0,
            "reneging_rate": 0.2,
        },
        {
            "subtask_id": "b",
            "total_cost": 200.0,
            "mean_wait_time": 1.0,
            "reneging_rate": 0.0,
        },
        {
            "subtask_id": "c",
            "total_cost": 150.0,
            "mean_wait_time": 12.0,
            "reneging_rate": 0.3,
        },
    ]
    front = extract_pareto_front(cands)
    ids = {x["subtask_id"] for x in front}
    assert "a" in ids
    assert "b" in ids
    assert "c" not in ids


def test_run_planning_agent_exports(tmp_path: Path) -> None:
    out = tmp_path / "autonomous_planning_summary.json"
    # Smoke with fewer cells via monkeypatching max_search_subtasks=2
    state = run_planning_agent(export_path=out, max_search_subtasks=2)
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "pareto_candidates" in payload
    assert "subtask_results" in payload
    assert len(payload["subtask_results"]) == 2
    assert state["plan_source"] in {"rules", "ollama"}

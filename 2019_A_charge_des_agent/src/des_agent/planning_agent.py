"""
Autonomous experiment planner (dual-LLM + reflection-agent scheduler).

Planner decomposes the HiMCM charging goal into four subtasks, then
sequentially schedules ``run_reflection_agent`` local searches. A second
LLM (or rule fallback) aggregates Pareto candidates.

Run (pack root):
  python -m src.des_agent.planning_agent
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PACK_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PACK_ROOT / "src"
for _p in (str(SRC_ROOT), str(PACK_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from des_agent.reflection_agent import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    _extract_json_object,
    _get_chat_llm,
    _ollama_available,
    run_reflection_agent,
)
from mcp_server.server import AREA_CAP_M2, validate_infrastructure_budget

console = Console()

SUMMARY_PATH = PACK_ROOT / "results" / "autonomous_planning_summary.json"
OVERALL_GOAL = (
    "Design a mobile charging infrastructure mix for an airport terminal "
    "under CapEx / floor-area caps, minimizing wait and reneging while "
    "serving battery-anxious passengers across a 16-hour DES day."
)


class SubtaskSpec(TypedDict):
    """One planned experiment cell."""

    id: str
    title: str
    description: str
    budget_cap_usd: float
    pricing_mode: float
    pricing_label: str
    baseline_params: Dict[str, int]
    max_iterations: int


class PlanningState(TypedDict):
    """Macro planner bookkeeping (not a LangGraph graph — sequential driver)."""

    overall_goal: str
    plan_source: str
    subtasks: List[SubtaskSpec]
    subtask_results: List[Dict[str, Any]]
    pareto_candidates: List[Dict[str, Any]]
    aggregator_notes: Dict[str, Any]
    llm_mode: str


def _default_subtasks() -> List[SubtaskSpec]:
    """Canonical four-cell experiment plan (rule fallback / schema anchor)."""
    return [
        {
            "id": "S1_low_budget",
            "title": "Extreme low-budget emergency system",
            "description": (
                "Find a minimal feasible pack under CapEx <= $40,000 "
                "that still provides emergency charging coverage."
            ),
            "budget_cap_usd": 40_000.0,
            "pricing_mode": 1.0,
            "pricing_label": "standard_pay_per_rental",
            "baseline_params": {"c_fast": 8, "c_slow": 6, "c_bank": 1},
            "max_iterations": 4,
        },
        {
            "id": "S2_standard_budget",
            "title": "Standard-budget fast/powerbank gold ratio",
            "description": (
                "Under CapEx <= $80,000, search the best mix of fast chargers "
                "and powerbank kiosks (trim idle slow lockers)."
            ),
            "budget_cap_usd": 80_000.0,
            "pricing_mode": 1.0,
            "pricing_label": "standard_pay_per_rental",
            "baseline_params": {"c_fast": 12, "c_slow": 15, "c_bank": 2},
            "max_iterations": 4,
        },
        {
            "id": "S3a_pricing_free_window",
            "title": "Pricing elasticity: free first 15 minutes",
            "description": (
                "Same $80k hardware search with discounted powerbank pricing "
                "(proxy for free first 15 minutes) to measure reneging response."
            ),
            "budget_cap_usd": 80_000.0,
            "pricing_mode": 0.35,
            "pricing_label": "free_first_15min_proxy",
            "baseline_params": {"c_fast": 12, "c_slow": 14, "c_bank": 2},
            "max_iterations": 3,
        },
        {
            "id": "S3b_pricing_full_metered",
            "title": "Pricing elasticity: full metered charging",
            "description": (
                "Contrast cell: elevated powerbank price (full timed fee) "
                "to estimate demand / reneging elasticity vs free-window."
            ),
            "budget_cap_usd": 80_000.0,
            "pricing_mode": 1.60,
            "pricing_label": "full_metered_fee",
            "baseline_params": {"c_fast": 12, "c_slow": 14, "c_bank": 2},
            "max_iterations": 3,
        },
    ]


def _planner_llm_decompose(goal: str) -> Optional[List[SubtaskSpec]]:
    """Ask Planner LLM to emit four subtasks; return None on failure."""
    if not _ollama_available():
        return None
    try:
        llm = _get_chat_llm()
        prompt = (
            "You are an autonomous experiment planner for airport charging DES. "
            "Decompose the goal into EXACTLY 4 subtasks. "
            "Return ONLY JSON: {\"subtasks\": [ "
            "{\"id\": str, \"title\": str, \"description\": str, "
            "\"budget_cap_usd\": number, \"pricing_mode\": number, "
            "\"pricing_label\": str, "
            "\"baseline_params\": {\"c_fast\": int, \"c_slow\": int, \"c_bank\": int}, "
            "\"max_iterations\": int } x4 ] }. "
            "Required coverage: (1) $40k emergency minimum system; "
            "(2) $80k fast/powerbank gold ratio; "
            "(3) pricing elasticity free-15min vs full metered "
            "(use two pricing_mode values, you may encode both in descriptions "
            "but still return 4 items total — prefer two pricing cells plus "
            "low/standard budget); "
            "(4) synthesis placeholder with budget_cap_usd=80000. "
            f"Goal: {goal}"
        )
        reply = llm.invoke(prompt)
        content = reply.content if hasattr(reply, "content") else str(reply)
        data = _extract_json_object(str(content))
        raw = data.get("subtasks", [])
        if not isinstance(raw, list) or len(raw) < 3:
            return None
        out: List[SubtaskSpec] = []
        for i, item in enumerate(raw[:4]):
            bp = item.get("baseline_params") or {"c_fast": 10, "c_slow": 8, "c_bank": 2}
            out.append(
                {
                    "id": str(item.get("id", f"S{i+1}")),
                    "title": str(item.get("title", f"Subtask {i+1}")),
                    "description": str(item.get("description", "")),
                    "budget_cap_usd": float(item.get("budget_cap_usd", 80_000)),
                    "pricing_mode": float(item.get("pricing_mode", 1.0)),
                    "pricing_label": str(item.get("pricing_label", "standard")),
                    "baseline_params": {
                        "c_fast": int(bp.get("c_fast", 10)),
                        "c_slow": int(bp.get("c_slow", 8)),
                        "c_bank": int(bp.get("c_bank", 2)),
                    },
                    "max_iterations": int(item.get("max_iterations", 3)),
                }
            )
        # Ensure we always have at least the four canonical themes if LLM short.
        if len(out) < 4:
            defaults = _default_subtasks()
            out.extend(defaults[len(out) :])
        return out[:4]
    except Exception:
        return None


def plan_subtasks(goal: str = OVERALL_GOAL) -> tuple[List[SubtaskSpec], str]:
    """
    Dual-LLM planner front-end: LLM decompose or rule template.

    Returns
    -------
    (subtasks, plan_source)
        ``plan_source`` is ``ollama`` or ``rules``.
    """
    llm_plan = _planner_llm_decompose(goal)
    if llm_plan is not None:
        return llm_plan, "ollama"
    return _default_subtasks(), "rules"


def _candidate_from_reflection(
    subtask: SubtaskSpec,
    final_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one Pareto point from a finished reflection search."""
    params = dict(final_state.get("current_params") or {})
    metrics = dict(final_state.get("simulation_results") or {})
    budget = validate_infrastructure_budget(
        int(params.get("c_fast", 0)),
        int(params.get("c_slow", 0)),
        int(params.get("c_bank", 0)),
        budget_cap_usd=float(subtask["budget_cap_usd"]),
        area_cap_m2=AREA_CAP_M2,
    )
    return {
        "subtask_id": subtask["id"],
        "title": subtask["title"],
        "pricing_label": subtask["pricing_label"],
        "pricing_mode": float(subtask["pricing_mode"]),
        "budget_cap_usd": float(subtask["budget_cap_usd"]),
        "params": params,
        "total_cost": float(budget.get("total_cost", 0.0)),
        "total_area": float(budget.get("total_area", 0.0)),
        "mean_wait_time": float(metrics.get("mean_wait_time", 1e9)),
        "reneging_rate": float(metrics.get("reneging_rate", 1.0)),
        "max_queue_len": int(metrics.get("max_queue_len", -1)),
        "max_system_load": int(metrics.get("max_system_load", -1)),
        "utilization_fast": float(metrics.get("utilization_fast", 0.0)),
        "utilization_slow": float(metrics.get("utilization_slow", 0.0)),
        "served_fast": int(metrics.get("served_fast", 0)),
        "served_slow": int(metrics.get("served_slow", 0)),
        "served_powerbank": int(metrics.get("served_powerbank", 0)),
        "converged": bool(final_state.get("converged", False)),
        "iterations": int(final_state.get("iteration", 0)),
    }


def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """
    True if ``a`` Pareto-dominates ``b`` on (cost, wait, reneging) — all minimize.
    """
    a_vec = (
        float(a["total_cost"]),
        float(a["mean_wait_time"]),
        float(a["reneging_rate"]),
    )
    b_vec = (
        float(b["total_cost"]),
        float(b["mean_wait_time"]),
        float(b["reneging_rate"]),
    )
    le_all = all(x <= y + 1e-12 for x, y in zip(a_vec, b_vec))
    lt_any = any(x < y - 1e-12 for x, y in zip(a_vec, b_vec))
    return bool(le_all and lt_any)


def extract_pareto_front(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return non-dominated candidates (cost, wait, reneging)."""
    front: List[Dict[str, Any]] = []
    for c in candidates:
        if any(_dominates(other, c) for other in candidates if other is not c):
            continue
        front.append(c)
    front.sort(key=lambda x: (x["total_cost"], x["mean_wait_time"], x["reneging_rate"]))
    return front


def _rule_aggregate(
    candidates: List[Dict[str, Any]],
    pareto: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Deterministic synthesis notes for Subtask 4."""
    by_id = {c["subtask_id"]: c for c in candidates}
    low = by_id.get("S1_low_budget")
    std = by_id.get("S2_standard_budget")
    free = by_id.get("S3a_pricing_free_window")
    metered = by_id.get("S3b_pricing_full_metered")

    elasticity: Dict[str, Any] = {}
    if free and metered:
        elasticity = {
            "free_window_reneging": free["reneging_rate"],
            "metered_reneging": metered["reneging_rate"],
            "delta_reneging": metered["reneging_rate"] - free["reneging_rate"],
            "free_window_wait": free["mean_wait_time"],
            "metered_wait": metered["mean_wait_time"],
            "interpretation": (
                "Higher metered price should raise reneging if passengers "
                "reject paid banks; free-window proxy uses pricing_mode=0.35."
            ),
        }

    recommended = pareto[0] if pareto else (std or (candidates[0] if candidates else {}))
    return {
        "engine": "rules",
        "n_candidates": len(candidates),
        "n_pareto": len(pareto),
        "recommended_point": recommended,
        "low_budget_emergency": low,
        "standard_budget_gold": std,
        "pricing_elasticity": elasticity,
        "summary": (
            f"Collected {len(candidates)} reflection endpoints; "
            f"{len(pareto)} lie on the (cost, wait, reneging) Pareto front. "
            f"Recommended seed: {recommended.get('params')} "
            f"under ${recommended.get('total_cost', 0):,.0f}."
        ),
    }


def _llm_aggregate(
    candidates: List[Dict[str, Any]],
    pareto: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Second LLM: narrative synthesis over Pareto set."""
    base = _rule_aggregate(candidates, pareto)
    if not _ollama_available():
        return base
    try:
        llm = _get_chat_llm()
        prompt = (
            "You are the aggregator agent for airport charging experiments. "
            "Given candidates and Pareto front, return JSON with keys: "
            "summary (str), recommended_subtask_id (str), "
            "hardware_advice (list[str]), pricing_advice (str). "
            f"Candidates: {json.dumps(candidates)}. "
            f"Pareto: {json.dumps(pareto)}."
        )
        reply = llm.invoke(prompt)
        content = reply.content if hasattr(reply, "content") else str(reply)
        data = _extract_json_object(str(content))
        base["engine"] = "ollama+rules"
        base["llm_summary"] = data
        if data.get("summary"):
            base["summary"] = str(data["summary"])
        return base
    except Exception as exc:
        base["llm_error"] = str(exc)
        return base


def _print_plan(subtasks: List[SubtaskSpec], source: str) -> None:
    """Rich table of the macro plan."""
    table = Table(title=f"Macro experiment plan ({source})", header_style="bold cyan")
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Budget")
    table.add_column("Pricing")
    table.add_column("Baseline")
    for s in subtasks:
        bp = s["baseline_params"]
        table.add_row(
            s["id"],
            s["title"][:40],
            f"${s['budget_cap_usd']:,.0f}",
            f"{s['pricing_label']} ({s['pricing_mode']})",
            f"{bp['c_fast']}/{bp['c_slow']}/{bp['c_bank']}",
        )
    console.print(table)


def run_planning_agent(
    *,
    goal: str = OVERALL_GOAL,
    export_path: Path = SUMMARY_PATH,
    max_search_subtasks: int = 4,
) -> PlanningState:
    """
    Plan → schedule reflection searches → aggregate Pareto → export JSON.

    Subtask 4 in the narrative is the aggregation step (not a fifth DES run).
    Pricing elasticity uses two DES cells (free-window proxy vs metered).
    """
    console.rule("[bold blue]Autonomous experiment planner")
    console.print(
        Panel.fit(
            goal,
            title="Overall goal",
            border_style="blue",
        )
    )
    console.print(
        f"Planner LLM: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL} | "
        f"online={_ollama_available()}"
    )

    # --- Layer 1: Planner decomposition ------------------------------------
    # Default template already encodes S1/S2/S3a/S3b; LLM may override.
    # Aggregation (Subtask 4) is always the final synthesis step below.
    search_cells, plan_source = plan_subtasks(goal)
    # Drop any LLM "synthesis-only" cell that is not meant for DES search.
    executable = [
        s
        for s in search_cells
        if "synth" not in s["id"].lower() and "aggreg" not in s["id"].lower()
    ]
    if not executable:
        executable = _default_subtasks()
        plan_source = "rules"
    executable = executable[:max_search_subtasks]
    _print_plan(executable, plan_source)

    # --- Layer 2: schedule reflection_agent local searches -----------------
    subtask_results: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []

    for idx, sub in enumerate(executable, start=1):
        console.rule(f"[bold yellow]Execute {sub['id']} ({idx}/{len(executable)})")
        console.print(sub["description"])
        log_path = (
            PACK_ROOT
            / "results"
            / f"reflection_{sub['id']}.json"
        )
        try:
            final = run_reflection_agent(
                export_path=log_path,
                budget_cap_usd=float(sub["budget_cap_usd"]),
                area_cap_m2=AREA_CAP_M2,
                pricing_mode=float(sub["pricing_mode"]),
                baseline_params=dict(sub["baseline_params"]),
                max_iterations=int(sub["max_iterations"]),
                quiet=True,
            )
            cand = _candidate_from_reflection(sub, dict(final))
            candidates.append(cand)
            subtask_results.append(
                {
                    "subtask": sub,
                    "status": "ok",
                    "reflection_log": str(log_path),
                    "candidate": cand,
                    "error_message": final.get("error_message"),
                }
            )
            console.print(
                f"[green]OK[/green] params={cand['params']} "
                f"cost=${cand['total_cost']:,.0f} "
                f"wait={cand['mean_wait_time']:.2f} "
                f"reneging={cand['reneging_rate']:.3f}"
            )
        except Exception as exc:
            subtask_results.append(
                {
                    "subtask": sub,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            console.print(f"[red]FAILED {sub['id']}: {exc}[/red]")

    # --- Subtask 4: structured aggregation / Pareto ------------------------
    console.rule("[bold magenta]Subtask 4 · aggregate Pareto set")
    pareto = extract_pareto_front(candidates)
    notes = _llm_aggregate(candidates, pareto)
    console.print(Panel.fit(str(notes.get("summary", "")), title="Aggregator"))

    state: PlanningState = {
        "overall_goal": goal,
        "plan_source": plan_source,
        "subtasks": executable,
        "subtask_results": subtask_results,
        "pareto_candidates": pareto,
        "aggregator_notes": notes,
        "llm_mode": "ollama" if _ollama_available() else "rules",
    }

    export_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "overall_goal": state["overall_goal"],
        "plan_source": state["plan_source"],
        "llm_mode": state["llm_mode"],
        "subtasks": state["subtasks"],
        "subtask_results": state["subtask_results"],
        "all_candidates": candidates,
        "pareto_candidates": state["pareto_candidates"],
        "aggregator_notes": state["aggregator_notes"],
        "meta": {
            "ollama_model": OLLAMA_MODEL,
            "ollama_base_url": OLLAMA_BASE_URL,
            "area_cap_m2": AREA_CAP_M2,
        },
    }
    export_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(
        Panel.fit(
            f"Wrote {export_path}\n"
            f"candidates={len(candidates)} pareto={len(pareto)} "
            f"plan_source={plan_source}",
            title="Planning complete",
            border_style="green",
        )
    )
    return state


def main() -> None:
    """CLI entry for the autonomous planner."""
    try:
        run_planning_agent()
    except Exception:
        console.print("[red]Fatal planning error[/red]")
        console.print(traceback.format_exc())
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(
            json.dumps(
                {"error": traceback.format_exc(), "pareto_candidates": []},
                indent=2,
            ),
            encoding="utf-8",
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

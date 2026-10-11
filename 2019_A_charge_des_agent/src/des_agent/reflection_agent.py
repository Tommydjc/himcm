"""
LangGraph self-reflection agent for airport charging DES sizing.

Loop: propose → validate → simulate → reflect → check_convergence
LLM: Ollama qwen2.5:7b at http://localhost:11434 (rule fallback if offline).

Run (pack root):
  python -m src.des_agent.reflection_agent
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Path bootstrap: support ``python -m src.des_agent.reflection_agent``
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PACK_ROOT / "src"
for _p in (str(SRC_ROOT), str(PACK_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langgraph.graph import END, StateGraph

from des_env.airport_sim import run_airport_simulation
from mcp_server.server import (
    AREA_CAP_M2,
    BUDGET_CAP_USD,
    TARGET_WAIT_MIN,
    validate_infrastructure_budget,
)

console = Console()

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
MAX_ITERATIONS = 5
MAX_BUDGET_REJECTS = 3
TARGET_RENEGING = 0.05
BASELINE_PARAMS: Dict[str, int] = {"c_fast": 12, "c_slow": 15, "c_bank": 2}
RESULTS_PATH = PACK_ROOT / "results" / "des_agent_reflection_log.json"


class ReflectionState(TypedDict):
    """LangGraph shared state for the sizing reflection loop."""

    current_params: Dict[str, Any]
    simulation_results: Dict[str, Any]
    budget_valid: bool
    budget_info: Dict[str, Any]
    reflection_trace: List[Dict[str, Any]]
    iteration: int
    converged: bool
    error_message: Optional[str]
    budget_reject_count: int
    llm_mode: str  # "ollama" | "rules"
    budget_cap_usd: float
    area_cap_m2: float
    pricing_mode: float
    baseline_params: Dict[str, int]
    max_iterations: int


# ---------------------------------------------------------------------------
# LLM helpers (Ollama) + rule fallback
# ---------------------------------------------------------------------------


def _get_chat_llm() -> Any:
    """Build ChatOllama client; raise if import/connection setup fails."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )


def _ollama_available() -> bool:
    """Return True if Ollama answers /api/tags within a short timeout."""
    try:
        import urllib.request

        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return int(resp.status) == 200
    except Exception:
        return False


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Parse the first JSON object from an LLM reply (fences allowed)."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM JSON root is not an object")
    return data


def _clamp_params(params: Dict[str, Any]) -> Dict[str, int]:
    """Force integer non-negative capacities with small upper bounds."""
    return {
        "c_fast": max(0, min(40, int(params.get("c_fast", 0)))),
        "c_slow": max(0, min(30, int(params.get("c_slow", 0)))),
        "c_bank": max(0, min(8, int(params.get("c_bank", 0)))),
    }


def _caps_from_state(state: ReflectionState) -> tuple[float, float]:
    """Read CapEx / area caps from state with MCP defaults."""
    return (
        float(state.get("budget_cap_usd", BUDGET_CAP_USD)),
        float(state.get("area_cap_m2", AREA_CAP_M2)),
    )


def _force_feasible(
    params: Dict[str, int],
    *,
    budget_cap_usd: float = BUDGET_CAP_USD,
    area_cap_m2: float = AREA_CAP_M2,
) -> Dict[str, int]:
    """
    Greedy repair until CapEx and area caps hold.

    Prefer cutting slow lockers (large area), then fast, then banks.
    """
    p = dict(params)
    for _ in range(80):
        info = validate_infrastructure_budget(
            p["c_fast"],
            p["c_slow"],
            p["c_bank"],
            budget_cap_usd=budget_cap_usd,
            area_cap_m2=area_cap_m2,
        )
        if info["is_valid"]:
            return p
        reason = str(info.get("violation_reason", ""))
        if "floor area" in reason and p["c_slow"] > 0:
            p["c_slow"] -= 1
        elif "floor area" in reason and p["c_fast"] > 0:
            p["c_fast"] -= 1
        elif "CapEx" in reason and p["c_fast"] > 1:
            p["c_fast"] -= 1
        elif p["c_slow"] > 0:
            p["c_slow"] -= 1
        elif p["c_fast"] > 1:
            p["c_fast"] -= 1
        elif p["c_bank"] > 0:
            p["c_bank"] -= 1
        else:
            break
    return p


def _rule_propose(state: ReflectionState) -> Dict[str, int]:
    """Deterministic sizing heuristic when Ollama is unavailable."""
    budget_cap, area_cap = _caps_from_state(state)
    baseline = dict(state.get("baseline_params") or BASELINE_PARAMS)
    prev = _clamp_params(state.get("current_params") or baseline)
    results = state.get("simulation_results") or {}
    budget_valid = bool(state.get("budget_valid", True))
    budget_info = state.get("budget_info") or {}

    if not budget_valid:
        repaired = _force_feasible(
            prev, budget_cap_usd=budget_cap, area_cap_m2=area_cap
        )
        if repaired == prev:
            repaired = _force_feasible(
                {
                    "c_fast": max(4, prev["c_fast"] - 2),
                    "c_slow": max(0, prev["c_slow"] - 3),
                    "c_bank": min(8, max(1, prev["c_bank"])),
                },
                budget_cap_usd=budget_cap,
                area_cap_m2=area_cap,
            )
        return repaired

    wait = float(results.get("mean_wait_time", 0.0))
    reneg = float(results.get("reneging_rate", 0.0))
    util_slow = float(results.get("utilization_slow", 0.0))
    util_fast = float(results.get("utilization_fast", 0.0))
    p = dict(prev)

    if reneg > TARGET_RENEGING or wait > TARGET_WAIT_MIN:
        if util_slow < 0.40 and p["c_slow"] > 0:
            p["c_slow"] -= 1
            p["c_fast"] += 1
        else:
            p["c_fast"] += 1
        if reneg > 0.10:
            p["c_bank"] = min(8, p["c_bank"] + 1)
    elif util_slow < 0.25 and p["c_slow"] >= 2:
        p["c_slow"] -= 2
        p["c_fast"] += 1
    elif util_fast > 0.75:
        p["c_fast"] += 1
    else:
        if p["c_slow"] > 0:
            p["c_slow"] -= 1
            p["c_bank"] = min(8, p["c_bank"] + 1)

    _ = budget_info
    return _force_feasible(
        _clamp_params(p), budget_cap_usd=budget_cap, area_cap_m2=area_cap
    )


def _llm_propose(state: ReflectionState) -> Dict[str, int]:
    """Ask Qwen for the next JSON parameter vector under hard caps."""
    llm = _get_chat_llm()
    budget_cap, area_cap = _caps_from_state(state)
    constraints = {
        "budget_cap_usd": budget_cap,
        "area_cap_m2": area_cap,
        "pricing_mode": float(state.get("pricing_mode", 1.0)),
        "unit_cost_usd": {"fast": 2000, "slow": 1200, "bank": 3500},
        "unit_area_m2": {"fast": 0.8, "slow": 2.0, "bank": 0.5},
    }
    prompt = (
        "You size airport charging hardware. "
        "Return ONLY JSON: {\"c_fast\": int, \"c_slow\": int, \"c_bank\": int}. "
        f"Hard caps: CapEx <= ${budget_cap:.0f}, area <= {area_cap} m^2. "
        f"Constraints: {json.dumps(constraints)}. "
        f"Previous params: {json.dumps(state.get('current_params', {}))}. "
        f"Budget info: {json.dumps(state.get('budget_info', {}))}. "
        f"Last simulation: {json.dumps(state.get('simulation_results', {}))}. "
        f"Reflection trace (tail): {json.dumps(state.get('reflection_trace', [])[-2:])}. "
        "Prefer fewer slow lockers if utilization_slow is low. "
        "Prefer more fast chargers or powerbank groups if reneging_rate or wait is high."
    )
    reply = llm.invoke(prompt)
    content = reply.content if hasattr(reply, "content") else str(reply)
    data = _extract_json_object(str(content))
    return _force_feasible(
        _clamp_params(data), budget_cap_usd=budget_cap, area_cap_m2=area_cap
    )

def _rule_reflect(state: ReflectionState) -> Dict[str, Any]:
    """Template bottleneck analysis without an LLM."""
    params = state.get("current_params") or {}
    res = state.get("simulation_results") or {}
    wait = float(res.get("mean_wait_time", 0.0))
    reneg = float(res.get("reneging_rate", 0.0))
    util_slow = float(res.get("utilization_slow", 0.0))
    util_fast = float(res.get("utilization_fast", 0.0))
    max_q = int(res.get("max_queue_len", 0))
    max_load = int(res.get("max_system_load", max_q))

    # Waiting-line / reneging define congestion; in-service load alone is not a jam.
    morning_congestion = (
        wait > TARGET_WAIT_MIN or reneg > TARGET_RENEGING or max_q >= 15
    )
    slow_idle = util_slow < 0.35
    strategy: List[str] = []
    if morning_congestion:
        strategy.append("Increase c_fast before the 08:00 boarding wave.")
    if slow_idle and int(params.get("c_slow", 0)) > 0:
        strategy.append("Cut idle slow lockers to free area for fast chargers.")
    if reneg > TARGET_RENEGING:
        strategy.append("Raise c_bank so reneging passengers can rent power banks.")
    if util_fast > 0.70:
        strategy.append("Fast pool is busy; add 1–2 fast chargers if budget remains.")
    if not strategy:
        strategy.append("KPIs meet targets; keep or slightly trim CapEx.")

    return {
        "engine": "rules",
        "iteration": int(state.get("iteration", 0)),
        "params": dict(params),
        "kpis": {
            "mean_wait_time": wait,
            "reneging_rate": reneg,
            "utilization_fast": util_fast,
            "utilization_slow": util_slow,
            "max_queue_len": max_q,
            "max_system_load": max_load,
        },
        "morning_peak_congestion": bool(morning_congestion),
        "slow_locker_underused": bool(slow_idle),
        "analysis": (
            f"Morning congestion={'yes' if morning_congestion else 'no'}; "
            f"slow util={util_slow:.2f}; reneging={reneg:.3f}; wait={wait:.2f} min; "
            f"max_system_load={max_load}."
        ),
        "strategy": strategy,
    }


def _llm_reflect(state: ReflectionState) -> Dict[str, Any]:
    """Ask Qwen for mechanistic attribution; fall back to rules on parse errors."""
    llm = _get_chat_llm()
    prompt = (
        "You are an airport OR analyst. Analyze this charging DES result. "
        "Answer in JSON with keys: "
        "morning_peak_congestion (bool), slow_locker_underused (bool), "
        "reneging_cause (str), strategy (list of str), analysis (str). "
        "Focus: Did 08:00 peak congest? Is slow-locker utilization low? "
        "Why is reneging high? Give hardware swap advice. "
        f"Params: {json.dumps(state.get('current_params', {}))}. "
        f"Results: {json.dumps(state.get('simulation_results', {}))}."
    )
    reply = llm.invoke(prompt)
    content = reply.content if hasattr(reply, "content") else str(reply)
    data = _extract_json_object(str(content))
    data["engine"] = "ollama"
    data["iteration"] = int(state.get("iteration", 0))
    data["params"] = dict(state.get("current_params") or {})
    return data


def _print_transition(node: str, state: ReflectionState) -> None:
    """Rich one-row status table for the current node."""
    params = state.get("current_params") or {}
    res = state.get("simulation_results") or {}
    table = Table(title=f"LangGraph node · {node}", show_header=True, header_style="bold cyan")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("iteration", str(state.get("iteration", 0)))
    table.add_row(
        "params",
        f"fast={params.get('c_fast')} slow={params.get('c_slow')} bank={params.get('c_bank')}",
    )
    table.add_row("budget_valid", str(state.get("budget_valid")))
    table.add_row(
        "budget_rejects",
        str(state.get("budget_reject_count", 0)),
    )
    table.add_row(
        "mean_wait / reneging",
        f"{res.get('mean_wait_time', '—')} / {res.get('reneging_rate', '—')}",
    )
    table.add_row("converged", str(state.get("converged")))
    table.add_row("llm_mode", str(state.get("llm_mode", "—")))
    if state.get("error_message"):
        table.add_row("error", str(state.get("error_message")))
    console.print(table)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def node_propose(state: ReflectionState) -> Dict[str, Any]:
    """Propose baseline once, then LLM/rule candidates under hard caps."""
    use_llm = _ollama_available()
    llm_mode = "ollama" if use_llm else "rules"
    baseline = dict(state.get("baseline_params") or BASELINE_PARAMS)
    budget_cap, area_cap = _caps_from_state(state)
    is_first = (
        int(state.get("iteration", 0)) == 0
        and not state.get("reflection_trace")
        and not state.get("current_params")
    )

    try:
        if is_first:
            params = dict(baseline)
            msg = f"Initial baseline proposal {params}."
        elif use_llm:
            try:
                params = _llm_propose(state)
                msg = "Ollama proposed next capacities."
            except Exception as exc:
                params = _rule_propose(state)
                llm_mode = "rules"
                msg = f"Ollama propose failed ({exc}); used rules."
        else:
            params = _rule_propose(state)
            msg = "Rules proposed next capacities (Ollama offline)."
    except Exception as exc:
        params = _force_feasible(
            dict(baseline), budget_cap_usd=budget_cap, area_cap_m2=area_cap
        )
        llm_mode = "rules"
        msg = f"Propose hard-fallback: {exc}"

    out: Dict[str, Any] = {
        "current_params": params,
        "llm_mode": llm_mode,
        "error_message": msg,
        "converged": False,
    }
    merged = {**state, **out}
    _print_transition("propose", merged)  # type: ignore[arg-type]
    return out


def node_validate(state: ReflectionState) -> Dict[str, Any]:
    """Validate CapEx and floor area via in-process MCP tool."""
    budget_cap, area_cap = _caps_from_state(state)
    baseline = dict(state.get("baseline_params") or BASELINE_PARAMS)
    params = _clamp_params(state.get("current_params") or baseline)
    info = validate_infrastructure_budget(
        params["c_fast"],
        params["c_slow"],
        params["c_bank"],
        budget_cap_usd=budget_cap,
        area_cap_m2=area_cap,
    )
    valid = bool(info.get("is_valid", False))
    reject_count = int(state.get("budget_reject_count", 0))
    err = state.get("error_message")
    if not valid:
        reject_count += 1
        err = (
            f"Budget/area invalid (reject {reject_count}/{MAX_BUDGET_REJECTS}): "
            f"{info.get('violation_reason', '')}"
        )
        if reject_count >= MAX_BUDGET_REJECTS:
            params = _force_feasible(
                {"c_fast": 8, "c_slow": 6, "c_bank": 1},
                budget_cap_usd=budget_cap,
                area_cap_m2=area_cap,
            )
            info = validate_infrastructure_budget(
                params["c_fast"],
                params["c_slow"],
                params["c_bank"],
                budget_cap_usd=budget_cap,
                area_cap_m2=area_cap,
            )
            valid = bool(info.get("is_valid", False))
            err = (
                f"Exceeded {MAX_BUDGET_REJECTS} budget rejects; "
                f"forced feasible params {params}."
            )
            reject_count = 0
    else:
        reject_count = 0

    out = {
        "current_params": params,
        "budget_valid": valid,
        "budget_info": info,
        "budget_reject_count": reject_count,
        "error_message": err,
    }
    merged = {**state, **out}
    _print_transition("validate", merged)  # type: ignore[arg-type]
    return out


def node_simulate(state: ReflectionState) -> Dict[str, Any]:
    """Run 16-hour SimPy DES for the current capacities."""
    baseline = dict(state.get("baseline_params") or BASELINE_PARAMS)
    params = _clamp_params(state.get("current_params") or baseline)
    pricing = float(state.get("pricing_mode", 1.0))
    try:
        metrics = run_airport_simulation(
            c_fast=params["c_fast"],
            c_slow=params["c_slow"],
            c_bank=params["c_bank"],
            pricing_mode=pricing,
            seed=42 + int(state.get("iteration", 0)),
        )
        metrics = dict(metrics)
        metrics.pop("queue_timeline", None)
        err = None
    except Exception as exc:
        metrics = {
            "mean_wait_time": 1e9,
            "reneging_rate": 1.0,
            "max_queue_len": -1,
            "error": str(exc),
        }
        err = f"Simulation failed: {exc}"

    out = {"simulation_results": metrics, "error_message": err}
    merged = {**state, **out}
    _print_transition("simulate", merged)  # type: ignore[arg-type]
    return out


def node_reflect(state: ReflectionState) -> Dict[str, Any]:
    """Bottleneck attribution via Ollama or rule engine; append trace."""
    use_llm = _ollama_available() and state.get("llm_mode") != "rules"
    try:
        if use_llm:
            try:
                note = _llm_reflect(state)
                mode = "ollama"
            except Exception as exc:
                note = _rule_reflect(state)
                note["fallback_reason"] = str(exc)
                mode = "rules"
        else:
            note = _rule_reflect(state)
            mode = "rules"
    except Exception as exc:
        note = _rule_reflect(state)
        note["fallback_reason"] = str(exc)
        mode = "rules"

    trace = list(state.get("reflection_trace") or [])
    trace.append(note)
    out = {
        "reflection_trace": trace,
        "llm_mode": mode,
        "error_message": None,
    }
    merged = {**state, **out}
    _print_transition("reflect", merged)  # type: ignore[arg-type]
    console.print(
        Panel.fit(
            str(note.get("analysis", note))[:500],
            title="Reflection analysis",
            border_style="magenta",
        )
    )
    return out


def node_check_convergence(state: ReflectionState) -> Dict[str, Any]:
    """Mark converged if KPI targets hold or iteration budget is exhausted."""
    iteration = int(state.get("iteration", 0)) + 1
    max_iters = int(state.get("max_iterations", MAX_ITERATIONS))
    res = state.get("simulation_results") or {}
    wait = float(res.get("mean_wait_time", 1e9))
    reneg = float(res.get("reneging_rate", 1.0))
    kpi_ok = (wait <= TARGET_WAIT_MIN) and (reneg <= TARGET_RENEGING)
    done = bool(kpi_ok or iteration >= max_iters)
    msg = None
    if kpi_ok:
        msg = (
            f"Converged: mean_wait={wait:.2f} <= {TARGET_WAIT_MIN}, "
            f"reneging={reneg:.3f} <= {TARGET_RENEGING}."
        )
    elif done:
        msg = f"Stopped at iteration={iteration} without full KPI success."

    out = {
        "iteration": iteration,
        "converged": done,
        "error_message": msg,
    }
    merged = {**state, **out}
    _print_transition("check_convergence", merged)  # type: ignore[arg-type]
    return out


# ---------------------------------------------------------------------------
# Conditional routers
# ---------------------------------------------------------------------------


def route_after_validate(
    state: ReflectionState,
) -> Literal["simulate", "propose"]:
    """Budget OK → simulate; else re-propose (reject counter handled in node)."""
    if state.get("budget_valid"):
        return "simulate"
    return "propose"


def route_after_check(
    state: ReflectionState,
) -> Literal["propose", "end"]:
    """Converged → END; else another propose round."""
    if state.get("converged"):
        return "end"
    return "propose"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_reflection_graph() -> Any:
    """Compile the propose→validate→simulate→reflect→check StateGraph."""
    graph = StateGraph(ReflectionState)
    graph.add_node("propose", node_propose)
    graph.add_node("validate", node_validate)
    graph.add_node("simulate", node_simulate)
    graph.add_node("reflect", node_reflect)
    graph.add_node("check_convergence", node_check_convergence)

    graph.set_entry_point("propose")
    graph.add_edge("propose", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"simulate": "simulate", "propose": "propose"},
    )
    graph.add_edge("simulate", "reflect")
    graph.add_edge("reflect", "check_convergence")
    graph.add_conditional_edges(
        "check_convergence",
        route_after_check,
        {"propose": "propose", "end": END},
    )
    return graph.compile()


def _initial_state(
    *,
    budget_cap_usd: float = BUDGET_CAP_USD,
    area_cap_m2: float = AREA_CAP_M2,
    pricing_mode: float = 1.0,
    baseline_params: Optional[Dict[str, int]] = None,
    max_iterations: int = MAX_ITERATIONS,
) -> ReflectionState:
    """Empty starting state for a fresh optimization run."""
    return {
        "current_params": {},
        "simulation_results": {},
        "budget_valid": False,
        "budget_info": {},
        "reflection_trace": [],
        "iteration": 0,
        "converged": False,
        "error_message": None,
        "budget_reject_count": 0,
        "llm_mode": "rules",
        "budget_cap_usd": float(budget_cap_usd),
        "area_cap_m2": float(area_cap_m2),
        "pricing_mode": float(pricing_mode),
        "baseline_params": dict(baseline_params or BASELINE_PARAMS),
        "max_iterations": int(max_iterations),
    }


def run_reflection_agent(
    *,
    export_path: Path = RESULTS_PATH,
    budget_cap_usd: float = BUDGET_CAP_USD,
    area_cap_m2: float = AREA_CAP_M2,
    pricing_mode: float = 1.0,
    baseline_params: Optional[Dict[str, int]] = None,
    max_iterations: int = MAX_ITERATIONS,
    quiet: bool = False,
) -> ReflectionState:
    """
    Execute the full reflection loop and persist the trajectory JSON.

    Parameters
    ----------
    export_path :
        JSON log path.
    budget_cap_usd, area_cap_m2 :
        Hard CapEx / floor-area caps for this search.
    pricing_mode :
        Power-bank price multiplier passed to SimPy.
    baseline_params :
        First propose vector (defaults to 12/15/2).
    max_iterations :
        Outer reflection rounds before forced stop.
    quiet :
        If True, suppress Rich banners (planning driver uses this).

    Returns
    -------
    ReflectionState
        Final graph state after convergence or iteration cap.
    """
    if not quiet:
        console.rule("[bold green]DES reflection agent")
        console.print(
            f"Ollama target: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL} | "
            f"online={_ollama_available()} | "
            f"budget=${budget_cap_usd:,.0f} pricing={pricing_mode}"
        )
    app = build_reflection_graph()
    final_state: ReflectionState = app.invoke(
        _initial_state(
            budget_cap_usd=budget_cap_usd,
            area_cap_m2=area_cap_m2,
            pricing_mode=pricing_mode,
            baseline_params=baseline_params,
            max_iterations=max_iterations,
        )
    )  # type: ignore[assignment]

    export_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "current_params": final_state.get("current_params"),
        "simulation_results": final_state.get("simulation_results"),
        "budget_valid": final_state.get("budget_valid"),
        "budget_info": final_state.get("budget_info"),
        "reflection_trace": final_state.get("reflection_trace"),
        "iteration": final_state.get("iteration"),
        "converged": final_state.get("converged"),
        "error_message": final_state.get("error_message"),
        "budget_reject_count": final_state.get("budget_reject_count"),
        "llm_mode": final_state.get("llm_mode"),
        "pricing_mode": final_state.get("pricing_mode"),
        "targets": {
            "mean_wait_min": TARGET_WAIT_MIN,
            "reneging_rate": TARGET_RENEGING,
            "budget_cap_usd": float(budget_cap_usd),
            "area_cap_m2": float(area_cap_m2),
            "max_iterations": int(max_iterations),
            "pricing_mode": float(pricing_mode),
        },
    }
    export_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not quiet:
        console.print(
            Panel.fit(
                f"converged={final_state.get('converged')} | "
                f"iters={final_state.get('iteration')} | "
                f"params={final_state.get('current_params')} | "
                f"log={export_path}",
                title="Final state",
                border_style="green",
            )
        )
    return final_state

def main() -> None:
    """CLI entry: run agent and guarantee a JSON artifact even on crash."""
    try:
        run_reflection_agent()
    except Exception:
        console.print("[red]Fatal error in reflection agent[/red]")
        console.print(traceback.format_exc())
        # Persist a minimal failure log so the pipeline remains inspectable.
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(
            json.dumps(
                {
                    "converged": False,
                    "error_message": traceback.format_exc(),
                    "reflection_trace": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

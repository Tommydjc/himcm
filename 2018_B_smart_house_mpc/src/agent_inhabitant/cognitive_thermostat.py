"""
Local-LLM household MAS that retunes MPC comfort weights (Prompt 6).

Agents
------
- Agent_EMS: grid/TOU steward — propose peak shedding / higher afternoon set.
- Agent_Senior: cold-sensitive elder — floor ≥ 21.5°C (no deep pre-cool).
- Agent_Parent: bill-conscious parent — allow brief 25°C on peak.

Backend
-------
1. ``langchain_ollama.ChatOllama`` (qwen2.5:7b) if installed and reachable.
2. Else raw Ollama HTTP ``/api/generate``.
3. Else deterministic heuristic Mock — never crash.

Run:
  PYTHONPATH=src python -m agent_inhabitant.cognitive_thermostat
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Heavy physics/MPC imports are deferred to closed_loop_with_mas() so that
# pure negotiation / Mock unit tests do not pull SciPy at import time.

PACK_ROOT = Path(__file__).resolve().parents[2]
DIALOGUE_JSON = PACK_ROOT / "results" / "agent_thermostat_dialogue.json"
COMPARE_JSON = PACK_ROOT / "results" / "agent_mpc_vs_bang.json"

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"


# ---------------------------------------------------------------------------
# Pydantic-compatible contract (with stdlib fallback)
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, Field

    class ComfortConsensus(BaseModel):
        """Structured MAS output fed into MPCComfortConfig."""

        agreed_temp_min: float = Field(..., ge=20.0, le=24.0)
        agreed_temp_max: float = Field(..., ge=23.0, le=27.0)
        comfort_weight_scale: float = Field(..., ge=0.5, le=3.0)
        consensus_notes: str

except ImportError:  # pragma: no cover - pydantic normally present

    @dataclass
    class ComfortConsensus:  # type: ignore[no-redef]
        """Stdlib fallback when pydantic is unavailable."""

        agreed_temp_min: float
        agreed_temp_max: float
        comfort_weight_scale: float
        consensus_notes: str

        def model_dump(self) -> Dict[str, Any]:
            return asdict(self)


def consensus_to_comfort(c: ComfortConsensus) -> Any:
    """Map MAS contract → ``MPCComfortConfig`` (lazy-imported)."""
    from control_mpc.mpc_controller import MPCComfortConfig

    data = c.model_dump() if hasattr(c, "model_dump") else asdict(c)  # type: ignore[arg-type]
    t_min = float(data["agreed_temp_min"])
    t_max = float(data["agreed_temp_max"])
    scale = float(data["comfort_weight_scale"])
    return MPCComfortConfig(
        T_set_C=0.5 * (t_min + t_max),
        T_precool_C=max(t_min, 21.5),
        T_peak_C=min(t_max, 25.0),
        T_min_C=t_min,
        T_max_C=t_max,
        lambda_comf=0.05 * scale,
    )


# ---------------------------------------------------------------------------
# Agent personas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentPersona:
    """One household negotiator."""

    agent_id: str
    role: str
    priority: str
    hard_constraint: str


AGENTS: Sequence[AgentPersona] = (
    AgentPersona(
        agent_id="Agent_EMS",
        role="Thermostat / EMS",
        priority="Minimize on-peak TOU spend using wall pre-cool and peak shed.",
        hard_constraint="May propose higher afternoon setpoints when rate is $0.48/kWh.",
    ),
    AgentPersona(
        agent_id="Agent_Senior",
        role="Elder inhabitant",
        priority="Avoid cold drafts and deep pre-cool; fear of catching cold.",
        hard_constraint="Indoor never below 21.5°C; reject aggressive pre-cool.",
    ),
    AgentPersona(
        agent_id="Agent_Parent",
        role="Eco parent / bill payer",
        priority="Accept mild warmth on peak if it cuts the bill.",
        hard_constraint="Allow brief float up to 25.0°C during 14:00–19:00.",
    ),
)


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


def ollama_reachable(timeout_s: float = 1.5) -> bool:
    """True if local Ollama answers /api/tags."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status) == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Parse first JSON object from an LLM string."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _prompt_for_peak_day(context: Dict[str, Any]) -> str:
    """Build the multi-agent negotiation prompt."""
    roster = [
        {
            "id": a.agent_id,
            "role": a.role,
            "priority": a.priority,
            "hard_constraint": a.hard_constraint,
        }
        for a in AGENTS
    ]
    return (
        "You are mediating a household climate negotiation for an extreme TOU peak day.\n"
        "Return ONLY valid JSON with keys:\n"
        '  agreed_temp_min, agreed_temp_max, comfort_weight_scale, consensus_notes\n'
        "Constraints that MUST hold:\n"
        "  - agreed_temp_min >= 21.5\n"
        "  - agreed_temp_max <= 25.0\n"
        "  - comfort_weight_scale in [0.8, 2.0]\n"
        "Respect the elder's cold sensitivity and the parent's bill goals.\n"
        f"Agents: {json.dumps(roster, ensure_ascii=False)}\n"
        f"Context: {json.dumps(context, ensure_ascii=False)}\n"
    )


def negotiate_via_langchain(context: Dict[str, Any]) -> Optional[ComfortConsensus]:
    """Try langchain_ollama ChatOllama; return None on any failure."""
    if not ollama_reachable(timeout_s=0.8):
        return None
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return None
    try:
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.2,
            format="json",
            timeout=12.0,
        )
        msg = llm.invoke(
            [
                SystemMessage(content="Output only compact JSON. No markdown."),
                HumanMessage(content=_prompt_for_peak_day(context)),
            ]
        )
        raw = msg.content if isinstance(msg.content, str) else str(msg.content)
        data = _extract_json_object(raw)
        return ComfortConsensus(**data)
    except Exception:
        return None


def negotiate_via_http(context: Dict[str, Any]) -> Optional[ComfortConsensus]:
    """Raw Ollama /api/generate JSON path (hard 12s timeout)."""
    if not ollama_reachable(timeout_s=0.8):
        return None
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": _prompt_for_peak_day(context),
            "stream": False,
            "format": "json",
            "options": {"num_predict": 256},
        }
    ).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = payload.get("response", "{}")
        data = _extract_json_object(text if isinstance(text, str) else json.dumps(text))
        return ComfortConsensus(**data)
    except Exception:
        return None


def negotiate_via_mock(context: Dict[str, Any]) -> ComfortConsensus:
    """
    Deterministic heuristic bargain (offline / CI safe).

    Raises elder floor to 21.5°C, allows peak float to 25°C, scales λ_comf.
    """
    peak = bool(context.get("extreme_peak_day", True))
    scale = 1.25 if peak else 1.0
    return ComfortConsensus(
        agreed_temp_min=21.5,
        agreed_temp_max=25.0 if peak else 24.5,
        comfort_weight_scale=scale,
        consensus_notes=(
            "Mock consensus: respect elder anti-cold floor at 21.5°C; "
            "allow brief peak float to 25.0°C to cut TOU spend; "
            f"comfort_weight_scale={scale}."
        ),
    )


def run_household_negotiation(
    *,
    extreme_peak_day: bool = True,
    T_out_peak_C: float = 38.0,
    peak_rate: float = 0.48,
) -> Dict[str, Any]:
    """
    Run EMS / Senior / Parent negotiation; always returns a valid contract.

    Preference order: LangChain Ollama → HTTP Ollama → Mock.
    """
    context = {
        "extreme_peak_day": extreme_peak_day,
        "T_out_peak_C": T_out_peak_C,
        "peak_rate_usd_per_kwh": peak_rate,
        "peak_hours": "14:00-19:00",
    }
    dialogue: List[Dict[str, str]] = [
        {
            "speaker": a.agent_id,
            "utterance": f"{a.priority} Hard rule: {a.hard_constraint}",
        }
        for a in AGENTS
    ]
    # Prefer live LLM only when tags endpoint is up; otherwise Mock immediately.
    consensus: Optional[ComfortConsensus] = None
    source = "mock_heuristic"
    if ollama_reachable(timeout_s=0.8):
        consensus = negotiate_via_langchain(context)
        if consensus is not None:
            source = "langchain_ollama"
        else:
            consensus = negotiate_via_http(context)
            if consensus is not None:
                source = "ollama_http"
    if consensus is None:
        consensus = negotiate_via_mock(context)
        source = "mock_heuristic"

    # Enforce elder floor even if LLM drifts.
    dump = consensus.model_dump() if hasattr(consensus, "model_dump") else asdict(consensus)
    dump["agreed_temp_min"] = max(21.5, float(dump["agreed_temp_min"]))
    dump["agreed_temp_max"] = min(25.0, max(float(dump["agreed_temp_max"]), dump["agreed_temp_min"] + 0.5))
    consensus = ComfortConsensus(**dump)

    dialogue.append(
        {
            "speaker": "Mediator",
            "utterance": dump["consensus_notes"],
        }
    )
    # Build comfort dict without importing SciPy-backed MPC module.
    t_min = float(dump["agreed_temp_min"])
    t_max = float(dump["agreed_temp_max"])
    scale = float(dump["comfort_weight_scale"])
    comfort_dict = {
        "T_set_C": 0.5 * (t_min + t_max),
        "T_precool_C": max(t_min, 21.5),
        "T_peak_C": min(t_max, 25.0),
        "T_min_C": t_min,
        "T_max_C": t_max,
        "lambda_comf": 0.05 * scale,
    }
    return {
        "source": source,
        "context": context,
        "agents": [asdict(a) for a in AGENTS],
        "dialogue": dialogue,
        "consensus": dump,
        "comfort_config": comfort_dict,
    }


def closed_loop_with_mas(
    *,
    extreme_peak_day: bool = True,
    mpc_maxiter: int = 30,
) -> Dict[str, Any]:
    """Negotiate → inject into MPC → compare vs Bang-Bang on teaching day."""
    from baseline_thermostat.bang_bang import run_bang_bang_day, write_weather_csv
    from control_mpc.mpc_controller import (
        MPCComfortConfig,
        compare_to_bang,
        load_weather_1min,
        simulate_mpc_day,
    )

    bargain = run_household_negotiation(extreme_peak_day=extreme_peak_day)
    comfort = MPCComfortConfig(**bargain["comfort_config"])
    write_weather_csv()
    weather = load_weather_1min()
    bang = run_bang_bang_day(weather)
    mpc = simulate_mpc_day(weather=weather, comfort=comfort, maxiter=mpc_maxiter)
    cmp = compare_to_bang(mpc["metrics"], bang["metrics"])
    return {
        "bargain": bargain,
        "bang_metrics": bang["metrics"],
        "mpc_metrics": mpc["metrics"],
        "comparison": cmp,
        "mpc_trajectory": mpc["trajectory"],
    }


def main() -> None:
    """Run MAS bargain + MPC day; export dialogue JSON."""
    print("[mas] negotiating household comfort contract…")
    out = closed_loop_with_mas()
    DIALOGUE_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bargain": out["bargain"],
        "bang_metrics": out["bang_metrics"],
        "mpc_metrics": out["mpc_metrics"],
        "comparison": {
            k: out["comparison"][k]
            for k in (
                "cost_savings_pct",
                "energy_savings_pct",
                "peak_energy_savings_pct",
                "short_cycle_reduction",
            )
        },
    }
    DIALOGUE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    COMPARE_JSON.write_text(json.dumps(out["comparison"], indent=2, default=str), encoding="utf-8")
    c = out["bargain"]["consensus"]
    print(f"[mas] source = {out['bargain']['source']}")
    print(
        f"[mas] band = [{c['agreed_temp_min']}, {c['agreed_temp_max']}]°C  "
        f"λ_scale = {c['comfort_weight_scale']}"
    )
    print(f"[mas] wrote {DIALOGUE_JSON}")
    print(
        f"[mas] Bang ${out['bang_metrics']['cost_usd']:.3f} → "
        f"MPC ${out['mpc_metrics']['cost_usd']:.3f} "
        f"(cost Δ {out['comparison']['cost_savings_pct']:.1f}%)"
    )


if __name__ == "__main__":
    main()

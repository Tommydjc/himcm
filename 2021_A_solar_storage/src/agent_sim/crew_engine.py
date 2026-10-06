r"""Local Ollama + crewAI household crisis negotiation engine.

Sequential process
    Task 1 — butler posts a blizzard / SOC<20% brief and lists
             sheddable catalog loads.
    Task 2 — parent then remote worker each table a concession.
    Task 3 — butler adjudicates into ``LoadSheddingAgreement``.

If Ollama is down, the requested model is missing, or crewAI is not
installed, ``run_family_negotiation`` prints a hint and returns a
catalog-based heuristic mock. The process never crashes on LLM outage.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import pandas as pd

try:
    from crewai import Agent, Crew, LLM, Process, Task
except ImportError:
    Agent = None  # type: ignore[misc, assignment]
    Crew = None  # type: ignore[misc, assignment]
    LLM = None  # type: ignore[misc, assignment]
    Process = None  # type: ignore[misc, assignment]
    Task = None  # type: ignore[misc, assignment]

from src.agent_sim.models import LoadSheddingAgreement

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
CATALOG_CSV: Path = PACK_ROOT / "data" / "raw" / "appliances_catalog.csv"
RESULTS_DIR: Path = PACK_ROOT / "results"
DEFAULT_AGREEMENT_JSON: Path = RESULTS_DIR / "crew_load_shedding_agreement.json"

DEFAULT_OLLAMA_MODEL: str = "ollama/qwen2.5:7b"
DEFAULT_OLLAMA_URL: str = "http://localhost:11434"

PROTECTED: tuple[str, ...] = (
    "Refrigerator_Freezer",
    "WiFi_Router_Comm",
    "Home_Office_Laptop",
    "Well_Water_Pump",
)

# Coincidence-weighted kW used only for the heuristic mock / post-check.
MOCK_SHED: tuple[str, ...] = (
    "Electric_Clothes_Dryer",
    "Dishwasher",
)
MOCK_HVAC_OFFSET_C: float = 2.5
SOC_LINE: float = 0.20


def load_appliance_catalog(path: Path | None = None) -> pd.DataFrame:
    """Read ``appliances_catalog.csv`` (name, watts, priority, deferrable)."""

    csv_path = Path(path) if path is not None else CATALOG_CSV
    frame = pd.read_csv(csv_path)
    need = {
        "appliance_name",
        "rated_power_W",
        "priority_level",
        "is_deferrable",
    }
    missing = need.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns {missing}")
    return frame


def catalog_brief(frame: pd.DataFrame) -> str:
    """Plain-text table injected into butler Task 1."""

    lines = [
        "appliance_name | rated_kW | priority | deferrable",
        "-------------- | -------- | -------- | ----------",
    ]
    for _, row in frame.iterrows():
        kw = float(row["rated_power_W"]) / 1000.0
        lines.append(
            f"{row['appliance_name']} | {kw:.3f} | "
            f"{row['priority_level']} | {row['is_deferrable']}"
        )
    return "\n".join(lines)


def _model_slug() -> str:
    raw = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
    if raw.startswith("ollama/"):
        return raw
    return f"ollama/{raw}"


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")


def _bare_model_name(slug: str) -> str:
    name = slug.split("/", 1)[-1]
    return name


def check_ollama_status(
    base_url: str | None = None,
    model_slug: str | None = None,
    timeout_s: float = 2.0,
) -> tuple[bool, str]:
    """Probe ``GET /api/tags``. Returns ``(ok, message)``; never raises."""

    url = (base_url or _ollama_base_url()) + "/api/tags"
    want = _bare_model_name(model_slug or _model_slug())
    try:
        with urlopen(url, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8")
    except (URLError, TimeoutError, OSError) as exc:
        msg = (
            f"Ollama is not reachable at {url} ({exc}). "
            "Start it with `ollama serve`, then `ollama pull qwen2.5:7b`. "
            "Falling back to the local heuristic mock."
        )
        print(msg)
        return False, msg
    try:
        import json

        data = json.loads(payload)
    except ValueError:
        msg = "Ollama /api/tags returned non-JSON; using heuristic mock."
        print(msg)
        return False, msg
    names: list[str] = []
    for row in data.get("models", []):
        names.append(str(row.get("name", "")))
        names.append(str(row.get("model", "")))
    have = {n for n in names if n}
    aliases = {want, want.split(":")[0], f"{want.split(':')[0]}:latest"}
    if not any(a in have or any(h.startswith(want) for h in have) for a in aliases):
        listed = ", ".join(sorted({n for n in have if n})) or "(none)"
        msg = (
            f"Ollama is up but model '{want}' is not pulled. "
            f"Installed: {listed}. Run `ollama pull {want}`. "
            "Falling back to the local heuristic mock."
        )
        print(msg)
        return False, msg
    ok_msg = f"Ollama ready: {want} at {base_url or _ollama_base_url()}"
    print(ok_msg)
    return True, ok_msg


def _coincidence_kw(row: pd.Series) -> float:
    """Expected kW if the appliance is shed during a crisis hour."""

    rated = float(row["rated_power_W"]) / 1000.0
    duty = float(row.get("duty_cycle", 1.0))
    return rated * max(min(duty, 1.0), 0.05)


def heuristic_agreement(
    catalog: pd.DataFrame,
    *,
    soc: float = 0.18,
    target_cut_frac: float = 0.45,
) -> LoadSheddingAgreement:
    """Parent/worker compromise without an LLM (deterministic mock)."""

    by_name = catalog.set_index("appliance_name")
    shed: list[str] = []
    cut = 0.0
    for name in MOCK_SHED:
        if name not in by_name.index:
            continue
        if name in PROTECTED:
            continue
        shed.append(name)
        cut += _coincidence_kw(by_name.loc[name])
    if "Heat_Pump_HVAC" in by_name.index:
        hvac = by_name.loc["Heat_Pump_HVAC"]
        # 2.5 C of a 20 C setpoint ≈ 12.5% of heat-pump electrical load.
        cut += _coincidence_kw(hvac) * (MOCK_HVAC_OFFSET_C / 20.0)
    if "LED_Lighting_WholeHouse" in by_name.index:
        cut += 0.45 * _coincidence_kw(by_name.loc["LED_Lighting_WholeHouse"])
    rationale = (
        f"SOC {soc:.0%} is below the {SOC_LINE:.0%} blizzard line "
        f"(target cut {target_cut_frac:.0%}). "
        "Worker keeps laptop/router; parent defers dryer and dishwasher; "
        f"HVAC setpoint lowered by {MOCK_HVAC_OFFSET_C:.1f} C. Heuristic mock."
    )
    return LoadSheddingAgreement(
        consensus_reached=True,
        shed_appliances=shed,
        hvac_temp_offset_c=MOCK_HVAC_OFFSET_C,
        negotiated_reduction_kW=round(cut, 3),
        rationale=rationale,
    )


def sanitize_agreement(
    raw: LoadSheddingAgreement,
    catalog: pd.DataFrame,
) -> LoadSheddingAgreement:
    """Drop unknown / protected names; recompute kW if the LLM omitted it."""

    valid = set(catalog["appliance_name"].astype(str))
    shed = [n for n in raw.shed_appliances if n in valid and n not in PROTECTED]
    cut = float(raw.negotiated_reduction_kW)
    if cut <= 1.0e-9 and shed:
        by_name = catalog.set_index("appliance_name")
        cut = float(sum(_coincidence_kw(by_name.loc[n]) for n in shed))
        if raw.hvac_temp_offset_c > 0.0 and "Heat_Pump_HVAC" in by_name.index:
            cut += _coincidence_kw(by_name.loc["Heat_Pump_HVAC"]) * (
                float(raw.hvac_temp_offset_c) / 20.0
            )
    return LoadSheddingAgreement(
        consensus_reached=bool(raw.consensus_reached),
        shed_appliances=shed,
        hvac_temp_offset_c=float(raw.hvac_temp_offset_c),
        negotiated_reduction_kW=round(cut, 3),
        rationale=str(raw.rationale).strip() or "Adjudicated family compromise.",
    )


def _try_import_crewai() -> bool:
    if Agent is None or Crew is None or LLM is None or Process is None or Task is None:
        print(
            "crewAI is not installed (`pip install crewai`). "
            "Falling back to the local heuristic mock."
        )
        return False
    return True


def _build_local_llm() -> Any:
    return LLM(
        model=_model_slug(),
        base_url=_ollama_base_url(),
    )


def _build_crew(
    catalog_text: str,
    soc: float,
    target_cut_frac: float,
) -> Any:
    """Wire three personas and the sequential Task 1–3 pipeline."""

    local_llm = _build_local_llm()

    butler_agent = Agent(
        role="Microgrid Energy Specialist",
        goal=(
            "Protect battery SOC and household supply-demand balance. "
            "Issue crisis alerts and quantify the kW that must be shed."
        ),
        backstory=(
            "You operate an off-grid PV+storage EMS. The hard SOC safety "
            f"line is {SOC_LINE:.0%}. You never invent loads that are not "
            "in the appliance catalog. Critical circuits (fridge, well pump, "
            "router, work laptop) stay energized."
        ),
        llm=local_llm,
        verbose=True,
        allow_delegation=False,
    )
    parent_agent = Agent(
        role="Pragmatic Household Manager",
        goal=(
            "Keep food cold and a minimum emergency circuit alive. "
            "Prefer shutting the dryer and non-essential heating."
        ),
        backstory=(
            "Risk-averse parent. You would rather sit in a cooler house "
            "than risk a blackout that spoils the freezer. You unplug "
            "deferrable wet appliances first."
        ),
        llm=local_llm,
        verbose=True,
        allow_delegation=False,
    )
    worker_agent = Agent(
        role="Remote Knowledge Worker",
        goal=(
            "Defend power for the computer and Wi-Fi router. "
            "Concede on dishwasher and washing-machine delay."
        ),
        backstory=(
            "You work from home. A dropped VPN is unacceptable. You can "
            "skip drying clothes and dim lights, but the laptop and router "
            "must not be shed."
        ),
        llm=local_llm,
        verbose=True,
        allow_delegation=False,
    )

    task_1 = Task(
        name="Task_1_butler_blizzard_brief",
        description=(
            f"Blizzard crisis brief. Current battery SOC is {soc:.1%}, "
            f"below the {SOC_LINE:.0%} safety line. The household must cut "
            f"about {target_cut_frac:.0%} of electrical load.\n\n"
            "Appliance catalog (only these names may be shed):\n"
            f"{catalog_text}\n\n"
            "List sheddable (deferrable / low-priority) loads with rated kW. "
            "Name protected circuits that must stay on. State the kW quota."
        ),
        expected_output=(
            "A short crisis bulletin: SOC, cut quota, sheddable catalog names, "
            "and protected names."
        ),
        agent=butler_agent,
    )
    task_2_parent = Task(
        name="Task_2a_parent_concession",
        description=(
            "Respond to the butler brief. Propose which catalog appliances "
            "you would shut or defer, and how many degrees to lower heating. "
            "Keep fridge and well pump on. Be concrete; use catalog names."
        ),
        expected_output=(
            "Parent concession: shed list, HVAC offset in C, and a one-line why."
        ),
        agent=parent_agent,
        context=[task_1],
    )
    task_2_worker = Task(
        name="Task_2b_worker_concession",
        description=(
            "Respond to the butler brief and the parent's plan. You may delay "
            "the dishwasher and washing machine. You refuse to shed "
            "Home_Office_Laptop or WiFi_Router_Comm. Offer a compromise."
        ),
        expected_output=(
            "Worker concession: what you will give up, what you will not, "
            "catalog names only."
        ),
        agent=worker_agent,
        context=[task_1, task_2_parent],
    )
    task_3 = Task(
        name="Task_3_butler_verdict",
        description=(
            "Adjudicate the parent and worker offers into one household "
            "contract. Use only catalog appliance names. Never shed "
            "Refrigerator_Freezer, WiFi_Router_Comm, Home_Office_Laptop, "
            "or Well_Water_Pump. HVAC offset is degrees the heating setpoint "
            "is LOWERED (positive number, e.g. 2.5). Estimate negotiated "
            "reduction in kW from catalog rated watts and duty cycles. "
            "Set consensus_reached true only if both roles keep their "
            "non-negotiables."
        ),
        expected_output=(
            "A LoadSheddingAgreement object: consensus_reached, "
            "shed_appliances, hvac_temp_offset_c, negotiated_reduction_kW, "
            "rationale."
        ),
        agent=butler_agent,
        context=[task_1, task_2_parent, task_2_worker],
        output_pydantic=LoadSheddingAgreement,
    )
    return Crew(
        agents=[butler_agent, parent_agent, worker_agent],
        tasks=[task_1, task_2_parent, task_2_worker, task_3],
        process=Process.sequential,
        verbose=True,
        tracing=False,
    )


def _agreement_from_crew_result(result: Any, catalog: pd.DataFrame) -> LoadSheddingAgreement:
    raw_model = getattr(result, "pydantic", None)
    if isinstance(raw_model, LoadSheddingAgreement):
        return sanitize_agreement(raw_model, catalog)
    raw_text = str(getattr(result, "raw", result))
    try:
        parsed = LoadSheddingAgreement.model_validate_json(raw_text)
        return sanitize_agreement(parsed, catalog)
    except Exception:
        return heuristic_agreement(catalog)


def run_family_negotiation(
    *,
    soc: float = 0.18,
    target_cut_frac: float = 0.45,
    catalog_path: Path | None = None,
    force_mock: bool = False,
    out_json: Path | None = None,
) -> LoadSheddingAgreement:
    """Run crewAI+Ollama, or the heuristic mock. Always returns an agreement."""

    catalog = load_appliance_catalog(catalog_path)
    if force_mock:
        agreement = heuristic_agreement(
            catalog, soc=soc, target_cut_frac=target_cut_frac
        )
        _write_json(agreement, out_json)
        return agreement

    ollama_ok, _status = check_ollama_status()
    crew_ok = _try_import_crewai() if ollama_ok else False
    if not ollama_ok or not crew_ok:
        agreement = heuristic_agreement(
            catalog, soc=soc, target_cut_frac=target_cut_frac
        )
        _write_json(agreement, out_json)
        return agreement

    brief = catalog_brief(catalog)
    try:
        crew = _build_crew(brief, soc, target_cut_frac)
        result = crew.kickoff()
        agreement = _agreement_from_crew_result(result, catalog)
    except Exception as exc:
        print(
            f"crewAI/Ollama run failed ({exc!r}). "
            "Falling back to the local heuristic mock."
        )
        agreement = heuristic_agreement(
            catalog, soc=soc, target_cut_frac=target_cut_frac
        )
    _write_json(agreement, out_json)
    return agreement


def _write_json(agreement: LoadSheddingAgreement, out_json: Path | None) -> None:
    path = Path(out_json) if out_json is not None else DEFAULT_AGREEMENT_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(agreement.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    """CLI: one full negotiation against local Ollama, with mock fallback."""

    print("=== household crewAI negotiation (Ollama) ===")
    print(f"model={_model_slug()}  base={_ollama_base_url()}")
    agreement = run_family_negotiation(soc=0.18, target_cut_frac=0.45)
    print(agreement.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

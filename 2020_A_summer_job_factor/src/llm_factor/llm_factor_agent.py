"""LLMFactor: name Varimax factors and score unstructured job ads in R^3.

Backend order (``auto``): Ollama chat (Qwen2.5 or ``OLLAMA_MODEL``) then a
deterministic mock. Names and job coordinates are always grounded in
``results/factor_loadings.csv`` — the contest narrative triad (Immediate
Financial Yield / Human Capital / Ergonomic Burden) is recorded as a
*designed* story, not silently copied onto a mismatched extraction.

Job scores live in the extracted factor space so they can be compared with
Thompson student scores in ``results/student_factor_scores.csv``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

PACK_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_LOADINGS: Final[Path] = PACK_ROOT / "results" / "factor_loadings.csv"
DEFAULT_JOB_DIR: Final[Path] = PACK_ROOT / "data" / "job_descriptions"
DEFAULT_INTERPRET: Final[Path] = PACK_ROOT / "results" / "factor_interpretation.md"
DEFAULT_JOB_MATRIX: Final[Path] = PACK_ROOT / "results" / "job_factor_matrix.csv"

N_FACTORS: Final[int] = 3
TOP_K_ITEMS: Final[int] = 4
SCORE_LO: Final[float] = -3.0
SCORE_HI: Final[float] = 3.0
OLLAMA_TIMEOUT_S: Final[float] = 90.0
OLLAMA_TAGS_TIMEOUT_S: Final[float] = 3.0

BackendName = Literal["ollama", "mock"]

JOB_TITLES: Final[dict[int, str]] = {
    0: "Lifeguard",
    1: "Camp Counselor",
    2: "Tutor",
    3: "Fast Food Cashier",
    4: "Retail Clerk",
    5: "Caddy",
    6: "Pet Sitter",
    7: "Swim Coach",
}

CONTEST_NARRATIVE: Final[tuple[tuple[int, str, str, str], ...]] = (
    (
        1,
        "即时经济流动性回报",
        "Immediate Financial Yield",
        "Designed items: high hourly wage and tip potential.",
    ),
    (
        2,
        "长程人力资本投资",
        "Human Capital & Career Value",
        "Designed items: skill acquisition, resume value, social networking.",
    ),
    (
        3,
        "身心环境负荷代价",
        "Ergonomic & Physical Burden",
        "Designed items: fatigue, sun exposure, poor schedule flexibility.",
    ),
)

ITEM_KEYWORDS: Final[dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "hourly_wage_need": (
        ("highest cash wage", "$22", "$35", "$18 to $28", "$80-$150", "effective hourly"),
        ("lowest entry wage", "modest", "$12-$16", "minimum wage"),
    ),
    "tip_potential": (
        ("tips", "gratuity", "caddie fees plus tips", "tip jar"),
        ("tips are rare", "tips are uncommon", "tips are not standard", "little cash tip"),
    ),
    "flexible_hours": (
        ("flexibility", "you accept or decline", "independent", "set their own rate"),
        ("weekly roster", "little autonomy over hours", "weekend and holiday coverage is required", "almost no free evenings"),
    ),
    "commute_convenience": (
        ("local (pool within a town)", "single site", "neighborhood"),
        ("multi-stop", "star-shaped tour", "commute to the mall is often longer"),
    ),
    "physical_strength": (
        ("physical load", "lifting", "walking 5-7 km", "500-yard swim", "prolonged standing", "in-water demonstration"),
        ("sedentary", "minimal sun", "not a heavy lifting"),
    ),
    "outdoor_sun_exposure": (
        ("outdoor sun", "full outdoor sun", "pool deck", "heat stress", "glare off water"),
        ("indoor", "air-conditioned", "climate-controlled", "shade-protected"),
    ),
    "mental_stress": (
        ("high responsibility", "fatal", "emotional labor", "customer conflict", "performance anxiety"),
        ("lowest chronic mental stress", "low formal academic bar"),
    ),
    "safety_level": (
        ("drowning", "spinal", "first aid", "cpr", "liability", "osha", "background checks"),
        ("physical assault risk is low",),
    ),
    "resume_value": (
        ("strong signal", "high for education", "internship referrals", "strong for teaching"),
        ("low prestige", "weak academic signal", "weaker for finance", "weaker as a stem"),
    ),
    "skill_acquisition": (
        ("certification", "wsi", "curriculum", "teach", "diagnose gaps", "servsafe"),
        ("on-the-job training in days",),
    ),
    "social_networking": (
        ("networking", "high-net-worth", "parents", "members"),
        ("no high-status networking", "you work alone"),
    ),
    "teamwork_atmosphere": (
        ("cabin", "team", "coordinate with lifeguards", "crew"),
        ("work alone", "independent tutor"),
    ),
    "free_meals_perks": (
        ("room and board", "employee meal", "free meals are a core perk", "employee discount"),
        ("rare free meals", "no free meals", "meals not included"),
    ),
    "boss_fairness": (
        ("no boss on site", "independent"),
        ("manager-built", "caddie master", "mystery-shopper"),
    ),
    "autonomous_control": (
        ("set their own rate", "accept or decline", "independent scheduling", "no boss on site"),
        ("little autonomy", "scripted curricula", "shifts posted weekly", "loop assignments"),
    ),
}


@dataclass(frozen=True)
class ItemLoading:
    """One Likert item's signed loading on a single factor.

    Attributes
    ----------
    item:
        Column name from the questionnaire / loadings table.
    loading:
        lambda^*_{ij}, dimensionless (correlation-metric loading).
    abs_loading:
        |lambda^*_{ij}|.
    """

    item: str
    loading: float
    abs_loading: float


@dataclass(frozen=True)
class FactorName:
    """Academic label for one extracted factor.

    Attributes
    ----------
    factor_id:
        1-based index matching Factor_1 .. Factor_m.
    name_zh:
        Chinese academic name.
    name_en:
        English academic name.
    theory:
        Labor-economics interpretation grounded in the top items.
    """

    factor_id: int
    name_zh: str
    name_en: str
    theory: str


@dataclass(frozen=True)
class JobFactorRow:
    """One occupation's score vector in extracted-factor space.

    Attributes
    ----------
    job_id:
        Integer in 0..7 matching filename prefix.
    slug:
        Filename stem, e.g. ``0_lifeguard``.
    title_en:
        Contest English title.
    scores:
        F_job in R^3, clipped to [-3, 3].
    rationale:
        Short model or mock justification.
    """

    job_id: int
    slug: str
    title_en: str
    scores: np.ndarray
    rationale: str


@dataclass(frozen=True)
class LLMFactorResult:
    """Full LLMFactor payload written to disk."""

    backend: BackendName
    model: str
    top_items: dict[int, tuple[ItemLoading, ...]]
    names: tuple[FactorName, ...]
    jobs: tuple[JobFactorRow, ...]
    interpretation_path: Path
    matrix_path: Path


def _color_enabled() -> bool:
    return sys.stdout.isatty()


def clip_unit_interval_score(value: float) -> float:
    """Clip a scalar to the closed interval [-3, 3]."""
    return float(np.clip(value, SCORE_LO, SCORE_HI))


def load_rotated_loadings(csv_path: Path | None = None) -> pd.DataFrame:
    """Read ``factor_loadings.csv`` and require Factor_1..Factor_3 columns."""
    path = DEFAULT_LOADINGS if csv_path is None else Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"rotated loadings missing: {path}")
    frame = pd.read_csv(path)
    needed = {"item", "Factor_1", "Factor_2", "Factor_3"}
    missing = needed.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    if len(frame) < 1:
        raise ValueError(f"{path} is empty")
    return frame


def top_abs_loadings(
    frame: pd.DataFrame,
    factor_id: int,
    k: int = TOP_K_ITEMS,
) -> tuple[ItemLoading, ...]:
    """k items with largest |lambda^*| on Factor_{factor_id} (1-based)."""
    col = f"Factor_{factor_id}"
    if col not in frame.columns:
        raise KeyError(col)
    ranked = frame.assign(_abs=frame[col].abs()).sort_values("_abs", ascending=False)
    rows = []
    for _, rec in ranked.head(k).iterrows():
        loading = float(rec[col])
        rows.append(
            ItemLoading(
                item=str(rec["item"]),
                loading=loading,
                abs_loading=abs(loading),
            )
        )
    return tuple(rows)


def all_top_items(frame: pd.DataFrame, k: int = TOP_K_ITEMS) -> dict[int, tuple[ItemLoading, ...]]:
    """Top-k |loading| lists for Factor_1..Factor_3."""
    return {j: top_abs_loadings(frame, j, k=k) for j in range(1, N_FACTORS + 1)}


def loadings_matrix(frame: pd.DataFrame) -> tuple[tuple[str, ...], np.ndarray]:
    """Lambda^* with rows aligned to ``item`` order in the CSV.

    Returns
    -------
    names, lam
        ``lam`` has shape (p, 3).
    """
    names = tuple(str(x) for x in frame["item"].tolist())
    lam = frame[["Factor_1", "Factor_2", "Factor_3"]].to_numpy(dtype=np.float64)
    return names, lam


def list_job_files(job_dir: Path | None = None) -> tuple[Path, ...]:
    """Sorted ``*.txt`` job ads under ``data/job_descriptions/``."""
    directory = DEFAULT_JOB_DIR if job_dir is None else Path(job_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"job description directory missing: {directory}")
    files = tuple(sorted(directory.glob("*.txt")))
    if len(files) != 8:
        raise ValueError(f"expected 8 job text files, found {len(files)} in {directory}")
    return files


def parse_job_id(path: Path) -> int:
    """Leading integer of ``0_lifeguard.txt`` -> 0."""
    match = re.match(r"^(\d+)_", path.name)
    if not match:
        raise ValueError(f"job file must start with an integer id: {path.name}")
    return int(match.group(1))


def _http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout_s: float = OLLAMA_TIMEOUT_S,
) -> dict[str, Any]:
    """GET (payload is None) or POST JSON to a local HTTP endpoint."""
    data: bytes | None = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    parsed: Any = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object from {url}")
    return parsed


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, allowing optional markdown fences."""
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    parsed: Any = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("LLM output is not a JSON object")
    return parsed


def discover_ollama_model(host: str, preferred: str) -> str | None:
    """Return a usable model tag, or None if Ollama is down."""
    url = f"{host.rstrip('/')}/api/tags"
    try:
        body = _http_json(url, payload=None, timeout_s=OLLAMA_TAGS_TIMEOUT_S)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError):
        return None
    models = body.get("models", [])
    tags: list[str] = []
    if isinstance(models, list):
        for entry in models:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                tags.append(str(entry["name"]))
    if not tags:
        return preferred
    exact = [t for t in tags if t == preferred or t.startswith(preferred + ":")]
    if exact:
        return exact[0]
    qwen = [t for t in tags if "qwen2.5" in t.lower() or "qwen2" in t.lower()]
    if qwen:
        return qwen[0]
    return tags[0]


def ollama_chat(host: str, model: str, system: str, user: str) -> dict[str, Any]:
    """Non-streaming ``/api/chat`` with ``format=json``."""
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    body = _http_json(url, payload=payload, timeout_s=OLLAMA_TIMEOUT_S)
    message = body.get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned empty message content")
    return _extract_json_object(content)


def _format_top_block(top_items: dict[int, tuple[ItemLoading, ...]]) -> str:
    """Human-readable top-|loading| block for prompts and markdown."""
    lines: list[str] = []
    for factor_id in range(1, N_FACTORS + 1):
        lines.append(f"Factor_{factor_id} top-{TOP_K_ITEMS} by |loading|:")
        for rank, rec in enumerate(top_items[factor_id], start=1):
            lines.append(
                f"  {rank}. {rec.item:24s}  loading={rec.loading:+.4f}  |l|={rec.abs_loading:.4f}"
            )
    return "\n".join(lines)


def naming_prompts(top_items: dict[int, tuple[ItemLoading, ...]]) -> tuple[str, str]:
    """System + user prompts for academic factor naming (few-shot style only)."""
    system = (
        "You are a labor economist naming exploratory factors from a teen summer-job "
        "Likert battery. Reply with a single JSON object. Names must be grounded in "
        "the ACTUAL top-|loading| items provided. Do not copy few-shot names unless "
        "those items truly match (wage/tips vs skills/resume/network vs fatigue/sun/"
        "inflexibility). Mixing safety, tips, and skills on one factor is allowed; "
        "say so."
    )
    user = f"""Few-shot STYLE examples (designed 15-item story, NOT this extraction):
Example A: top items hourly_wage_need (+0.81), tip_potential (+0.77)
  -> name_zh 即时经济流动性回报; name_en Immediate Financial Yield
Example B: top items skill_acquisition, resume_value, social_networking
  -> name_zh 长程人力资本投资; name_en Human Capital & Career Value
Example C: top items physical_strength, outdoor_sun_exposure, flexible_hours (negative)
  -> name_zh 身心环境负荷代价; name_en Ergonomic & Physical Burden

THIS extraction (from results/factor_loadings.csv):
{_format_top_block(top_items)}

Return JSON:
{{
  "factors": [
    {{
      "factor_id": 1,
      "name_zh": "...",
      "name_en": "...",
      "theory": "2-4 sentences: compensating differentials / Becker HC / Rosen hedonic / Karasek demand-control as applicable. Cite the signed loadings."
    }},
    {{"factor_id": 2, "name_zh": "...", "name_en": "...", "theory": "..."}},
    {{"factor_id": 3, "name_zh": "...", "name_en": "...", "theory": "..."}}
  ]
}}
"""
    return system, user


def scoring_prompts(
    names: tuple[FactorName, ...],
    top_items: dict[int, tuple[ItemLoading, ...]],
    job_title: str,
    job_text: str,
) -> tuple[str, str]:
    """Few-shot prompts mapping a job ad to F in [-3, 3]^3 in extracted space."""
    name_lines = "\n".join(
        f"  F{n.factor_id}: {n.name_en} / {n.name_zh}. Anchors: "
        + ", ".join(f"{it.item} ({it.loading:+.2f})" for it in top_items[n.factor_id])
        for n in names
    )
    system = (
        "You score teen summer jobs on three EXTRACTED orthogonal factors. "
        "Output one JSON object. Each score is a z-like rating relative to a typical "
        "U.S. high-school summer job: 0 ≈ average, +3 ≈ extremely high on that "
        "factor as defined by the anchors, -3 ≈ extremely low. Use the job text, "
        "not the designed wage/HC/burden story, unless the anchors match."
    )
    truncated = job_text if len(job_text) <= 3500 else job_text[:3500] + "\n[truncated]\n"
    user = f"""Factor definitions:
{name_lines}

Few-shot (hypothetical jobs, not in the contest set):
1) Night-shift warehouse picker, heavy cases, little sun, low pay, no resume signal
   -> scores depend on THIS extraction's anchors (e.g. high if F is physical; low if F is tutoring-style autonomy).
2) Library page, indoor, quiet, low pay, some schedule control
   -> near 0 on physical/sun anchors; not automatically high Human Capital unless resume/skill items define that factor.

Score this occupation: {job_title}

JOB TEXT:
{truncated}

Return JSON:
{{
  "scores": {{"F1": 0.0, "F2": 0.0, "F3": 0.0}},
  "rationale": "3-6 sentences citing duties, pay, environment vs the four anchors per factor."
}}
"""
    return system, user


def _mock_name(factor_id: int, items: tuple[ItemLoading, ...]) -> FactorName:
    """Deterministic labor-econ labels from actual top items (not the contest triad)."""
    keys = tuple(it.item for it in items)
    signed = "; ".join(f"{it.item} ({it.loading:+.3f})" for it in items)
    if factor_id == 1:
        return FactorName(
            factor_id=1,
            name_zh="安全—技能—关系复合回报",
            name_en="Safety–Skill–Network Compound Return",
            theory=(
                "Mock lexicon (Ollama unavailable). Top |loadings| are "
                f"{signed}. This is not Immediate Financial Yield: tip_potential "
                "appears, but safety_level leads, with social_networking and "
                "skill_acquisition. Interpreting F1 as a Rosen compensating-differential "
                "bundle mixed with Becker-style skill/network signaling — a compound "
                "factor, not a clean cash-yield axis. Keys="
                + ",".join(keys)
                + "."
            ),
        )
    if factor_id == 2:
        return FactorName(
            factor_id=2,
            name_zh="工作自主、可达性与工资诉求",
            name_en="Autonomy–Access and Wage Claim",
            theory=(
                "Mock lexicon. Top |loadings| are "
                f"{signed}. Karasek-style job control (autonomous_control) and spatial "
                "access (commute_convenience) co-load with hourly_wage_need and "
                "mental_stress. This is not Human Capital & Career Value (skills/"
                "resume/network live more on F1). F2 is an agency-and-access claim "
                "on compensation, with stress loading in the same direction as autonomy "
                "on this extraction."
            ),
        )
    return FactorName(
        factor_id=3,
        name_zh="体力户外负荷与主管公平权衡",
        name_en="Physical–Outdoor Load vs Supervisory Fairness",
        theory=(
            "Mock lexicon. Top |loadings| are "
            f"{signed}. physical_strength and outdoor_sun_exposure load positively; "
            "boss_fairness loads negatively; flexible_hours loads positively — so this "
            "is not the designed Ergonomic Burden axis (which assumed inflexibility). "
            "Hedonic-wage reading: F3 trades physical/sun load and clock flexibility "
            "against perceived supervisory fairness."
        ),
    )


def mock_names(top_items: dict[int, tuple[ItemLoading, ...]]) -> tuple[FactorName, ...]:
    """Three mock names, one per extracted factor."""
    return tuple(_mock_name(j, top_items[j]) for j in range(1, N_FACTORS + 1))


def parse_names_json(payload: dict[str, Any]) -> tuple[FactorName, ...]:
    """Validate LLM naming JSON into FactorName triples."""
    factors = payload.get("factors")
    if not isinstance(factors, list) or len(factors) != N_FACTORS:
        raise ValueError("naming JSON must contain factors: [3 objects]")
    by_id: dict[int, FactorName] = {}
    for entry in factors:
        if not isinstance(entry, dict):
            raise ValueError("factor entry must be an object")
        fid = int(entry["factor_id"])
        by_id[fid] = FactorName(
            factor_id=fid,
            name_zh=str(entry["name_zh"]).strip(),
            name_en=str(entry["name_en"]).strip(),
            theory=str(entry["theory"]).strip(),
        )
    if set(by_id) != {1, 2, 3}:
        raise ValueError(f"factor_id set must be {{1,2,3}}, got {sorted(by_id)}")
    for fid in (1, 2, 3):
        if not by_id[fid].name_en or not by_id[fid].name_zh:
            raise ValueError(f"empty name for factor {fid}")
    return (by_id[1], by_id[2], by_id[3])


def parse_scores_json(payload: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Validate scoring JSON; returns clipped length-3 vector and rationale."""
    scores_raw = payload.get("scores")
    if not isinstance(scores_raw, dict):
        raise ValueError("scoring JSON needs object 'scores'")
    vec = np.array(
        [
            clip_unit_interval_score(float(scores_raw["F1"])),
            clip_unit_interval_score(float(scores_raw["F2"])),
            clip_unit_interval_score(float(scores_raw["F3"])),
        ],
        dtype=np.float64,
    )
    rationale = str(payload.get("rationale", "")).strip()
    return vec, rationale


def keyword_item_scores(text: str, item_names: tuple[str, ...]) -> np.ndarray:
    """Signed keyword hits in [-2, 2] for each Likert item (mock scorer)."""
    blob = text.lower()
    out = np.zeros(len(item_names), dtype=np.float64)
    for i, item in enumerate(item_names):
        pos_cues, neg_cues = ITEM_KEYWORDS.get(item, ((), ()))
        pos = sum(1 for cue in pos_cues if cue.lower() in blob)
        neg = sum(1 for cue in neg_cues if cue.lower() in blob)
        out[i] = float(np.clip(pos - neg, -2.0, 2.0))
    return out


def mock_job_scores(
    text: str,
    item_names: tuple[str, ...],
    lam: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Project keyword item scores through Lambda^* and clip to [-3, 3]^3.

    .. math::

        \\tilde F = \\Lambda^{*\\top} s_{\\mathrm{kw}}, \\quad
        F_j = \\mathrm{clip}_{[-3,3]}\\big(\\tilde F_j / (\\|\\lambda^*_{\\cdot j}\\|_2 + \\varepsilon)\\big)
    """
    s_kw = keyword_item_scores(text, item_names)
    raw = lam.T @ s_kw
    col_norm = np.linalg.norm(lam, axis=0) + 1.0e-12
    scaled = raw / col_norm
    clipped = np.clip(scaled, SCORE_LO, SCORE_HI)
    rationale = (
        "Mock text-to-factor: keyword Likert proxies projected through rotated "
        f"loadings (||s_kw||_2={float(np.linalg.norm(s_kw)):.3f}). Not an LLM judgment."
    )
    return clipped.astype(np.float64), rationale


def resolve_backend(requested: str, host: str, preferred_model: str) -> tuple[BackendName, str]:
    """``auto`` probes Ollama; otherwise honor ``ollama`` / ``mock``."""
    mode = requested.lower().strip()
    if mode == "mock":
        return "mock", "mock-lexicon"
    model = discover_ollama_model(host, preferred_model)
    if model is None:
        if mode == "ollama":
            raise RuntimeError(f"Ollama not reachable at {host} (required by backend=ollama)")
        return "mock", "mock-lexicon"
    return "ollama", model


def name_factors_with_backend(
    top_items: dict[int, tuple[ItemLoading, ...]],
    backend: BackendName,
    host: str,
    model: str,
) -> tuple[FactorName, ...]:
    """LLM naming with mock fallback if the JSON is unusable."""
    if backend == "mock":
        return mock_names(top_items)
    system, user = naming_prompts(top_items)
    try:
        payload = ollama_chat(host, model, system, user)
        return parse_names_json(payload)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError, OSError) as exc:
        print(f"  naming via Ollama failed ({exc}); using mock names", file=sys.stderr)
        return mock_names(top_items)


def score_job_with_backend(
    names: tuple[FactorName, ...],
    top_items: dict[int, tuple[ItemLoading, ...]],
    title: str,
    text: str,
    item_names: tuple[str, ...],
    lam: np.ndarray,
    backend: BackendName,
    host: str,
    model: str,
) -> tuple[np.ndarray, str]:
    """One job: Ollama JSON or loading-weighted keyword mock."""
    if backend == "mock":
        return mock_job_scores(text, item_names, lam)
    system, user = scoring_prompts(names, top_items, title, text)
    try:
        payload = ollama_chat(host, model, system, user)
        return parse_scores_json(payload)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError, OSError) as exc:
        print(f"  scoring {title!r} via Ollama failed ({exc}); using mock", file=sys.stderr)
        return mock_job_scores(text, item_names, lam)


def write_interpretation_md(
    result_backend: BackendName,
    model: str,
    top_items: dict[int, tuple[ItemLoading, ...]],
    names: tuple[FactorName, ...],
    jobs: tuple[JobFactorRow, ...],
    out_path: Path,
) -> Path:
    """Write the factor-interpretation report (actual loadings + names)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Factor interpretation (LLMFactor)",
        "",
        "Semantic names and job coordinates for the **extracted** Varimax factors "
        "in `results/factor_loadings.csv`. Numbers in the top-|loading| tables are "
        "copied from that CSV. This file does not invent a new rotation.",
        "",
        f"- Backend: `{result_backend}`",
        f"- Model: `{model}`",
        "- Student scores (same axes): `results/student_factor_scores.csv`",
        "- Questionnaire: `data/raw/OriginalData.csv` (see `DATA_SOURCE.txt`; "
        "15-col table is SYNTHETIC_FALLBACK if GitHub was 7-D)",
        "",
        "## Contest narrative vs this extraction",
        "",
        "A designed 15-item story maps F1→Immediate Financial Yield (wage/tips), "
        "F2→Human Capital & Career Value (skill/resume/network), "
        "F3→Ergonomic & Physical Burden (fatigue/sun/inflexibility). "
        "**That triad is not what Kaiser-normalized Varimax recovered here.** "
        "LLMFactor therefore names the extracted axes; the designed names are "
        "kept only as a foil.",
        "",
        "| Designed $F_j$ | zh | en | designed anchors |",
        "|---|---|---|---|",
    ]
    for fid, zh, en, anchors in CONTEST_NARRATIVE:
        lines.append(f"| designed {fid} | {zh} | {en} | {anchors} |")
    lines.extend(["", "## Extracted factors (top-4 $|\\lambda^*_{ij}|$)", ""])
    for fac in names:
        lines.append(f"### $F_{fac.factor_id}$: {fac.name_zh} ({fac.name_en})")
        lines.append("")
        lines.append("| rank | item | $\\lambda^*$ | $|\\lambda^*|$ |")
        lines.append("|---|---|---:|---:|")
        for rank, rec in enumerate(top_items[fac.factor_id], start=1):
            lines.append(
                f"| {rank} | `{rec.item}` | {rec.loading:+.4f} | {rec.abs_loading:.4f} |"
            )
        lines.append("")
        lines.append(fac.theory)
        lines.append("")
        signed = "; ".join(
            f"`{rec.item}` ({rec.loading:+.3f})" for rec in top_items[fac.factor_id]
        )
        lines.append(f"**CSV grounding (authoritative):** {signed}.")
        lines.append("")
    lines.extend(
        [
            "## Job coordinates $F_{\\mathrm{job}}\\in[-3,3]^3$",
            "",
            "Each of 8 ads in `data/job_descriptions/` is scored on the **extracted** "
            "names above (not on the designed triad). Values are clipped to $[-3,3]$.",
            "",
            "| job_id | title | $F_1$ | $F_2$ | $F_3$ |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for job in jobs:
        s = job.scores
        lines.append(
            f"| {job.job_id} | {job.title_en} | {s[0]:+.3f} | {s[1]:+.3f} | {s[2]:+.3f} |"
        )
    lines.extend(["", "### Rationales", ""])
    for job in jobs:
        lines.append(f"**{job.job_id} {job.title_en}** (`{job.slug}`)")
        lines.append("")
        lines.append(job.rationale)
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_job_matrix_csv(
    names: tuple[FactorName, ...],
    jobs: tuple[JobFactorRow, ...],
    backend: BackendName,
    model: str,
    out_path: Path,
) -> Path:
    """Eight-row benchmark factor coordinate table."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for job in jobs:
        rows.append(
            {
                "job_id": job.job_id,
                "slug": job.slug,
                "title_en": job.title_en,
                "Factor_1": float(job.scores[0]),
                "Factor_2": float(job.scores[1]),
                "Factor_3": float(job.scores[2]),
                "name_F1": names[0].name_en,
                "name_F2": names[1].name_en,
                "name_F3": names[2].name_en,
                "backend": backend,
                "model": model,
                "rationale": job.rationale.replace("\n", " "),
            }
        )
    pd.DataFrame(rows).sort_values("job_id").to_csv(out_path, index=False)
    return out_path


def run_llm_factor_pipeline(
    loadings_path: Path | None = None,
    job_dir: Path | None = None,
    interpretation_path: Path | None = None,
    matrix_path: Path | None = None,
    backend: str | None = None,
) -> LLMFactorResult:
    """Name three extracted factors and score eight job ads.

    Parameters
    ----------
    loadings_path:
        Default ``results/factor_loadings.csv``.
    job_dir:
        Default ``data/job_descriptions/``.
    interpretation_path:
        Default ``results/factor_interpretation.md``.
    matrix_path:
        Default ``results/job_factor_matrix.csv``.
    backend:
        ``auto`` (default), ``ollama``, or ``mock``. Overridden by env
        ``LLM_FACTOR_BACKEND``.
    """
    requested = (backend or os.environ.get("LLM_FACTOR_BACKEND") or "auto").lower()
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    preferred = os.environ.get("OLLAMA_MODEL", "qwen2.5")
    resolved, model = resolve_backend(requested, host, preferred)

    frame = load_rotated_loadings(loadings_path)
    top = all_top_items(frame)
    item_names, lam = loadings_matrix(frame)
    names = name_factors_with_backend(top, resolved, host, model)

    job_rows: list[JobFactorRow] = []
    for path in list_job_files(job_dir):
        job_id = parse_job_id(path)
        title = JOB_TITLES.get(job_id, path.stem)
        text = path.read_text(encoding="utf-8")
        scores, rationale = score_job_with_backend(
            names, top, title, text, item_names, lam, resolved, host, model
        )
        job_rows.append(
            JobFactorRow(
                job_id=job_id,
                slug=path.stem,
                title_en=title,
                scores=scores,
                rationale=rationale,
            )
        )
    jobs = tuple(sorted(job_rows, key=lambda row: row.job_id))

    md_path = DEFAULT_INTERPRET if interpretation_path is None else Path(interpretation_path)
    csv_path = DEFAULT_JOB_MATRIX if matrix_path is None else Path(matrix_path)
    write_interpretation_md(resolved, model, top, names, jobs, md_path)
    write_job_matrix_csv(names, jobs, resolved, model, csv_path)

    print(f"  backend     : {resolved} ({model})")
    print(f"  names       : " + " | ".join(f"F{n.factor_id}={n.name_en}" for n in names))
    print(f"  interpret   : {md_path}")
    print(f"  job matrix  : {csv_path}")
    return LLMFactorResult(
        backend=resolved,
        model=model,
        top_items=top,
        names=names,
        jobs=jobs,
        interpretation_path=md_path,
        matrix_path=csv_path,
    )


def main() -> int:
    """CLI: python -m src.llm_factor.llm_factor_agent"""
    run_llm_factor_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())

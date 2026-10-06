"""Load pack artifacts and score a 1-10 slider profile against 8 jobs.

The Streamlit labels follow the contest triad (yield / career / strain
tolerance). The numeric vector is mapped into the extracted Thompson space
that the Stage-6 Softmax was trained on. Slider 3 is tolerance: high values
map to high extracted F3 (burden), matching the Pragmatist / Leisure presets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.classification.dual_recommender import (
    FACTOR_COLUMNS,
    N_CLASSES,
    align_features_and_labels,
    fit_linear_softmax,
    softmax_logits,
    stable_softmax,
)
from src.entropy_weight.entropy_engine import PACK_ROOT, load_job_factor_matrix
from src.llm_factor.llm_factor_agent import JOB_TITLES

JOB_DIR: Final[Path] = PACK_ROOT / "data" / "job_descriptions"
WEIGHTS_CSV: Final[Path] = PACK_ROOT / "results" / "entropy_weights.csv"
DOLLAR_SPAN: Final[re.Pattern[str]] = re.compile(
    r"\$\s*[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*(?:to|-|–)\s*\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?)?"
)


@dataclass(frozen=True)
class JobCard:
    """One occupation card for the navigator UI.

    Attributes
    ----------
    job_id:
        Class index 0..7.
    title_en:
        English job title.
    slug:
        Description filename stem.
    proba:
        Softmax P(y=k | x), in [0, 1].
    match_pct:
        100 * proba.
    wage_range:
        First dollar span parsed from the job ad, or a fallback note.
    reason:
        Short match / conflict explanation.
    factors:
        Job coordinates in extracted F-space, length 3.
    """

    job_id: int
    title_en: str
    slug: str
    proba: float
    match_pct: float
    wage_range: str
    reason: str
    factors: np.ndarray


@dataclass(frozen=True)
class NavigatorBundle:
    """Cached artifacts for one Streamlit session."""

    weight: np.ndarray
    bias: np.ndarray
    col_min: np.ndarray
    col_max: np.ndarray
    job_ids: np.ndarray
    job_factors: np.ndarray
    job_titles: tuple[str, ...]
    extracted_names: tuple[str, ...]
    wages: tuple[str, ...]
    rationales: tuple[str, ...]
    slugs: tuple[str, ...]


def parse_wage_range(text: str) -> str:
    """Return the first '$a to $b' / '$a-$b' span in a job ad."""
    match = DOLLAR_SPAN.search(text)
    if match is None:
        return "See job ad (no $ span parsed)"
    return re.sub(r"\s+", " ", match.group(0)).strip()


def load_wage_table(job_dir: Path = JOB_DIR) -> dict[int, str]:
    """Map job_id -> wage span from ``data/job_descriptions/*.txt``."""
    out: dict[int, str] = {}
    for path in sorted(job_dir.glob("*.txt")):
        stem = path.stem
        job_id = int(stem.split("_", 1)[0])
        out[job_id] = parse_wage_range(path.read_text(encoding="utf-8"))
    return out


def load_extracted_names(path: Path = WEIGHTS_CSV) -> tuple[str, ...]:
    """English extracted-axis names from entropy_weights.csv."""
    if not path.is_file():
        return tuple(FACTOR_COLUMNS)
    frame = pd.read_csv(path)
    if "name_en" not in frame.columns or len(frame) < 3:
        return tuple(FACTOR_COLUMNS)
    return tuple(str(v) for v in frame["name_en"].tolist()[:3])


def slider_to_factor(value: float, col_min: float, col_max: float) -> float:
    """Map a 1..10 slider onto [col_min, col_max] (student Thompson range).

    .. math::

        f = f_{\\min} + \\frac{s-1}{9}(f_{\\max}-f_{\\min})
    """
    if not 1.0 <= value <= 10.0:
        raise ValueError(f"slider must be in [1, 10], got {value}")
    t = (float(value) - 1.0) / 9.0
    return float(col_min + t * (col_max - col_min))


def factor_to_radar(value: float, col_min: float, col_max: float) -> float:
    """Map a factor score back onto the 1..10 radar scale."""
    span = col_max - col_min
    if span <= 0.0:
        return 5.5
    t = (float(value) - col_min) / span
    return float(1.0 + 9.0 * np.clip(t, 0.0, 1.0))


def profile_to_x(
    s1: float,
    s2: float,
    s3: float,
    col_min: np.ndarray,
    col_max: np.ndarray,
) -> np.ndarray:
    """Three sliders -> shape (1, 3) Thompson vector."""
    x = np.array(
        [
            slider_to_factor(s1, float(col_min[0]), float(col_max[0])),
            slider_to_factor(s2, float(col_min[1]), float(col_max[1])),
            slider_to_factor(s3, float(col_min[2]), float(col_max[2])),
        ],
        dtype=np.float64,
    )
    return x.reshape(1, 3)


def load_navigator_bundle() -> NavigatorBundle:
    """Fit full-sample Softmax once and attach job ads / wages."""
    x, y, _sid_s, _sid_c = align_features_and_labels()
    col_min = x.min(axis=0)
    col_max = x.max(axis=0)
    weight, bias = fit_linear_softmax(x, y, seed=2020)
    job_ids, job_factors, titles, _axis = load_job_factor_matrix()
    wages = load_wage_table()
    frame = pd.read_csv(PACK_ROOT / "results" / "job_factor_matrix.csv").sort_values("job_id")
    rationales = tuple(str(v) for v in frame["rationale"].tolist())
    slugs = tuple(str(v) for v in frame["slug"].tolist())
    wage_tuple = tuple(wages.get(int(j), "See job ad") for j in job_ids)
    return NavigatorBundle(
        weight=weight,
        bias=bias,
        col_min=col_min,
        col_max=col_max,
        job_ids=job_ids,
        job_factors=job_factors,
        job_titles=titles,
        extracted_names=load_extracted_names(),
        wages=wage_tuple,
        rationales=rationales,
        slugs=slugs,
    )


def _axis_conflicts(
    user_x: np.ndarray,
    job_f: np.ndarray,
    names: tuple[str, ...],
) -> list[str]:
    """Largest coordinate gaps between the user vector and a job."""
    delta = job_f - user_x.reshape(-1)
    order = np.argsort(-np.abs(delta))
    lines = []
    for j in order[:3]:
        lines.append(
            f"{names[j]}: you {user_x.reshape(-1)[j]:+.2f} vs job {job_f[j]:+.2f} "
            f"(gap {delta[j]:+.2f})"
        )
    return lines


def match_reason(
    bundle: NavigatorBundle,
    job_id: int,
    user_x: np.ndarray,
    kind: str,
) -> str:
    """One-sentence reason using extracted axes, not the slider nicknames."""
    fvec = bundle.job_factors[job_id]
    gaps = _axis_conflicts(user_x, fvec, bundle.extracted_names)
    snippet = bundle.rationales[job_id].split(".")[0].strip()
    if kind == "avoid":
        return (
            "Largest mismatches on the extracted axes: "
            + "; ".join(gaps)
            + f". Ad note: {snippet}."
        )
    return (
        "Closest on extracted axes: "
        + "; ".join(gaps)
        + f". Ad note: {snippet}."
    )


def rank_jobs(bundle: NavigatorBundle, s1: float, s2: float, s3: float) -> tuple[JobCard, ...]:
    """Softmax-rank all 8 jobs for a slider triple."""
    user_x = profile_to_x(s1, s2, s3, bundle.col_min, bundle.col_max)
    proba = stable_softmax(softmax_logits(user_x, bundle.weight, bundle.bias))[0]
    cards: list[JobCard] = []
    for k in range(N_CLASSES):
        kind = "match"
        cards.append(
            JobCard(
                job_id=k,
                title_en=bundle.job_titles[k] if k < len(bundle.job_titles) else JOB_TITLES.get(k, f"job_{k}"),
                slug=bundle.slugs[k],
                proba=float(proba[k]),
                match_pct=float(100.0 * proba[k]),
                wage_range=bundle.wages[k],
                reason=match_reason(bundle, k, user_x, kind),
                factors=bundle.job_factors[k].copy(),
            )
        )
    ranked = tuple(sorted(cards, key=lambda c: c.proba, reverse=True))
    avoid_id = ranked[-1].job_id
    patched: list[JobCard] = []
    for card in ranked:
        if card.job_id == avoid_id:
            patched.append(
                JobCard(
                    job_id=card.job_id,
                    title_en=card.title_en,
                    slug=card.slug,
                    proba=card.proba,
                    match_pct=card.match_pct,
                    wage_range=card.wage_range,
                    reason=match_reason(bundle, card.job_id, user_x, "avoid"),
                    factors=card.factors,
                )
            )
        else:
            patched.append(card)
    return tuple(patched)


def radar_series(
    bundle: NavigatorBundle,
    s1: float,
    s2: float,
    s3: float,
    job_factors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """User sliders and one job, both on the 1..10 radar scale."""
    user = np.array([s1, s2, s3], dtype=np.float64)
    job = np.array(
        [
            factor_to_radar(float(job_factors[j]), float(bundle.col_min[j]), float(bundle.col_max[j]))
            for j in range(3)
        ],
        dtype=np.float64,
    )
    return user, job

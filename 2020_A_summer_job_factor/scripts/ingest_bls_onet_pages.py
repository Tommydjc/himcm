#!/usr/bin/env python3
"""Parse Firecrawl BLS OOH / O*NET pages into numeric tables and a correlation matrix.

Does not overwrite the 15-column student Likert file. Job-side correlations come
from occupations named in 2020 HiMCM Problem A (cashier, lifeguard, wait staff,
data analysis, office work, research, virtual/electronic work).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

PACK_BOOT = Path(__file__).resolve().parents[1]
if str(PACK_BOOT) not in sys.path:
    sys.path.insert(0, str(PACK_BOOT))

from src.factor_analysis.suitability import evaluate_suitability, plot_correlation_heatmap

PACK_ROOT: Final[Path] = PACK_BOOT
PAGE_DIR: Final[Path] = PACK_ROOT / "data" / "raw" / "bls_onet_pages"
RAW: Final[Path] = PACK_ROOT / "data" / "raw"
FIG: Final[Path] = PACK_ROOT / "paper_figures"
RESULTS: Final[Path] = PACK_ROOT / "results"

EDU_RANK: Final[dict[str, int]] = {
    "no formal educational credential": 1,
    "no formal education": 1,
    "high school diploma or equivalent": 2,
    "some college": 3,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5,
    "doctoral or professional degree": 6,
}


def _first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.I)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def parse_bls_quickfacts(path: Path) -> dict[str, float | str | None]:
    """Extract OOH Quick Facts numerics from a Firecrawl markdown file."""
    text = path.read_text(encoding="utf-8")
    hourly = _first_float(r"\$([0-9]+(?:\.[0-9]+)?)\s+per hour", text)
    annual = _first_float(r"\$([0-9,]+)\s+per year", text)
    outlook = _first_float(
        r"Job Outlook[^\n]*\|[^\n]*?([+\-]?\d+)\s*%",
        text,
    )
    if outlook is None:
        outlook = _first_float(
            r"projected to (?:grow|decline|increase|decrease)\s+([+\-]?\d+)\s+percent",
            text,
        )
        if outlook is not None and re.search(r"projected to decline", text, flags=re.I):
            outlook = -abs(outlook)
    n_jobs = _first_float(r"Number of Jobs[^\n]*\|[^\n]*?([0-9,]{4,})", text)
    edu_match = re.search(
        r"Typical Entry-Level Education.*?\|\s*([^|\n]+)",
        text,
        flags=re.I | re.S,
    )
    edu_raw = edu_match.group(1).strip().split("<")[0].strip() if edu_match else ""
    edu_rank = None
    low = edu_raw.lower()
    for key, val in EDU_RANK.items():
        if key in low:
            edu_rank = val
            break
    standing = 1.0 if re.search(r"\bstand", text, flags=re.I) else 0.0
    outdoor = 1.0 if re.search(r"\boutdoor|\boutside\b|\bsun\b", text, flags=re.I) else 0.0
    sedentary = 1.0 if re.search(r"\bsit|\boffice\b|\bcomputer\b", text, flags=re.I) else 0.0
    return {
        "occupation": path.stem,
        "median_hourly_usd": hourly,
        "median_annual_usd": annual,
        "outlook_pct": outlook,
        "n_jobs_2025": n_jobs,
        "education_rank": edu_rank,
        "env_standing_mention": standing,
        "env_outdoor_mention": outdoor,
        "env_sedentary_mention": sedentary,
        "education_raw": edu_raw[:80],
    }


def parse_onet_work_context(path: Path) -> dict[str, float]:
    """Parse '**Context name** — 76% responded' bullets from an O*NET summary."""
    text = path.read_text(encoding="utf-8")
    start = text.find("## Work Context")
    if start < 0:
        return {}
    end = text.find("\n## ", start + 5)
    block = text[start:end] if end > 0 else text[start:]
    out: dict[str, float] = {}
    for match in re.finditer(
        r"\*\*([^*]+)\*\*\s+[—-]\s+(\d+)\s*%\s+responded",
        block,
    ):
        name = re.sub(r"\s+", " ", match.group(1)).strip()
        out[name] = float(match.group(2))
    return out


def main() -> int:
    """Write BLS and O*NET CSVs plus job-side correlation heatmap."""
    bls_rows: list[dict[str, float | str | None]] = []
    for path in sorted(PAGE_DIR.glob("*.md")):
        if path.name.startswith("onet_") or path.name == "manifest.json":
            continue
        bls_rows.append(parse_bls_quickfacts(path))
    bls_frame = pd.DataFrame(bls_rows)
    bls_path = RAW / "bls_ooh_quickfacts.csv"
    bls_frame.to_csv(bls_path, index=False)

    onet_maps: dict[str, dict[str, float]] = {}
    for path in sorted(PAGE_DIR.glob("onet_*.md")):
        if path.stem == "onet_work_context_db":
            continue
        ctx = parse_onet_work_context(path)
        if ctx:
            onet_maps[path.stem.replace("onet_", "")] = ctx
    onet_frame = pd.DataFrame.from_dict(onet_maps, orient="index")
    onet_frame.index.name = "occupation"
    onet_path = RAW / "onet_work_context_problem_jobs.csv"
    onet_frame.to_csv(onet_path)

    # Correlation on BLS numeric columns (complete cases).
    bls_num_cols = [
        "median_hourly_usd",
        "outlook_pct",
        "education_rank",
        "env_standing_mention",
        "env_outdoor_mention",
    ]
    bls_num = bls_frame[bls_num_cols].apply(pd.to_numeric, errors="coerce")
    bls_complete = bls_num.dropna(axis=0, how="any")
    bls_kmo = float("nan")
    bls_p = float("nan")
    if len(bls_complete) >= 5 and bls_complete.shape[1] >= 3:
        # Drop zero-variance columns before KMO.
        keep = [c for c in bls_complete.columns if float(bls_complete[c].std(ddof=0)) > 1.0e-12]
        x = bls_complete[keep].to_numpy(dtype=np.float64)
        names = tuple(keep)
        report = evaluate_suitability(x, names)
        bls_kmo = report.kmo_overall
        bls_p = report.bartlett_p
        fig_bls = FIG / "fig_correlation_matrix_bls_ooh.png"
        plot_correlation_heatmap(report.corr, names, fig_bls)
        print(f"BLS OOH  N={report.n_obs} p={report.n_var} KMO={bls_kmo:.4f} Bartlett p={bls_p:.3e}")
        print(f"  {fig_bls}")

    # O*NET: keep context items observed for at least 6 occupations; mean-impute rest.
    onet_kmo = float("nan")
    onet_p = float("nan")
    if not onet_frame.empty:
        ranked = onet_frame.notna().sum().sort_values(ascending=False)
        keep_ctx = [c for c in ranked.index.tolist() if int(ranked[c]) >= 11][:6]
        sub = onet_frame[keep_ctx].copy()
        filled = sub.dropna(axis=0, how="any")
        keep2 = [c for c in filled.columns if float(filled[c].std(ddof=0)) > 1.0e-12]
        filled = filled[keep2]
        if len(filled) >= 5 and filled.shape[1] >= 3:
            x = filled.to_numpy(dtype=np.float64)
            names = tuple(str(c)[:40] for c in filled.columns)
            report = evaluate_suitability(x, names)
            onet_kmo = report.kmo_overall
            onet_p = report.bartlett_p
            fig_onet = FIG / "fig_correlation_matrix_onet_work_context.png"
            plot_correlation_heatmap(report.corr, names, fig_onet)
            print(
                f"O*NET context N={report.n_obs} p={report.n_var} "
                f"KMO={onet_kmo:.4f} Bartlett p={onet_p:.3e}"
            )
            print(f"  {fig_onet}")
            filled.to_csv(RAW / "onet_work_context_imputed.csv")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "bls_onet_job_suitability.txt").write_text(
        (
            "pdf=2020_HiMCM_Problem_A.pdf contains zero URLs; occupations were mapped to BLS OOH and O*NET.\n"
            f"bls_rows={len(bls_frame)} file={bls_path}\n"
            f"bls_kmo={bls_kmo} bartlett_p={bls_p}\n"
            f"onet_occupations={len(onet_frame)} file={onet_path}\n"
            f"onet_kmo={onet_kmo} bartlett_p={onet_p}\n"
            "note=These are JOB-side matrices. They do not replace student Likert OriginalData.csv.\n"
        ),
        encoding="utf-8",
    )

    source = RAW / "DATA_SOURCE.txt"
    prev = source.read_text(encoding="utf-8") if source.is_file() else ""
    extra = (
        "\nbls_ooh=data/raw/bls_ooh_quickfacts.csv  Firecrawl of occupations named in Problem A\n"
        "onet_pages=data/raw/bls_onet_pages/  O*NET Work Context % from summary pages\n"
        f"bls_kmo={bls_kmo} onet_kmo={onet_kmo}\n"
        "job_side_vs_student=job-attribute R is not student-item R; KMO of OriginalData.csv is unchanged.\n"
    )
    if "bls_ooh=" not in prev:
        source.write_text(prev.rstrip() + extra, encoding="utf-8")

    man = PAGE_DIR / "ingest_summary.json"
    man.write_text(
        json.dumps(
            {
                "bls_rows": int(len(bls_frame)),
                "onet_occupations": int(len(onet_frame)),
                "bls_kmo": bls_kmo,
                "onet_kmo": onet_kmo,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"wrote {bls_path} ({len(bls_frame)} rows)")
    print(f"wrote {onet_path} ({len(onet_frame)} occupations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

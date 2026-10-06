#!/usr/bin/env python3
"""Ingest the Firecrawl GitHub mirror into typed CSVs. Does not invent 15 Likert items.

The HiMCM2020/HiMCM_2020 survey layout (from Models.py) is:
    N = 50 students, K = 8 jobs, D = 7 unnamed factor scores on a 10-100 scale.
    OriginalData.txt is 400 x 7 = (student, job) rows.

This script never maps those 7 dimensions onto the 15 semantic Likert columns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

PACK_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PACK_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT_BOOTSTRAP))

import numpy as np
import pandas as pd

from src.factor_analysis.suitability import (
    evaluate_suitability,
    plot_correlation_heatmap,
)

PACK_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MIRROR: Final[Path] = PACK_ROOT / "data" / "raw" / "himcm2020_github"
RAW: Final[Path] = PACK_ROOT / "data" / "raw"
FIG_DIR: Final[Path] = PACK_ROOT / "paper_figures"
RESULTS: Final[Path] = PACK_ROOT / "results"

N_STUDENTS: Final[int] = 50
N_JOBS: Final[int] = 8
N_DIMS: Final[int] = 7
DIM_COLS: Final[list[str]] = [f"job_factor_{i}" for i in range(1, N_DIMS + 1)]


def load_numeric(path: Path) -> np.ndarray:
    """Parse whitespace numeric tables saved by Firecrawl markdown."""
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        rows.append([float(tok) for tok in stripped.replace(",", " ").split()])
    return np.asarray(rows, dtype=np.float64)


def main() -> int:
    """Write long 400 x 7 ratings, replace work_choice labels, record KMO."""
    orig_path = MIRROR / "OriginalData.txt"
    choice_path = MIRROR / "work_choice.txt"
    if not orig_path.is_file() or not choice_path.is_file():
        print("Run scripts/fetch_himcm2020_firecrawl.py first.", file=sys.stderr)
        return 2

    orig = load_numeric(orig_path)
    choice = load_numeric(choice_path).reshape(-1)
    if orig.shape != (N_STUDENTS * N_JOBS, N_DIMS):
        raise ValueError(f"OriginalData shape {orig.shape} != {(400, 7)}")
    if choice.shape != (N_STUDENTS,):
        raise ValueError(f"work_choice length {choice.shape} != {(N_STUDENTS,)}")
    if np.any(choice < 0) or np.any(choice >= N_JOBS):
        raise ValueError("work_choice IDs must be in 0..7")

    student_id = np.repeat(np.arange(N_STUDENTS, dtype=np.int64), N_JOBS)
    job_id = np.tile(np.arange(N_JOBS, dtype=np.int64), N_STUDENTS)
    long_frame = pd.DataFrame(
        {
            "student_id": student_id,
            "job_id": job_id,
            **{DIM_COLS[j]: orig[:, j] for j in range(N_DIMS)},
        }
    )
    long_path = RAW / "student_job_factor_scores.csv"
    long_frame.to_csv(long_path, index=False)

    choice_frame = pd.DataFrame(
        {
            "student_id": np.arange(N_STUDENTS, dtype=np.int64),
            "job_choice_id": np.rint(choice).astype(np.int64),
        }
    )
    choice_csv = RAW / "work_choice.csv"
    choice_frame.to_csv(choice_csv, index=False)

    names = tuple(DIM_COLS)
    report = evaluate_suitability(orig, names)
    fig_path = FIG_DIR / "fig_correlation_matrix_github_7factor.png"
    plot_correlation_heatmap(report.corr, names, fig_path)

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = RESULTS / "github_survey_suitability.txt"
    summary.write_text(
        (
            "source=HiMCM2020/HiMCM_2020 OriginalData.txt via Firecrawl\n"
            f"layout=N{N_STUDENTS}_K{N_JOBS}_D{N_DIMS} long_rows={orig.shape[0]}\n"
            f"scale=[{orig.min():.0f},{orig.max():.0f}]\n"
            f"kmo_overall={report.kmo_overall:.6f}\n"
            f"bartlett_chi2={report.bartlett_chi2:.6f}\n"
            f"bartlett_df={report.bartlett_df}\n"
            f"bartlett_p={report.bartlett_p:.6e}\n"
            f"ln_det_R={report.log_det_r:.6f}\n"
            "note=Does not replace 15-column OriginalData.csv; 7 dims are unnamed in the repo.\n"
            "note=KMO still below 0.70; Bartlett p is not < 0.001 on this 400x7 table.\n"
        ),
        encoding="utf-8",
    )

    source = RAW / "DATA_SOURCE.txt"
    source.write_text(
        (
            "status=SPLIT_SOURCES\n"
            "likert_15col=data/raw/OriginalData.csv  SYNTHETIC_FALLBACK seed=2020  "
            "NOT in HiMCM2020 repo (repo is 7-D not 15-D)\n"
            "github_survey=data/raw/student_job_factor_scores.csv  "
            "400 rows = 50 students x 8 jobs x 7 unnamed factors, scale 10-100\n"
            "github_repo=https://github.com/HiMCM2020/HiMCM_2020\n"
            "retrieved_via=Firecrawl v1/scrape (GitHub raw Connection reset by peer)\n"
            "work_choice=data/raw/work_choice.csv  replaced with GitHub work_choice.txt (50 IDs in 0-7)\n"
            "work_choice_caveat=entropy.py in that repo can overwrite labels from argmax of entropy scores\n"
            f"github_400x7_KMO={report.kmo_overall:.4f}  bartlett_p={report.bartlett_p:.6e}\n"
            "kmo_gap=Neither the 15-col fallback nor the 7-col GitHub matrix reaches KMO>0.70; "
            "fix is new respondents on a designed common-factor scale, not more unrelated crawls.\n"
        ),
        encoding="utf-8",
    )

    print(f"wrote {long_path} shape={long_frame.shape}")
    print(f"wrote {choice_csv} n={len(choice_frame)}")
    print(
        f"GitHub 400x7 KMO={report.kmo_overall:.4f}  "
        f"Bartlett p={report.bartlett_p:.6e}  fig={fig_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

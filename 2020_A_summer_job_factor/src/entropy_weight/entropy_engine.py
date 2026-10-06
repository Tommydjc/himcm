"""Information-entropy weights: objective w in R^3 from student factor scores.

Reference implementation: pack mirror of HiMCM2020 ``entropy.py`` (minmax,
p_{ij} = y_{ij} / sum_i y_{ij}, E_j = -sum p ln p / ln N, D_j = 1 - E_j,
w = D / sum D). This module uses the contest three-factor Thompson scores
instead of the GitHub 7-column table, adds a 10^{-4} shift so p > 0, and
applies F_3 as a cost attribute only in the job utility inner product.

Entropy itself is computed on the unsigned student matrix; the minus sign
on F_3 is an evaluation polarity, not a hand-tuned magnitude.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACK_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_STUDENT_SCORES: Final[Path] = PACK_ROOT / "results" / "student_factor_scores.csv"
DEFAULT_JOB_MATRIX: Final[Path] = PACK_ROOT / "results" / "job_factor_matrix.csv"
DEFAULT_WEIGHTS_CSV: Final[Path] = PACK_ROOT / "results" / "entropy_weights.csv"
DEFAULT_UTILITY_CSV: Final[Path] = PACK_ROOT / "results" / "job_entropy_utility.csv"
DEFAULT_FIGURE: Final[Path] = PACK_ROOT / "paper_figures" / "fig_entropy_weights.png"

N_FACTORS: Final[int] = 3
SHIFT_EPS: Final[float] = 1.0e-4
COST_FACTOR_INDEX: Final[int] = 2
FACTOR_COLUMNS: Final[tuple[str, ...]] = ("Factor_1", "Factor_2", "Factor_3")

ANSI_RESET: Final[str] = "\033[0m"
ANSI_CYAN: Final[str] = "\033[36m"
ANSI_BOLD: Final[str] = "\033[1m"


@dataclass(frozen=True)
class EntropyWeightResult:
    """Entropy-weight payload for m = 3 extracted factors.

    Attributes
    ----------
    n_obs:
        N, number of students used in p_{ij}. Dimension: 1.
    scores:
        Raw Thompson matrix F, shape (N, 3).
    col_min, col_max:
        Per-factor min and max of F, length 3.
    y:
        Shifted min-max matrix, shape (N, 3), y_{ij} in [eps, 1+eps].
    p:
        Column-stochastic shares, shape (N, 3), columns sum to 1.
    entropy:
        E_j in [0, 1], length 3.
    divergence:
        D_j = 1 - E_j, length 3.
    weights:
        w_j = D_j / sum_k D_k, length 3, sum to 1.
    signed_weights:
        (w_1, w_2, -w_3) used in Utility(k).
    job_ids:
        Occupation ids 0..7.
    job_titles:
        English titles aligned with ``job_ids``.
    job_factors:
        LLMFactor coordinates, shape (8, 3).
    utility:
        Utility(k) = w_1 F_{1k} + w_2 F_{2k} - w_3 F_{3k}, length 8.
    factor_names:
        Optional semantic names from the job matrix header row.
    """

    n_obs: int
    scores: np.ndarray
    col_min: np.ndarray
    col_max: np.ndarray
    y: np.ndarray
    p: np.ndarray
    entropy: np.ndarray
    divergence: np.ndarray
    weights: np.ndarray
    signed_weights: np.ndarray
    job_ids: np.ndarray
    job_titles: tuple[str, ...]
    job_factors: np.ndarray
    utility: np.ndarray
    factor_names: tuple[str, ...]


def _color(code: str, text: str) -> str:
    """ANSI wrap when stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{ANSI_RESET}"


def load_student_factor_scores(csv_path: Path | None = None) -> np.ndarray:
    """Read 50 x 3 Thompson scores from ``student_factor_scores.csv``.

    Parameters
    ----------
    csv_path:
        Default pack ``results/student_factor_scores.csv``.

    Returns
    -------
    np.ndarray
        F in R^{N x 3}.
    """
    path = DEFAULT_STUDENT_SCORES if csv_path is None else Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"student factor scores missing: {path}")
    frame = pd.read_csv(path)
    missing = [c for c in FACTOR_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    values = frame.loc[:, list(FACTOR_COLUMNS)].to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_FACTORS:
        raise ValueError(f"expected (N, 3) factor scores, got {values.shape}")
    if values.shape[0] < 2:
        raise ValueError(f"need N >= 2 for entropy shares, got N={values.shape[0]}")
    return values


def load_job_factor_matrix(
    csv_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Read 8 x 3 job coordinates from ``job_factor_matrix.csv``."""
    path = DEFAULT_JOB_MATRIX if csv_path is None else Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"job factor matrix missing: {path}")
    frame = pd.read_csv(path)
    missing = [c for c in FACTOR_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    if "job_id" not in frame.columns:
        raise ValueError(f"{path} missing job_id")
    frame = frame.sort_values("job_id", kind="mergesort")
    job_ids = frame["job_id"].to_numpy(dtype=np.int64)
    if "title_en" in frame.columns:
        titles = tuple(str(x) for x in frame["title_en"].tolist())
    elif "slug" in frame.columns:
        titles = tuple(str(x) for x in frame["slug"].tolist())
    else:
        titles = tuple(f"job_{int(j)}" for j in job_ids)
    factors = frame.loc[:, list(FACTOR_COLUMNS)].to_numpy(dtype=np.float64)
    names = []
    for idx, col in enumerate(("name_F1", "name_F2", "name_F3"), start=1):
        if col in frame.columns and pd.notna(frame[col].iloc[0]):
            names.append(str(frame[col].iloc[0]))
        else:
            names.append(f"Factor_{idx}")
    if factors.shape[0] != 8 or factors.shape[1] != N_FACTORS:
        raise ValueError(f"expected (8, 3) job matrix, got {factors.shape}")
    return job_ids, factors, titles, tuple(names)


def shift_minmax(scores: np.ndarray, eps: float = SHIFT_EPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column min-max to (0, 1] then add eps.

    .. math::

        y_{ij} = \\frac{f_{ij} - \\min_i f_{ij}}{\\max_i f_{ij} - \\min_i f_{ij}} + \\varepsilon

    A degenerate column (max = min) is mapped to the constant 0.5 + eps so that
    E_j = 1 (zero discriminant information) rather than division by zero.

    Parameters
    ----------
    scores:
        F, shape (N, m).
    eps:
        Positive shift, default 10^{-4}.

    Returns
    -------
    y, col_min, col_max
        ``y`` has the same shape as ``scores``.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got {scores.shape}")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")
    col_min = scores.min(axis=0)
    col_max = scores.max(axis=0)
    span = col_max - col_min
    y = np.empty_like(scores, dtype=np.float64)
    for j in range(scores.shape[1]):
        if span[j] <= 0.0:
            y[:, j] = 0.5 + eps
        else:
            y[:, j] = (scores[:, j] - col_min[j]) / span[j] + eps
    return y, col_min.astype(np.float64), col_max.astype(np.float64)


def column_shares(y: np.ndarray) -> np.ndarray:
    """p_{ij} = y_{ij} / sum_i y_{ij}, shape (N, m), columns sum to 1."""
    if y.ndim != 2:
        raise ValueError(f"y must be 2-D, got {y.shape}")
    if np.any(y < 0.0):
        raise ValueError("entropy shares require nonnegative y")
    col_sum = y.sum(axis=0, keepdims=True)
    if np.any(col_sum <= 0.0):
        raise ValueError("column sums of y must be positive")
    return y / col_sum


def information_entropy(p: np.ndarray) -> np.ndarray:
    """E_j = - (1 / ln N) sum_i p_{ij} ln p_{ij}.

    Uses 0 ln 0 := 0 if a share is numerically zero. With the 10^{-4} shift
    this branch should not fire on contest scores.

    Parameters
    ----------
    p:
        Column-stochastic matrix, shape (N, m).

    Returns
    -------
    np.ndarray
        E of length m. For a discrete uniform column, E_j = 1.
    """
    if p.ndim != 2:
        raise ValueError(f"p must be 2-D, got {p.shape}")
    n_obs = p.shape[0]
    if n_obs < 2:
        raise ValueError("N must be >= 2")
    if np.any(p < -1.0e-12):
        raise ValueError("shares must be nonnegative")
    safe = np.clip(p, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(safe > 0.0, safe * np.log(safe), 0.0)
    scale = 1.0 / np.log(float(n_obs))
    entropy = -scale * plogp.sum(axis=0)
    return np.clip(entropy, 0.0, 1.0)


def divergence_coefficients(entropy: np.ndarray) -> np.ndarray:
    """D_j = 1 - E_j."""
    return 1.0 - np.asarray(entropy, dtype=np.float64)


def entropy_weights(divergence: np.ndarray) -> np.ndarray:
    """w_j = D_j / sum_k D_k. If every D_j = 0, return the uniform 1/m."""
    d = np.asarray(divergence, dtype=np.float64).reshape(-1)
    total = float(d.sum())
    if total <= 0.0:
        return np.full(d.size, 1.0 / float(d.size), dtype=np.float64)
    weights = d / total
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()
    return weights


def signed_utility_weights(weights: np.ndarray, cost_index: int = COST_FACTOR_INDEX) -> np.ndarray:
    """Copy of w with the cost factor (default F_3, index 2) negated."""
    signed = np.asarray(weights, dtype=np.float64).copy()
    if not 0 <= cost_index < signed.size:
        raise IndexError(f"cost_index {cost_index} out of range for m={signed.size}")
    signed[cost_index] = -signed[cost_index]
    return signed


def job_utility(job_factors: np.ndarray, signed_weights: np.ndarray) -> np.ndarray:
    """Utility(k) = sum_j s_j F_{k j} with s = (w_1, w_2, -w_3)."""
    if job_factors.ndim != 2 or signed_weights.ndim != 1:
        raise ValueError("job_factors (K, m) and signed_weights (m,)")
    if job_factors.shape[1] != signed_weights.shape[0]:
        raise ValueError(
            f"incompatible job_factors {job_factors.shape} and weights {signed_weights.shape}"
        )
    return job_factors @ signed_weights


def compute_entropy_weights(scores: np.ndarray, eps: float = SHIFT_EPS) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full EWM map F -> (y, p, E, D, w)."""
    y, col_min, col_max = shift_minmax(scores, eps=eps)
    p = column_shares(y)
    entropy = information_entropy(p)
    divergence = divergence_coefficients(entropy)
    weights = entropy_weights(divergence)
    return y, p, entropy, divergence, weights, np.stack([col_min, col_max], axis=0)


def weights_frame(result: EntropyWeightResult) -> pd.DataFrame:
    """Three-row table of E_j, D_j, w_j and evaluation polarity."""
    direction = ("benefit", "benefit", "cost")
    rows = []
    for j in range(N_FACTORS):
        rows.append(
            {
                "factor": FACTOR_COLUMNS[j],
                "name_en": result.factor_names[j],
                "col_min": float(result.col_min[j]),
                "col_max": float(result.col_max[j]),
                "entropy_E": float(result.entropy[j]),
                "divergence_D": float(result.divergence[j]),
                "weight_w": float(result.weights[j]),
                "direction": direction[j],
                "signed_weight": float(result.signed_weights[j]),
                "n_obs": int(result.n_obs),
                "shift_eps": SHIFT_EPS,
            }
        )
    return pd.DataFrame(rows)


def utility_frame(result: EntropyWeightResult) -> pd.DataFrame:
    """Eight-row objective utility table (F_3 enters with a minus)."""
    rank = np.argsort(-result.utility, kind="mergesort")
    rank_of = np.empty(result.utility.size, dtype=np.int64)
    rank_of[rank] = np.arange(1, result.utility.size + 1)
    rows = []
    for i in range(result.utility.size):
        fvec = result.job_factors[i]
        rows.append(
            {
                "job_id": int(result.job_ids[i]),
                "title_en": result.job_titles[i],
                "Factor_1": float(fvec[0]),
                "Factor_2": float(fvec[1]),
                "Factor_3": float(fvec[2]),
                "utility": float(result.utility[i]),
                "rank": int(rank_of[i]),
                "formula": "w1*F1 + w2*F2 - w3*F3",
            }
        )
    return pd.DataFrame(rows)


def export_entropy_tables(
    result: EntropyWeightResult,
    weights_path: Path | None = None,
    utility_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write ``entropy_weights.csv`` and the job-utility companion table."""
    w_path = DEFAULT_WEIGHTS_CSV if weights_path is None else Path(weights_path)
    u_path = DEFAULT_UTILITY_CSV if utility_path is None else Path(utility_path)
    w_path.parent.mkdir(parents=True, exist_ok=True)
    u_path.parent.mkdir(parents=True, exist_ok=True)
    weights_frame(result).to_csv(w_path, index=False)
    utility_frame(result).to_csv(u_path, index=False)
    return w_path, u_path


def plot_entropy_weights(
    weights: np.ndarray,
    entropy: np.ndarray,
    factor_names: tuple[str, ...],
    out_path: Path,
) -> Path:
    """Bar chart of w_j plus pie of the weight simplex."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"$F_{j+1}$" for j in range(len(weights))]
    short = []
    for j, name in enumerate(factor_names):
        tag = name if len(name) <= 28 else name[:26] + "…"
        short.append(f"$F_{j+1}$ {tag}")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), dpi=150)
    colors = ["#1f4e79", "#2e86ab", "#c0392b"]
    axes[0].bar(labels, weights, color=colors, width=0.62)
    axes[0].set_ylim(0.0, max(0.45, float(np.max(weights)) * 1.25))
    axes[0].set_ylabel(r"entropy weight $w_j$")
    axes[0].set_title("Objective weights (student $N=50$)")
    for j, (w_j, e_j) in enumerate(zip(weights, entropy, strict=True)):
        axes[0].text(j, w_j + 0.012, f"$w$={w_j:.3f}\n$E$={e_j:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.35)
    wedges, _texts, autotexts = axes[1].pie(
        weights,
        labels=short,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        pctdistance=0.62,
    )
    for text in autotexts:
        text.set_fontsize(8)
    axes[1].set_title(r"$w_j$ simplex  ($\sum w_j=1$)")
    fig.suptitle("Entropy-weight method on three Varimax factors", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_entropy_report(
    result: EntropyWeightResult,
    weights_path: Path,
    utility_path: Path,
    figure_path: Path,
) -> None:
    """Console summary of E, D, w and job ranking."""
    bar = "=" * 72
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))
    print(_color(ANSI_BOLD + ANSI_CYAN, "  Entropy weights  |  2020 A  (N=50 students, m=3)"))
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))
    print(f"  N students          : {result.n_obs}")
    print(f"  shift eps           : {SHIFT_EPS:g}")
    print("  E_j = -(1/ln N) sum_i p_ij ln p_ij ;  D_j = 1-E_j ;  w = D / sum D")
    print()
    for j in range(N_FACTORS):
        print(
            f"  {FACTOR_COLUMNS[j]:10s}  "
            f"[{result.col_min[j]:+.3f}, {result.col_max[j]:+.3f}]  "
            f"E={result.entropy[j]:.6f}  D={result.divergence[j]:.6f}  "
            f"w={result.weights[j]:.6f}  signed={result.signed_weights[j]:+.6f}  "
            f"| {result.factor_names[j]}"
        )
    print(f"  sum w               : {float(np.sum(result.weights)):.12f}")
    print()
    print("  Utility(k) = w1 F1 + w2 F2 - w3 F3")
    order = np.argsort(-result.utility, kind="mergesort")
    for rank, idx in enumerate(order, start=1):
        print(
            f"   {rank:2d}. job {int(result.job_ids[idx])} {result.job_titles[idx]:20s}  "
            f"U={result.utility[idx]:+.4f}  "
            f"F=({result.job_factors[idx, 0]:+.2f}, "
            f"{result.job_factors[idx, 1]:+.2f}, "
            f"{result.job_factors[idx, 2]:+.2f})"
        )
    print()
    print(f"  weights CSV         : {weights_path}")
    print(f"  utility CSV         : {utility_path}")
    print(f"  figure PNG          : {figure_path}")
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))


def run_entropy_pipeline(
    student_path: Path | None = None,
    job_path: Path | None = None,
    weights_path: Path | None = None,
    utility_path: Path | None = None,
    figure_path: Path | None = None,
) -> EntropyWeightResult:
    """Load student F, estimate w, score eight jobs, write CSV + figure.

    Parameters
    ----------
    student_path:
        Thompson scores. Default ``results/student_factor_scores.csv``.
    job_path:
        LLMFactor job matrix. Default ``results/job_factor_matrix.csv``.
    weights_path:
        Default ``results/entropy_weights.csv``.
    utility_path:
        Default ``results/job_entropy_utility.csv``.
    figure_path:
        Default ``paper_figures/fig_entropy_weights.png``.
    """
    scores = load_student_factor_scores(student_path)
    y, p, entropy, divergence, weights, minmax = compute_entropy_weights(scores)
    col_min, col_max = minmax[0], minmax[1]
    signed = signed_utility_weights(weights)
    job_ids, job_factors, titles, factor_names = load_job_factor_matrix(job_path)
    utility = job_utility(job_factors, signed)
    result = EntropyWeightResult(
        n_obs=int(scores.shape[0]),
        scores=scores,
        col_min=col_min,
        col_max=col_max,
        y=y,
        p=p,
        entropy=entropy,
        divergence=divergence,
        weights=weights,
        signed_weights=signed,
        job_ids=job_ids,
        job_titles=titles,
        job_factors=job_factors,
        utility=utility,
        factor_names=factor_names,
    )
    w_out, u_out = export_entropy_tables(result, weights_path, utility_path)
    fig_out = DEFAULT_FIGURE if figure_path is None else Path(figure_path)
    plot_entropy_weights(result.weights, result.entropy, result.factor_names, fig_out)
    print_entropy_report(result, w_out, u_out, fig_out)
    return result


def main() -> int:
    """CLI: python -m src.entropy_weight.entropy_engine"""
    run_entropy_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())

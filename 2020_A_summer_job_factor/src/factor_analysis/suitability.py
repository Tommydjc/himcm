"""EDA, KMO, and Bartlett sphericity for the 15-item summer-job questionnaire.

Pipeline (do not skip before extraction):
1. Z-score each of the p = 15 Likert columns.
2. Pearson R in R^{p x p}.
3. Kaiser–Meyer–Olkin overall KMO and per-item MSA from R and partial
   correlations implied by R^{-1}.
4. Bartlett test of H0: R = I_p.

All numbers are computed from ``data/raw/OriginalData.csv`` on disk.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2

PACK_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CSV: Final[Path] = PACK_ROOT / "data" / "raw" / "OriginalData.csv"
DEFAULT_CHOICE_CSV: Final[Path] = PACK_ROOT / "data" / "raw" / "work_choice.csv"
DEFAULT_FIGURE: Final[Path] = PACK_ROOT / "paper_figures" / "fig_correlation_matrix.png"

KMO_WARN_THRESHOLD: Final[float] = 0.70
BARTLETT_P_ASSERT: Final[float] = 1.0e-3
RIDGE: Final[float] = 1.0e-10

ANSI_RESET: Final[str] = "\033[0m"
ANSI_GREEN: Final[str] = "\033[32m"
ANSI_YELLOW: Final[str] = "\033[33m"
ANSI_RED: Final[str] = "\033[31m"
ANSI_CYAN: Final[str] = "\033[36m"
ANSI_BOLD: Final[str] = "\033[1m"


@dataclass(frozen=True)
class SuitabilityReport:
    """Numeric payload of the factor-suitability battery.

    Attributes
    ----------
    n_obs:
        Sample size N (students). Dimension: 1.
    n_var:
        Number of Likert items p. Dimension: 1.
    z:
        Z-scored matrix in R^{N x p}.
    corr:
        Pearson R in R^{p x p}.
    kmo_overall:
        Overall Kaiser–Meyer–Olkin statistic in (0, 1], or NaN if undefined.
    msa:
        Per-variable MSA, length p.
    bartlett_chi2:
        Chi-square statistic for H0: R = I_p. Dimension: 1.
    bartlett_df:
        Degrees of freedom p(p-1)/2.
    bartlett_p:
        Upper-tail p-value under chi-square(df).
    log_det_r:
        ln|R| used in the Bartlett formula.
    feature_names:
        Column names aligned with ``msa``.
    """

    n_obs: int
    n_var: int
    z: np.ndarray
    corr: np.ndarray
    kmo_overall: float
    msa: np.ndarray
    bartlett_chi2: float
    bartlett_df: int
    bartlett_p: float
    log_det_r: float
    feature_names: tuple[str, ...]


def _color(code: str, text: str) -> str:
    """ANSI wrap when stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{ANSI_RESET}"


def load_questionnaire_matrix(
    csv_path: Path | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read the 15-item wide table.

    Parameters
    ----------
    csv_path:
        Defaults to pack ``data/raw/OriginalData.csv``.

    Returns
    -------
    x, names
        ``x`` has shape (N, p) with p = 15; ``names`` are CSV headers.
    """
    path = DEFAULT_CSV if csv_path is None else Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"questionnaire CSV missing: {path}")
    frame = pd.read_csv(path)
    values = frame.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 15:
        raise ValueError(f"expected (N, 15) Likert table, got {values.shape}")
    if values.shape[0] < 3:
        raise ValueError(f"need N >= 3 for correlation, got N={values.shape[0]}")
    names = tuple(str(col) for col in frame.columns)
    return values, names


def zscore_standardize(x: np.ndarray) -> np.ndarray:
    """Column Z-score Z = (X - mu) / sigma (population ddof=0, matching PCA scaling).

    Parameters
    ----------
    x:
        Raw scores, shape (N, p).

    Returns
    -------
    np.ndarray
        Same shape; each column has mean 0 and (ddof=0) std 1.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got {x.shape}")
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, ddof=0, keepdims=True)
    if np.any(sigma <= 0.0):
        raise ValueError("at least one column has zero variance; cannot Z-score")
    return (x - mu) / sigma


def pearson_correlation(z: np.ndarray) -> np.ndarray:
    """Pearson R from already-centered (or raw) columns, shape (p, p).

    Using ``np.corrcoef`` on Z-scored data is algebraically the Gram matrix
    (1/(N)) Z^T Z when Z uses ddof=0, up to the 1/(N-1) convention inside
    ``corrcoef``. We use ``corrcoef`` so R is the standard Pearson matrix
    with unit diagonal.
    """
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError(f"need at least 2 variables, got {z.shape}")
    corr = np.corrcoef(z, rowvar=False)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def _partial_correlation_from_precision(corr: np.ndarray) -> np.ndarray:
    """Partial correlations a_{ij} = -p_{ij} / sqrt(p_{ii} p_{jj}) from R^{-1}.

    A small ridge is added if ``slogdet`` reports a non-positive determinant.
    """
    p = corr.shape[0]
    sign, logdet = np.linalg.slogdet(corr)
    matrix = corr.copy()
    if sign <= 0.0 or not np.isfinite(logdet):
        matrix = corr + RIDGE * np.eye(p)
    precision = np.linalg.inv(matrix)
    diag = np.clip(np.diag(precision), 1.0e-15, None)
    scale = np.sqrt(np.outer(diag, diag))
    partial = -precision / scale
    np.fill_diagonal(partial, 1.0)
    partial = 0.5 * (partial + partial.T)
    np.fill_diagonal(partial, 1.0)
    return partial


def kaiser_meyer_olkin(corr: np.ndarray) -> tuple[float, np.ndarray]:
    """Overall KMO and per-variable MSA.

    For i ≠ j let r_{ij} be Pearson correlations and a_{ij} partial correlations.

    .. math::

        \\mathrm{MSA}_i = \\frac{\\sum_{j \\neq i} r_{ij}^2}
        {\\sum_{j \\neq i} r_{ij}^2 + \\sum_{j \\neq i} a_{ij}^2}

        \\mathrm{KMO} = \\frac{\\sum_{i \\neq j} r_{ij}^2}
        {\\sum_{i \\neq j} r_{ij}^2 + \\sum_{i \\neq j} a_{ij}^2}

    Parameters
    ----------
    corr:
        Symmetric Pearson matrix, shape (p, p), unit diagonal.

    Returns
    -------
    kmo, msa
        Scalar KMO and vector MSA of length p. Both NaN if every off-diagonal
        r and a is zero (identity / undefined).
    """
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be square, got {corr.shape}")
    p = corr.shape[0]
    partial = _partial_correlation_from_precision(corr)
    r2 = corr.astype(np.float64) ** 2
    a2 = partial.astype(np.float64) ** 2
    np.fill_diagonal(r2, 0.0)
    np.fill_diagonal(a2, 0.0)
    num_i = r2.sum(axis=1)
    den_i = num_i + a2.sum(axis=1)
    msa = np.full(p, np.nan, dtype=np.float64)
    ok = den_i > 0.0
    msa[ok] = num_i[ok] / den_i[ok]
    num = float(r2.sum())
    den = num + float(a2.sum())
    kmo = float(num / den) if den > 0.0 else float("nan")
    return kmo, msa


def bartlett_sphericity(
    corr: np.ndarray,
    n_obs: int,
) -> tuple[float, int, float, float]:
    """Bartlett test of H0: R = I_p.

    .. math::

        \\chi^2 = -\\left(N - 1 - \\frac{2p + 5}{6}\\right) \\ln |R|

    with df = p(p-1)/2.

    Parameters
    ----------
    corr:
        Pearson R, shape (p, p).
    n_obs:
        N, number of students.

    Returns
    -------
    chi2_stat, df, p_value, log_det
    """
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be square, got {corr.shape}")
    p = int(corr.shape[0])
    if n_obs <= p:
        warnings.warn(
            f"N={n_obs} is not larger than p={p}; Bartlett chi-square is still "
            "reported but the Wishart approximation is poor.",
            RuntimeWarning,
            stacklevel=2,
        )
    sign, log_det = np.linalg.slogdet(corr)
    if sign <= 0.0 or not np.isfinite(log_det):
        log_det = float(-1.0e12)
    coef = float(n_obs - 1 - (2 * p + 5) / 6.0)
    chi2_stat = float(-coef * log_det)
    df = int(p * (p - 1) / 2)
    p_value = float(chi2.sf(chi2_stat, df))
    return chi2_stat, df, p_value, float(log_det)


def kmo_verbal(kmo: float) -> str:
    """Kaiser (1974) verbal labels for overall KMO."""
    if not np.isfinite(kmo):
        return "undefined"
    if kmo >= 0.90:
        return "marvelous"
    if kmo >= 0.80:
        return "meritorious"
    if kmo >= 0.70:
        return "middling"
    if kmo >= 0.60:
        return "mediocre"
    if kmo >= 0.50:
        return "miserable"
    return "unacceptable"


def evaluate_suitability(x: np.ndarray, feature_names: tuple[str, ...]) -> SuitabilityReport:
    """Z-score, R, KMO/MSA, and Bartlett on a numeric matrix."""
    z = zscore_standardize(x)
    corr = pearson_correlation(z)
    kmo, msa = kaiser_meyer_olkin(corr)
    n_obs, n_var = x.shape
    chi2_stat, df, p_value, log_det = bartlett_sphericity(corr, n_obs)
    return SuitabilityReport(
        n_obs=int(n_obs),
        n_var=int(n_var),
        z=z,
        corr=corr,
        kmo_overall=kmo,
        msa=msa,
        bartlett_chi2=chi2_stat,
        bartlett_df=df,
        bartlett_p=p_value,
        log_det_r=log_det,
        feature_names=feature_names,
    )


def plot_correlation_heatmap(
    corr: np.ndarray,
    feature_names: tuple[str, ...],
    out_path: Path,
) -> Path:
    """Save a p x p Pearson heatmap to ``out_path`` (PNG)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 9.0), dpi=150)
    sns.heatmap(
        corr,
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
        square=True,
        xticklabels=list(feature_names),
        yticklabels=list(feature_names),
        ax=ax,
        cbar_kws={"label": r"Pearson $r$"},
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_title("Pearson correlation of 15 summer-job Likert items (Z-scored)")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_suitability_report(
    report: SuitabilityReport,
    figure_path: Path,
    kmo_ok: bool,
    bartlett_ok: bool,
    n_choices: int | None,
) -> None:
    """Pretty console report (Chinese labels, English statistic names)."""
    bar = "=" * 72
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))
    print(_color(ANSI_BOLD + ANSI_CYAN, "  Factor suitability  |  2020 A questionnaire EDA"))
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))
    print(f"  N students          : {report.n_obs}")
    print(f"  p Likert items      : {report.n_var}")
    if n_choices is not None:
        print(f"  work_choice rows    : {n_choices}  (aligned check only, not in R)")
    off = report.corr.copy()
    np.fill_diagonal(off, np.nan)
    print(
        f"  off-diag r          : min={np.nanmin(off):+.3f}  "
        f"max={np.nanmax(off):+.3f}  mean={np.nanmean(off):+.3f}"
    )
    print()
    print(_color(ANSI_BOLD, "  Kaiser–Meyer–Olkin"))
    kmo_line = (
        f"  overall KMO         : {report.kmo_overall:.4f}  "
        f"({kmo_verbal(report.kmo_overall)})"
    )
    print(_color(ANSI_GREEN if kmo_ok else ANSI_YELLOW, kmo_line))
    print(f"  threshold           : KMO > {KMO_WARN_THRESHOLD:.2f}  (warning if violated)")
    print("  per-item MSA:")
    for name, msa_i in zip(report.feature_names, report.msa, strict=True):
        flag = "  " if np.isfinite(msa_i) and msa_i >= 0.50 else " !"
        print(f"   {flag} {name:24s}  {msa_i:7.4f}")
    print()
    print(_color(ANSI_BOLD, "  Bartlett sphericity  H0: R = I_p"))
    print(r"  chi^2 = -(N - 1 - (2p+5)/6) ln|R|")
    print(f"  ln|R|               : {report.log_det_r:.6f}")
    print(f"  chi^2               : {report.bartlett_chi2:.4f}")
    print(f"  df                  : {report.bartlett_df}")
    bart_line = f"  p-value             : {report.bartlett_p:.6e}"
    print(_color(ANSI_GREEN if bartlett_ok else ANSI_RED, bart_line))
    print(f"  assert              : p < {BARTLETT_P_ASSERT}")
    print()
    print(f"  heatmap             : {figure_path}")
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))


def _choice_row_count(path: Path = DEFAULT_CHOICE_CSV) -> int | None:
    """Optional alignment check against work_choice.csv."""
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    return int(len(frame))


def run_suitability_pipeline(
    csv_path: Path | None = None,
    figure_path: Path | None = None,
    assert_bartlett: bool = True,
) -> SuitabilityReport:
    """Load questionnaire, test factorability, write the correlation figure.

    Parameters
    ----------
    csv_path:
        Likert wide table. Default pack OriginalData.csv.
    figure_path:
        PNG output. Default ``paper_figures/fig_correlation_matrix.png``.
    assert_bartlett:
        If True, raise AssertionError when p >= 0.001.

    Returns
    -------
    SuitabilityReport
    """
    x, names = load_questionnaire_matrix(csv_path)
    report = evaluate_suitability(x, names)
    fig = DEFAULT_FIGURE if figure_path is None else Path(figure_path)
    plot_correlation_heatmap(report.corr, report.feature_names, fig)

    kmo_ok = bool(np.isfinite(report.kmo_overall) and report.kmo_overall > KMO_WARN_THRESHOLD)
    bartlett_ok = bool(report.bartlett_p < BARTLETT_P_ASSERT)
    n_choices = _choice_row_count()
    print_suitability_report(report, fig, kmo_ok, bartlett_ok, n_choices)

    if n_choices is not None and n_choices != report.n_obs:
        warnings.warn(
            f"work_choice.csv has {n_choices} rows but OriginalData has N={report.n_obs}",
            RuntimeWarning,
            stacklevel=2,
        )
    if not kmo_ok:
        warnings.warn(
            f"overall KMO={report.kmo_overall:.4f} is not > {KMO_WARN_THRESHOLD}; "
            "common-factor extraction may be weakly supported.",
            UserWarning,
            stacklevel=2,
        )
    if assert_bartlett and not bartlett_ok:
        raise AssertionError(
            f"Bartlett p-value {report.bartlett_p:.6e} is not < {BARTLETT_P_ASSERT}; "
            "cannot reject H0: R = I at the requested level."
        )
    return report


def main() -> int:
    """CLI entry: python -m src.factor_analysis.suitability"""
    run_suitability_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())

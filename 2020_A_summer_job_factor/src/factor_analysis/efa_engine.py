"""Principal-component EFA: Kaiser retention, Varimax, Thompson scores.

Extraction is the spectral decomposition of the Pearson matrix R (principal
component method). Rotation is Kaiser-normalized Varimax. Factor scores use
the Thomson/Thompson regression estimator W = R^{-1} Lambda^*.

Contest modeling choice: retain m = 3 factors (orthogonal scores for the
downstream NN). Kaiser (lambda > 1) and 65% cumulative variance are reported
as diagnostics; they do not silently override m = 3.
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

from src.factor_analysis.suitability import (
    DEFAULT_CSV,
    PACK_ROOT,
    evaluate_suitability,
    load_questionnaire_matrix,
)

CONTEST_N_FACTORS: Final[int] = 3
KAISER_EIGENVALUE: Final[float] = 1.0
CUMULATIVE_VARIANCE_MIN: Final[float] = 0.65
VARIMAX_MAX_ITER: Final[int] = 200
VARIMAX_TOL: Final[float] = 1.0e-8
RIDGE: Final[float] = 1.0e-10

DEFAULT_LOADINGS_CSV: Final[Path] = PACK_ROOT / "results" / "factor_loadings.csv"
DEFAULT_SCORES_CSV: Final[Path] = PACK_ROOT / "results" / "student_factor_scores.csv"
DEFAULT_SCREE_PNG: Final[Path] = PACK_ROOT / "paper_figures" / "fig_scree_plot.png"

ANSI_RESET: Final[str] = "\033[0m"
ANSI_CYAN: Final[str] = "\033[36m"
ANSI_BOLD: Final[str] = "\033[1m"
ANSI_YELLOW: Final[str] = "\033[33m"


@dataclass(frozen=True)
class EFAResult:
    """Payload of PC-EFA with Varimax and Thompson scores.

    Attributes
    ----------
    n_obs:
        N, number of students. Dimension: 1.
    n_var:
        p, number of Likert items. Dimension: 1.
    n_factors:
        m, retained factors (contest: 3). Dimension: 1.
    z:
        Z-scored questionnaire, shape (N, p).
    corr:
        Pearson R, shape (p, p).
    eigenvalues:
        lambda_1 >= ... >= lambda_p of R, length p.
    eigenvectors:
        Columns v_j corresponding to ``eigenvalues``, shape (p, p).
    n_kaiser:
        Count of eigenvalues strictly greater than 1.
    n_cum65:
        Smallest m with cumulative variance contribution > 65%.
    unrotated_loadings:
        Lambda = V_m diag(sqrt(lambda_m)), shape (p, m).
    rotation_matrix:
        Orthogonal T in R^{m x m} with Lambda^* = Lambda T.
    rotated_loadings:
        Lambda^*, shape (p, m).
    communalities:
        h_i^2 = sum_j (lambda^*_{ij})^2, length p.
    uniqueness:
        psi_i = 1 - h_i^2, length p.
    variance_explained:
        Column sums of squares of Lambda^* (eigenvalue analogue after rotation),
        length m.
    proportion_variance:
        variance_explained / p, length m.
    score_weights:
        W = R^{-1} Lambda^*, shape (p, m).
    factor_scores:
        F = Z W, shape (N, m).
    primary_factor:
        1-based factor index of max |loading| for each item, length p.
    feature_names:
        Likert column names, length p.
    factor_names:
        Column labels Factor_1 .. Factor_m.
    """

    n_obs: int
    n_var: int
    n_factors: int
    z: np.ndarray
    corr: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    n_kaiser: int
    n_cum65: int
    unrotated_loadings: np.ndarray
    rotation_matrix: np.ndarray
    rotated_loadings: np.ndarray
    communalities: np.ndarray
    uniqueness: np.ndarray
    variance_explained: np.ndarray
    proportion_variance: np.ndarray
    score_weights: np.ndarray
    factor_scores: np.ndarray
    primary_factor: np.ndarray
    feature_names: tuple[str, ...]
    factor_names: tuple[str, ...]


def _color(code: str, text: str) -> str:
    """ANSI wrap when stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{ANSI_RESET}"


def eigendecompose_correlation(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spectral decomposition of symmetric R with lambda_1 >= ... >= lambda_p.

    Parameters
    ----------
    corr:
        Pearson matrix, shape (p, p).

    Returns
    -------
    eigenvalues, eigenvectors
        Eigenvalues length p; eigenvectors shape (p, p), column j is v_j.
    """
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be square, got {corr.shape}")
    symmetric = 0.5 * (corr + corr.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    for j in range(eigenvectors.shape[1]):
        if eigenvectors[np.argmax(np.abs(eigenvectors[:, j])), j] < 0.0:
            eigenvectors[:, j] *= -1.0
    return eigenvalues.astype(np.float64), eigenvectors.astype(np.float64)


def kaiser_and_cumulative_rank(
    eigenvalues: np.ndarray,
    kaiser_threshold: float = KAISER_EIGENVALUE,
    cum_min: float = CUMULATIVE_VARIANCE_MIN,
) -> tuple[int, int, np.ndarray]:
    """Kaiser count (lambda > 1) and smallest m with cumulative share > cum_min.

    For a correlation matrix, sum_j lambda_j = p, so the cumulative contribution
    of the first m components is (sum_{j=1}^m lambda_j) / p.

    Parameters
    ----------
    eigenvalues:
        Descending eigenvalues of R, length p.
    kaiser_threshold:
        Retain lambda > threshold (Kaiser, 1960).
    cum_min:
        Cumulative variance floor (contest: 0.65).

    Returns
    -------
    n_kaiser, n_cum, cumulative_share
        ``cumulative_share`` has length p and values in (0, 1].
    """
    eig = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    p = eig.size
    if p == 0:
        raise ValueError("eigenvalues must be non-empty")
    total = float(np.sum(eig))
    if total <= 0.0:
        raise ValueError("sum of eigenvalues must be positive")
    cumulative_share = np.cumsum(eig) / total
    n_kaiser = int(np.sum(eig > kaiser_threshold))
    above = np.where(cumulative_share > cum_min)[0]
    n_cum = int(above[0] + 1) if above.size else p
    return n_kaiser, n_cum, cumulative_share


def unrotated_pc_loadings(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_factors: int,
) -> np.ndarray:
    """Lambda = V_m diag(sqrt(lambda_m)), shape (p, m).

    Negative eigenvalues (numerical noise) are clipped at 0 before the square
    root; those columns would be all zeros and should not be retained.
    """
    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors}")
    if eigenvectors.shape[1] < n_factors or eigenvalues.shape[0] < n_factors:
        raise ValueError("n_factors exceeds spectral rank")
    lam = np.clip(eigenvalues[:n_factors], 0.0, None)
    return eigenvectors[:, :n_factors] * np.sqrt(lam)


def varimax_criterion(loadings: np.ndarray) -> float:
    """Sum over columns of the (population) variance of squared loadings.

    .. math::

        Q = \\sum_{j=1}^{m}
        \\left[
            \\frac{1}{p}\\sum_{i=1}^{p} (\\lambda_{ij}^2)^2
            - \\left(\\frac{1}{p}\\sum_{i=1}^{p} \\lambda_{ij}^2\\right)^2
        \\right]
    """
    squared = np.asarray(loadings, dtype=np.float64) ** 2
    p = squared.shape[0]
    col_mean = squared.mean(axis=0)
    col_second = (squared**2).mean(axis=0)
    return float(np.sum(col_second - col_mean**2))


def varimax_rotate(
    loadings: np.ndarray,
    *,
    kaiser_normalize: bool = True,
    max_iter: int = VARIMAX_MAX_ITER,
    tol: float = VARIMAX_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal Varimax (gamma = 1) via the SVD gradient iteration.

    Solves for T with T^T T = I_m maximizing the Varimax criterion of
    Lambda^* = Lambda T (Kaiser row-normalization optional).

    Parameters
    ----------
    loadings:
        Unrotated Lambda, shape (p, m).
    kaiser_normalize:
        Divide each row by its communality sqrt before rotating, then
        rescale (Kaiser 1958).
    max_iter, tol:
        Iteration cap and relative SVD-sum tolerance.

    Returns
    -------
    rotated, T
        Lambda^* shape (p, m); T shape (m, m), orthogonal.
    """
    phi = np.asarray(loadings, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError(f"loadings must be 2-D, got {phi.shape}")
    p, m = phi.shape
    if m < 2:
        return phi.copy(), np.eye(m)

    row_norm = np.ones(p, dtype=np.float64)
    work = phi.copy()
    if kaiser_normalize:
        row_norm = np.sqrt(np.sum(work**2, axis=1))
        row_norm = np.where(row_norm > 0.0, row_norm, 1.0)
        work = work / row_norm[:, None]

    t_mat = np.eye(m, dtype=np.float64)
    d_old = 0.0
    gamma = 1.0
    for iteration in range(max_iter):
        rotated = work @ t_mat
        col_ss = np.sum(rotated**2, axis=0)
        gradient = rotated**3 - (gamma / float(p)) * rotated * col_ss
        u_mat, singular, vh = np.linalg.svd(work.T @ gradient, full_matrices=True)
        t_mat = u_mat @ vh
        d_new = float(np.sum(singular))
        if iteration > 0 and d_old > 0.0 and (d_new - d_old) < tol * abs(d_new):
            break
        d_old = d_new

    rotated_norm = work @ t_mat
    rotated = rotated_norm * row_norm[:, None]
    ortho_err = np.linalg.norm(t_mat.T @ t_mat - np.eye(m), ord="fro")
    if ortho_err > 1.0e-8:
        q_mat, _ = np.linalg.qr(t_mat)
        t_mat = q_mat
        rotated_norm = work @ t_mat
        rotated = rotated_norm * row_norm[:, None]
    return rotated, t_mat


def _sign_and_order_factors(
    rotated: np.ndarray,
    t_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive-sum columns, then sort by column sum of squares (desc)."""
    signed = rotated.copy()
    t_out = t_mat.copy()
    for j in range(signed.shape[1]):
        if np.sum(signed[:, j]) < 0.0:
            signed[:, j] *= -1.0
            t_out[:, j] *= -1.0
    ss = np.sum(signed**2, axis=0)
    order = np.argsort(ss)[::-1]
    return signed[:, order], t_out[:, order], ss[order]


def invert_correlation(corr: np.ndarray) -> np.ndarray:
    """R^{-1} with a ridge if slogdet reports a non-positive determinant."""
    p = corr.shape[0]
    sign, logdet = np.linalg.slogdet(corr)
    matrix = corr.copy()
    if sign <= 0.0 or not np.isfinite(logdet):
        matrix = corr + RIDGE * np.eye(p)
    return np.linalg.inv(matrix)


def thompson_score_weights(corr: np.ndarray, rotated_loadings: np.ndarray) -> np.ndarray:
    """Thomson/Thompson regression weights W = R^{-1} Lambda^*, shape (p, m)."""
    if corr.shape[0] != rotated_loadings.shape[0]:
        raise ValueError(
            f"R is {corr.shape} but Lambda^* is {rotated_loadings.shape}"
        )
    return invert_correlation(corr) @ rotated_loadings


def thompson_factor_scores(z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """F = Z W in R^{N x m}."""
    if z.ndim != 2 or weights.ndim != 2 or z.shape[1] != weights.shape[0]:
        raise ValueError(f"incompatible Z {z.shape} and W {weights.shape}")
    return z @ weights


def primary_factor_membership(rotated_loadings: np.ndarray) -> np.ndarray:
    """1-based index of the factor with largest |loading| for each item."""
    return np.argmax(np.abs(rotated_loadings), axis=1).astype(np.int64) + 1


def extract_efa(
    z: np.ndarray,
    corr: np.ndarray,
    feature_names: tuple[str, ...],
    n_factors: int = CONTEST_N_FACTORS,
) -> EFAResult:
    """Run PC extraction, Varimax, and Thompson scores on a prepared R and Z."""
    n_obs, n_var = z.shape
    if corr.shape != (n_var, n_var):
        raise ValueError(f"R shape {corr.shape} does not match Z columns {n_var}")
    if n_factors > n_var:
        raise ValueError(f"m={n_factors} exceeds p={n_var}")

    eigenvalues, eigenvectors = eigendecompose_correlation(corr)
    n_kaiser, n_cum65, _cum = kaiser_and_cumulative_rank(eigenvalues)
    if n_kaiser != n_factors:
        warnings.warn(
            f"Kaiser lambda>1 suggests m={n_kaiser}, contest extraction uses m={n_factors}.",
            UserWarning,
            stacklevel=2,
        )
    cum_at_m = float(np.sum(eigenvalues[:n_factors]) / np.sum(eigenvalues))
    if cum_at_m <= CUMULATIVE_VARIANCE_MIN:
        warnings.warn(
            f"cumulative variance at m={n_factors} is {cum_at_m:.4f} "
            f"(not > {CUMULATIVE_VARIANCE_MIN:.2f}); still extracting m={n_factors}.",
            UserWarning,
            stacklevel=2,
        )

    unrotated = unrotated_pc_loadings(eigenvalues, eigenvectors, n_factors)
    rotated_raw, t_raw = varimax_rotate(unrotated)
    rotated, t_mat, ss = _sign_and_order_factors(rotated_raw, t_raw)
    q_before = varimax_criterion(unrotated)
    q_after = varimax_criterion(rotated)
    if q_after + 1.0e-12 < q_before:
        warnings.warn(
            f"Varimax criterion decreased ({q_before:.6f} -> {q_after:.6f}); "
            "check rotation numerics.",
            UserWarning,
            stacklevel=2,
        )

    communalities = np.sum(rotated**2, axis=1)
    uniqueness = 1.0 - communalities
    proportion = ss / float(n_var)
    weights = thompson_score_weights(corr, rotated)
    scores = thompson_factor_scores(z, weights)
    factor_names = tuple(f"Factor_{j}" for j in range(1, n_factors + 1))
    return EFAResult(
        n_obs=int(n_obs),
        n_var=int(n_var),
        n_factors=int(n_factors),
        z=z,
        corr=corr,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        n_kaiser=n_kaiser,
        n_cum65=n_cum65,
        unrotated_loadings=unrotated,
        rotation_matrix=t_mat,
        rotated_loadings=rotated,
        communalities=communalities,
        uniqueness=uniqueness,
        variance_explained=ss,
        proportion_variance=proportion,
        score_weights=weights,
        factor_scores=scores,
        primary_factor=primary_factor_membership(rotated),
        feature_names=feature_names,
        factor_names=factor_names,
    )


def loadings_frame(result: EFAResult) -> pd.DataFrame:
    """Wide loadings table with communality, uniqueness, and primary factor."""
    data: dict[str, object] = {"item": list(result.feature_names)}
    for name, col in zip(result.factor_names, result.rotated_loadings.T, strict=True):
        data[name] = col
    data["communality"] = result.communalities
    data["uniqueness"] = result.uniqueness
    data["primary_factor"] = result.primary_factor
    abs_load = np.abs(result.rotated_loadings)
    rows = np.arange(result.n_var)
    data["primary_loading"] = result.rotated_loadings[rows, result.primary_factor - 1]
    data["primary_abs_loading"] = abs_load[rows, result.primary_factor - 1]
    frame = pd.DataFrame(data)
    return frame.sort_values(
        by=["primary_factor", "primary_abs_loading"],
        ascending=[True, False],
        ignore_index=True,
    )


def scores_frame(result: EFAResult) -> pd.DataFrame:
    """Student factor-score table, student_id = 1 .. N."""
    data: dict[str, object] = {
        "student_id": np.arange(1, result.n_obs + 1, dtype=np.int64),
    }
    for name, col in zip(result.factor_names, result.factor_scores.T, strict=True):
        data[name] = col
    return pd.DataFrame(data)


def export_efa_tables(
    result: EFAResult,
    loadings_path: Path | None = None,
    scores_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write rotated loadings and Thompson scores as CSV."""
    load_path = DEFAULT_LOADINGS_CSV if loadings_path is None else Path(loadings_path)
    score_path = DEFAULT_SCORES_CSV if scores_path is None else Path(scores_path)
    load_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    loadings_frame(result).to_csv(load_path, index=False)
    scores_frame(result).to_csv(score_path, index=False)
    return load_path, score_path


def plot_scree(
    eigenvalues: np.ndarray,
    n_factors: int,
    out_path: Path,
    n_kaiser: int | None = None,
) -> Path:
    """Scree plot with Kaiser lambda=1 line and retained-m marker."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eig = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    ranks = np.arange(1, eig.size + 1)
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=150)
    ax.plot(ranks, eig, color="#1f4e79", marker="o", linewidth=1.8, markersize=5.5)
    ax.axhline(
        KAISER_EIGENVALUE,
        color="#c0392b",
        linestyle="--",
        linewidth=1.4,
        label=r"Kaiser threshold $\lambda=1$",
    )
    ax.axvline(
        n_factors,
        color="#7d3c98",
        linestyle=":",
        linewidth=1.4,
        label=rf"retained $m={n_factors}$",
    )
    ax.scatter(
        [n_factors],
        [eig[n_factors - 1]],
        s=90,
        zorder=5,
        color="#7d3c98",
        label="elbow / contest $m$",
    )
    if n_kaiser is not None and 1 <= n_kaiser <= eig.size:
        ax.scatter(
            [n_kaiser],
            [eig[n_kaiser - 1]],
            s=55,
            zorder=4,
            facecolors="none",
            edgecolors="#c0392b",
            linewidths=1.6,
            label=rf"last $\lambda>1$ (rank {n_kaiser})",
        )
    ax.set_xlabel("component rank $j$")
    ax.set_ylabel(r"eigenvalue $\lambda_j$ of $R$")
    ax.set_title("Scree plot of the 15-item Pearson matrix")
    ax.set_xticks(ranks)
    ax.set_xlim(0.5, eig.size + 0.5)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=True, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_efa_report(
    result: EFAResult,
    loadings_path: Path,
    scores_path: Path,
    scree_path: Path,
) -> None:
    """Console summary of retention, rotation, and exports."""
    bar = "=" * 72
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))
    print(_color(ANSI_BOLD + ANSI_CYAN, "  PC-EFA + Varimax + Thompson  |  2020 A"))
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))
    print(f"  N students          : {result.n_obs}")
    print(f"  p Likert items      : {result.n_var}")
    print(f"  contest m           : {result.n_factors}")
    print(f"  Kaiser # (lambda>1) : {result.n_kaiser}")
    print(f"  smallest m, cum>65% : {result.n_cum65}")
    print("  eigenvalues (desc)  :")
    for j, lam in enumerate(result.eigenvalues, start=1):
        share = lam / float(np.sum(result.eigenvalues))
        cum = float(np.sum(result.eigenvalues[:j]) / np.sum(result.eigenvalues))
        mark = " *" if j <= result.n_factors else "  "
        print(f"   {mark} lambda_{j:02d} = {lam:8.4f}   share={share:6.3f}   cum={cum:6.3f}")
    print()
    print("  rotated SS / p (variance share of Lambda^*):")
    for name, ss, prop in zip(
        result.factor_names,
        result.variance_explained,
        result.proportion_variance,
        strict=True,
    ):
        print(f"    {name:10s}  SS={ss:7.3f}  prop={prop:.3f}")
    print(f"  cumulative rotated  : {float(np.sum(result.proportion_variance)):.3f}")
    print(f"  Varimax Q           : {varimax_criterion(result.rotated_loadings):.6f}")
    print(f"  T orthogonality F   : {np.linalg.norm(result.rotation_matrix.T @ result.rotation_matrix - np.eye(result.n_factors)):.2e}")
    print()
    print("  primary membership (max |loading|):")
    frame = loadings_frame(result)
    for _, row in frame.iterrows():
        print(
            f"    F{int(row['primary_factor'])}  "
            f"{row['item']:24s}  "
            f"loading={row['primary_loading']:+.3f}  "
            f"h^2={row['communality']:.3f}"
        )
    print()
    print(f"  loadings CSV        : {loadings_path}")
    print(f"  scores CSV          : {scores_path}")
    print(f"  scree PNG           : {scree_path}")
    if result.n_kaiser != result.n_factors:
        print(
            _color(
                ANSI_YELLOW,
                f"  note: Kaiser m={result.n_kaiser} differs from contest m={result.n_factors}",
            )
        )
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))


def run_efa_pipeline(
    csv_path: Path | None = None,
    loadings_path: Path | None = None,
    scores_path: Path | None = None,
    scree_path: Path | None = None,
    n_factors: int = CONTEST_N_FACTORS,
) -> EFAResult:
    """Load OriginalData.csv, extract m factors, write CSV + scree PNG.

    Parameters
    ----------
    csv_path:
        Likert wide table. Default pack ``data/raw/OriginalData.csv``.
    loadings_path:
        Default ``results/factor_loadings.csv``.
    scores_path:
        Default ``results/student_factor_scores.csv``.
    scree_path:
        Default ``paper_figures/fig_scree_plot.png``.
    n_factors:
        Contest default m = 3.
    """
    x, names = load_questionnaire_matrix(csv_path if csv_path is not None else DEFAULT_CSV)
    suit = evaluate_suitability(x, names)
    result = extract_efa(suit.z, suit.corr, names, n_factors=n_factors)
    load_out, score_out = export_efa_tables(result, loadings_path, scores_path)
    scree_out = DEFAULT_SCREE_PNG if scree_path is None else Path(scree_path)
    plot_scree(result.eigenvalues, result.n_factors, scree_out, n_kaiser=result.n_kaiser)
    print_efa_report(result, load_out, score_out, scree_out)
    return result


def main() -> int:
    """CLI entry: python -m src.factor_analysis.efa_engine"""
    run_efa_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())

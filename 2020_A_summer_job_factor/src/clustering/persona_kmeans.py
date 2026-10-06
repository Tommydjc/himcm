"""K-Means personas in the orthogonal 3-factor Thompson space.

Contest modeling choice: retain K = 3. Elbow inertias and mean silhouette
coefficients for K = 2..K_max are reported as diagnostics; they do not
silently override K = 3.

sklearn cluster ids are arbitrary. After fitting, clusters are permuted so
that Cluster 0/1/2 align as closely as possible with the three designed
persona prototypes (high F1; high F2 and lower F1; low F3). Narrative text
is then rewritten from the *extracted* axis names and the actual centroids,
because F1 on this battery is not Immediate Financial Yield.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from src.entropy_weight.entropy_engine import (
    FACTOR_COLUMNS,
    PACK_ROOT,
    load_job_factor_matrix,
    load_student_factor_scores,
)

CONTEST_K: Final[int] = 3
K_MAX: Final[int] = 8
RANDOM_STATE: Final[int] = 2020
FIGURE_DPI: Final[int] = 300

DEFAULT_STUDENT_SCORES: Final[Path] = PACK_ROOT / "results" / "student_factor_scores.csv"
DEFAULT_WEIGHTS: Final[Path] = PACK_ROOT / "results" / "entropy_weights.csv"
DEFAULT_LABELS_CSV: Final[Path] = PACK_ROOT / "results" / "student_cluster_labels.csv"
DEFAULT_CENTROIDS_CSV: Final[Path] = PACK_ROOT / "results" / "cluster_centroids.csv"
DEFAULT_KSEL_CSV: Final[Path] = PACK_ROOT / "results" / "kmeans_k_selection.csv"
DEFAULT_PERSONA_MD: Final[Path] = PACK_ROOT / "results" / "persona_profiles.md"
DEFAULT_FIGURE_3D: Final[Path] = PACK_ROOT / "paper_figures" / "fig_persona_clusters_3d.png"
DEFAULT_FIGURE_K: Final[Path] = PACK_ROOT / "paper_figures" / "fig_persona_k_selection.png"

CLUSTER_COLORS: Final[tuple[str, ...]] = ("#1f4e79", "#c0392b", "#1e8449")

CONTEST_PERSONAS: Final[tuple[tuple[int, str, str, str], ...]] = (
    (
        0,
        "务实经济型",
        "The Pragmatist",
        "Designed signature: very high F1 (cash/tips), moderate F2 and F3.",
    ),
    (
        1,
        "履历投资型",
        "The Career-Builder",
        "Designed signature: very high F2, lower F1 (tutoring / camp).",
    ),
    (
        2,
        "自由舒适型",
        "The Leisure-Seeker",
        "Designed signature: very low F3 burden, prefers easy indoor work.",
    ),
)

PROTOTYPES: Final[np.ndarray] = np.array(
    [
        [1.25, 0.00, 0.00],
        [-0.60, 1.25, 0.00],
        [0.00, 0.00, -1.25],
    ],
    dtype=np.float64,
)

ANSI_RESET: Final[str] = "\033[0m"
ANSI_CYAN: Final[str] = "\033[36m"
ANSI_BOLD: Final[str] = "\033[1m"
ANSI_YELLOW: Final[str] = "\033[33m"


@dataclass(frozen=True)
class PersonaSpec:
    """One named high-school persona after cluster permutation.

    Attributes
    ----------
    cluster_id:
        Relabeled id in {0, 1, 2}.
    name_zh, name_en:
        Contest persona titles.
    n_members:
        Count of students in the cluster.
    centroid:
        Mean F in R^3.
    extracted_reading:
        Axis-true blurb using extracted factor names and centroid signs.
    designed_signature:
        The contest geometric story (may not match this extraction).
    template_sse:
        Squared Euclidean distance from centroid to the designed prototype.
    """

    cluster_id: int
    name_zh: str
    name_en: str
    n_members: int
    centroid: np.ndarray
    extracted_reading: str
    designed_signature: str
    template_sse: float


@dataclass(frozen=True)
class PersonaClusterResult:
    """K-Means payload in the 50 x 3 factor space."""

    n_obs: int
    k: int
    scores: np.ndarray
    student_ids: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray
    inertias: np.ndarray
    silhouette_by_k: dict[int, float]
    silhouette_at_k: float
    silhouette_samples: np.ndarray
    best_silhouette_k: int
    elbow_k: int
    permutation: tuple[int, ...]
    personas: tuple[PersonaSpec, ...]
    job_ids: np.ndarray
    job_titles: tuple[str, ...]
    job_factors: np.ndarray
    factor_names: tuple[str, ...]


def _color(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{ANSI_RESET}"


def load_student_ids(csv_path: Path | None = None) -> np.ndarray:
    """student_id column aligned with the factor score rows."""
    path = DEFAULT_STUDENT_SCORES if csv_path is None else Path(csv_path)
    frame = pd.read_csv(path)
    if "student_id" not in frame.columns:
        return np.arange(1, len(frame) + 1, dtype=np.int64)
    return frame["student_id"].to_numpy(dtype=np.int64)


def load_factor_axis_names(weights_path: Path | None = None) -> tuple[str, ...]:
    """English axis names from entropy_weights.csv (extracted, not designed)."""
    path = DEFAULT_WEIGHTS if weights_path is None else Path(weights_path)
    if not path.is_file():
        return ("Factor_1", "Factor_2", "Factor_3")
    frame = pd.read_csv(path)
    if "name_en" not in frame.columns:
        return ("Factor_1", "Factor_2", "Factor_3")
    names = [str(x) for x in frame["name_en"].tolist()]
    if len(names) != 3:
        return ("Factor_1", "Factor_2", "Factor_3")
    return tuple(names)


def fit_kmeans(scores: np.ndarray, n_clusters: int, random_state: int = RANDOM_STATE) -> KMeans:
    """sklearn K-Means with k-means++ and a fixed seed."""
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    if scores.shape[0] < n_clusters:
        raise ValueError(f"N={scores.shape[0]} < K={n_clusters}")
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    model.fit(scores)
    return model


def inertia_curve(
    scores: np.ndarray,
    k_min: int = 1,
    k_max: int = K_MAX,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Within-cluster SSE for K = k_min .. k_max (length k_max-k_min+1)."""
    values = []
    for k in range(k_min, k_max + 1):
        model = fit_kmeans(scores, k, random_state=random_state)
        values.append(float(model.inertia_))
    return np.asarray(values, dtype=np.float64)


def silhouette_curve(
    scores: np.ndarray,
    k_min: int = 2,
    k_max: int = K_MAX,
    random_state: int = RANDOM_STATE,
) -> dict[int, float]:
    """Mean silhouette coefficient for each K in 2 .. k_max."""
    out: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        model = fit_kmeans(scores, k, random_state=random_state)
        out[k] = float(silhouette_score(scores, model.labels_))
    return out


def elbow_k_from_inertia(inertias: np.ndarray, k_min: int = 1) -> int:
    """Largest discrete curvature (second difference) on the inertia curve.

    For I(K), use Delta2(K) = I(K-1) - 2 I(K) + I(K+1) for interior K.
    """
    if inertias.size < 3:
        return int(k_min)
    second = inertias[:-2] - 2.0 * inertias[1:-1] + inertias[2:]
    offset = int(np.argmax(second))
    return int(k_min + 1 + offset)


def permute_labels_to_prototypes(
    centroids: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray = PROTOTYPES,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], np.ndarray]:
    """Relabel clusters so centroid i is nearest designed prototype i.

    Searches all 3! permutations; cost is sum of squared Euclidean distances.

    Returns
    -------
    new_labels, new_centroids, permutation, sse
        ``permutation[new_id] = old_id``.
    """
    k = centroids.shape[0]
    if k != prototypes.shape[0]:
        raise ValueError("centroid / prototype count mismatch")
    best_perm: tuple[int, ...] | None = None
    best_cost = float("inf")
    best_sse = np.zeros(k)
    for perm in itertools.permutations(range(k)):
        sse = np.array(
            [float(np.sum((centroids[perm[i]] - prototypes[i]) ** 2)) for i in range(k)],
            dtype=np.float64,
        )
        cost = float(sse.sum())
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
            best_sse = sse
    assert best_perm is not None
    old_to_new = {best_perm[new]: new for new in range(k)}
    new_labels = np.array([old_to_new[int(lab)] for lab in labels], dtype=np.int64)
    new_centroids = centroids[list(best_perm)]
    return new_labels, new_centroids, best_perm, best_sse


def _sign_word(value: float, lo: float = -0.35, hi: float = 0.35) -> str:
    if value >= hi:
        return "high"
    if value <= lo:
        return "low"
    return "near-average"


def extracted_persona_blurb(
    cluster_id: int,
    centroid: np.ndarray,
    factor_names: tuple[str, ...],
    n_members: int,
) -> str:
    """Describe the cluster on the extracted axes, not the cash/resume/sun story."""
    bits = [
        f"{factor_names[j]} is {_sign_word(float(centroid[j]))} "
        f"(centroid {centroid[j]:+.3f})"
        for j in range(3)
    ]
    distinctive = int(np.argmax(np.abs(centroid)))
    return (
        f"n={n_members}. On this extraction, Cluster {cluster_id} is most distinctive "
        f"on {factor_names[distinctive]}. " + "; ".join(bits) + ". "
        "F1 here is not hourly-wage yield; F2 is not resume value; F3 is burden "
        "with a positive flexibility loading in the EFA, so 'afraid of work' is "
        "only valid as a low-F3 reading, not as a designed cash-vs-leisure split."
    )


def build_personas(
    labels: np.ndarray,
    centroids: np.ndarray,
    template_sse: np.ndarray,
    factor_names: tuple[str, ...],
) -> tuple[PersonaSpec, ...]:
    """Attach contest titles plus extracted-axis readings."""
    specs: list[PersonaSpec] = []
    for cluster_id, name_zh, name_en, designed in CONTEST_PERSONAS:
        members = int(np.sum(labels == cluster_id))
        centroid = centroids[cluster_id].copy()
        specs.append(
            PersonaSpec(
                cluster_id=cluster_id,
                name_zh=name_zh,
                name_en=name_en,
                n_members=members,
                centroid=centroid,
                extracted_reading=extracted_persona_blurb(
                    cluster_id, centroid, factor_names, members
                ),
                designed_signature=designed,
                template_sse=float(template_sse[cluster_id]),
            )
        )
    return tuple(specs)


def plot_k_selection(
    inertias: np.ndarray,
    silhouette_by_k: dict[int, float],
    contest_k: int,
    elbow_k: int,
    best_sil_k: int,
    out_path: Path,
) -> Path:
    """Elbow inertia and silhouette vs K at 300 DPI."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ks_inertia = np.arange(1, inertias.size + 1)
    ks_sil = np.array(sorted(silhouette_by_k), dtype=np.int64)
    sil_vals = np.array([silhouette_by_k[int(k)] for k in ks_sil], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), dpi=FIGURE_DPI)
    axes[0].plot(ks_inertia, inertias, color="#1f4e79", marker="o", linewidth=1.8)
    axes[0].axvline(contest_k, color="#7d3c98", linestyle=":", label=rf"contest $K={contest_k}$")
    axes[0].axvline(elbow_k, color="#c0392b", linestyle="--", label=rf"curvature elbow $K={elbow_k}$")
    axes[0].set_xlabel(r"number of clusters $K$")
    axes[0].set_ylabel(r"inertia $\sum_i \|F_i - \mu_{c(i)}\|_2^2$")
    axes[0].set_title("Elbow method")
    axes[0].set_xticks(ks_inertia)
    axes[0].grid(True, axis="y", alpha=0.35)
    axes[0].legend(fontsize=8)
    axes[1].plot(ks_sil, sil_vals, color="#1e8449", marker="o", linewidth=1.8)
    axes[1].axvline(contest_k, color="#7d3c98", linestyle=":", label=rf"contest $K={contest_k}$")
    axes[1].axvline(best_sil_k, color="#c0392b", linestyle="--", label=rf"max silhouette $K={best_sil_k}$")
    axes[1].set_xlabel(r"number of clusters $K$")
    axes[1].set_ylabel("mean silhouette coefficient")
    axes[1].set_title("Silhouette")
    axes[1].set_xticks(ks_sil)
    axes[1].grid(True, axis="y", alpha=0.35)
    axes[1].legend(fontsize=8)
    fig.suptitle("K-Means model selection in the 3-factor space ($N=50$)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_clusters_3d(
    scores: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    personas: tuple[PersonaSpec, ...],
    job_factors: np.ndarray,
    job_titles: tuple[str, ...],
    factor_names: tuple[str, ...],
    out_path: Path,
) -> Path:
    """True 3-D scatter of students, centroids, and eight job anchors (300 DPI)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.2, 8.6), dpi=FIGURE_DPI)
    ax = fig.add_subplot(111, projection="3d")
    for cluster_id, persona in enumerate(personas):
        mask = labels == cluster_id
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            scores[mask, 2],
            c=CLUSTER_COLORS[cluster_id],
            s=36,
            alpha=0.82,
            depthshade=True,
            label=rf"$C_{cluster_id}$ {persona.name_en} ($n$={persona.n_members})",
            edgecolors="white",
            linewidths=0.3,
        )
        ax.scatter(
            [centroids[cluster_id, 0]],
            [centroids[cluster_id, 1]],
            [centroids[cluster_id, 2]],
            c=CLUSTER_COLORS[cluster_id],
            s=160,
            marker="*",
            edgecolors="black",
            linewidths=0.6,
            depthshade=False,
            zorder=6,
        )
    ax.scatter(
        job_factors[:, 0],
        job_factors[:, 1],
        job_factors[:, 2],
        c="#f4d03f",
        s=70,
        marker="D",
        edgecolors="black",
        linewidths=0.7,
        depthshade=False,
        label="job anchors (LLMFactor)",
        zorder=7,
    )
    for title, xyz in zip(job_titles, job_factors, strict=True):
        ax.text(xyz[0], xyz[1], xyz[2], f"  {title}", fontsize=7, color="#1c2833")
    ax.set_xlabel(rf"$F_1$  {factor_names[0]}")
    ax.set_ylabel(rf"$F_2$  {factor_names[1]}")
    ax.set_zlabel(rf"$F_3$  {factor_names[2]}")
    ax.set_title("High-school personas in orthogonal factor space (K-Means, $K=3$)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax.view_init(elev=18, azim=-55)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def labels_frame(result: PersonaClusterResult) -> pd.DataFrame:
    """One row per student: scores, cluster, persona, distance to centroid."""
    dist = np.linalg.norm(result.scores - result.centroids[result.labels], axis=1)
    persona_en = [result.personas[int(c)].name_en for c in result.labels]
    persona_zh = [result.personas[int(c)].name_zh for c in result.labels]
    return pd.DataFrame(
        {
            "student_id": result.student_ids,
            "Factor_1": result.scores[:, 0],
            "Factor_2": result.scores[:, 1],
            "Factor_3": result.scores[:, 2],
            "cluster": result.labels,
            "persona_en": persona_en,
            "persona_zh": persona_zh,
            "silhouette_i": result.silhouette_samples,
            "dist_to_centroid": dist,
        }
    )


def centroids_frame(result: PersonaClusterResult) -> pd.DataFrame:
    """Three centroid rows with contest titles and extracted names."""
    rows = []
    for spec in result.personas:
        rows.append(
            {
                "cluster": spec.cluster_id,
                "persona_en": spec.name_en,
                "persona_zh": spec.name_zh,
                "n_members": spec.n_members,
                "centroid_F1": float(spec.centroid[0]),
                "centroid_F2": float(spec.centroid[1]),
                "centroid_F3": float(spec.centroid[2]),
                "template_sse": spec.template_sse,
                "axis_F1": result.factor_names[0],
                "axis_F2": result.factor_names[1],
                "axis_F3": result.factor_names[2],
            }
        )
    return pd.DataFrame(rows)


def k_selection_frame(result: PersonaClusterResult) -> pd.DataFrame:
    """Inertia (K=1..K_max) and silhouette (K=2..K_max)."""
    rows = []
    for i, inertia in enumerate(result.inertias, start=1):
        rows.append(
            {
                "k": i,
                "inertia": float(inertia),
                "silhouette": result.silhouette_by_k.get(i, float("nan")),
                "is_contest_k": i == result.k,
                "is_elbow_k": i == result.elbow_k,
                "is_best_silhouette_k": i == result.best_silhouette_k,
            }
        )
    return pd.DataFrame(rows)


def write_persona_markdown(result: PersonaClusterResult, out_path: Path) -> Path:
    """Persona report with actual centroids and designed-template foil."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sil_map = ", ".join(
        f"K={k}: {result.silhouette_by_k[k]:.3f}" for k in sorted(result.silhouette_by_k)
    )
    lines = [
        "# High-school personas (K-Means, $K=3$)",
        "",
        "Input: Thompson scores `results/student_factor_scores.csv` "
        f"($N={result.n_obs}$, $m=3$). "
        f"Contest $K={result.k}$. "
        f"Elbow (inertia curvature) $K={result.elbow_k}$. "
        f"Max mean silhouette $K={result.best_silhouette_k}$ "
        f"(at contest $K$, silhouette $={result.silhouette_at_k:.3f}$).",
        "",
        f"Silhouette by $K$: {sil_map}.",
        "",
        "sklearn labels were permuted onto designed prototypes "
        "(Cluster 0 $\\approx$ high $F_1$, Cluster 1 $\\approx$ high $F_2$, "
        "Cluster 2 $\\approx$ low $F_3$). "
        "**Extracted axes are not the designed cash / resume / sun triad**; "
        "read the centroid numbers, not the contest nicknames, as the evidence.",
        "",
        f"| cluster | contest name | $n$ | $\\mu_{{F_1}}$ | $\\mu_{{F_2}}$ | $\\mu_{{F_3}}$ | template SSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for spec in result.personas:
        lines.append(
            f"| {spec.cluster_id} | {spec.name_zh} / {spec.name_en} | "
            f"{spec.n_members} | {spec.centroid[0]:+.3f} | {spec.centroid[1]:+.3f} | "
            f"{spec.centroid[2]:+.3f} | {spec.template_sse:.3f} |"
        )
    lines.extend(["", "## Axis names (extracted)", ""])
    for j, name in enumerate(result.factor_names, start=1):
        lines.append(f"- $F_{j}$: {name}")
    lines.extend(["", "## Cluster readings", ""])
    for spec in result.personas:
        lines.append(f"### Cluster {spec.cluster_id}: {spec.name_zh} ({spec.name_en})")
        lines.append("")
        lines.append(f"Designed story: {spec.designed_signature}")
        lines.append("")
        lines.append(spec.extracted_reading)
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def print_cluster_report(
    result: PersonaClusterResult,
    labels_path: Path,
    centroids_path: Path,
    fig3d: Path,
    figk: Path,
) -> None:
    """TTY summary."""
    bar = "=" * 72
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))
    print(_color(ANSI_BOLD + ANSI_CYAN, "  K-Means personas  |  2020 A  F in R^{50 x 3}"))
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))
    print(f"  N students          : {result.n_obs}")
    print(f"  contest K           : {result.k}")
    print(f"  elbow K (curvature) : {result.elbow_k}")
    print(f"  max silhouette K    : {result.best_silhouette_k}")
    print(f"  silhouette @ K=3    : {result.silhouette_at_k:.4f}")
    print(f"  label permutation   : old ids {result.permutation} -> new 0,1,2")
    print()
    for spec in result.personas:
        print(
            f"  C{spec.cluster_id} {spec.name_en:16s}  n={spec.n_members:2d}  "
            f"mu=({spec.centroid[0]:+.3f}, {spec.centroid[1]:+.3f}, {spec.centroid[2]:+.3f})  "
            f"SSE_proto={spec.template_sse:.3f}"
        )
        print(f"     {spec.extracted_reading}")
    if result.best_silhouette_k != result.k or result.elbow_k != result.k:
        print(
            _color(
                ANSI_YELLOW,
                f"  note: diagnostics prefer elbow K={result.elbow_k}, "
                f"silhouette K={result.best_silhouette_k}; extraction still uses K={result.k}.",
            )
        )
    print()
    print(f"  labels CSV          : {labels_path}")
    print(f"  centroids CSV       : {centroids_path}")
    print(f"  3-D figure          : {fig3d}")
    print(f"  K-selection figure  : {figk}")
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))


def run_persona_pipeline(
    student_path: Path | None = None,
    job_path: Path | None = None,
    weights_path: Path | None = None,
    labels_path: Path | None = None,
    centroids_path: Path | None = None,
    ksel_path: Path | None = None,
    persona_md_path: Path | None = None,
    figure_3d_path: Path | None = None,
    figure_k_path: Path | None = None,
    n_clusters: int = CONTEST_K,
) -> PersonaClusterResult:
    """Cluster 50 students, relabel to persona prototypes, write CSV + 300 DPI figures."""
    scores = load_student_factor_scores(student_path)
    student_ids = load_student_ids(student_path)
    if student_ids.shape[0] != scores.shape[0]:
        raise ValueError("student_id length does not match F")
    job_ids, job_factors, job_titles, job_axis_names = load_job_factor_matrix(job_path)
    factor_names = load_factor_axis_names(weights_path)
    if factor_names == ("Factor_1", "Factor_2", "Factor_3"):
        factor_names = job_axis_names

    inertias = inertia_curve(scores, k_min=1, k_max=K_MAX)
    sil_by_k = silhouette_curve(scores, k_min=2, k_max=K_MAX)
    elbow_k = elbow_k_from_inertia(inertias, k_min=1)
    best_sil_k = int(max(sil_by_k, key=lambda k: sil_by_k[k]))

    model = fit_kmeans(scores, n_clusters)
    raw_labels = model.labels_.astype(np.int64)
    raw_centroids = model.cluster_centers_.astype(np.float64)
    labels, centroids, perm, sse = permute_labels_to_prototypes(raw_centroids, raw_labels)
    sil_at_k = float(silhouette_score(scores, labels))
    sil_i = silhouette_samples(scores, labels)
    personas = build_personas(labels, centroids, sse, factor_names)

    result = PersonaClusterResult(
        n_obs=int(scores.shape[0]),
        k=int(n_clusters),
        scores=scores,
        student_ids=student_ids,
        labels=labels,
        centroids=centroids,
        inertias=inertias,
        silhouette_by_k=sil_by_k,
        silhouette_at_k=sil_at_k,
        silhouette_samples=sil_i,
        best_silhouette_k=best_sil_k,
        elbow_k=elbow_k,
        permutation=perm,
        personas=personas,
        job_ids=job_ids,
        job_titles=job_titles,
        job_factors=job_factors,
        factor_names=factor_names,
    )

    lab_path = DEFAULT_LABELS_CSV if labels_path is None else Path(labels_path)
    cen_path = DEFAULT_CENTROIDS_CSV if centroids_path is None else Path(centroids_path)
    ks_path = DEFAULT_KSEL_CSV if ksel_path is None else Path(ksel_path)
    md_path = DEFAULT_PERSONA_MD if persona_md_path is None else Path(persona_md_path)
    fig3d = DEFAULT_FIGURE_3D if figure_3d_path is None else Path(figure_3d_path)
    figk = DEFAULT_FIGURE_K if figure_k_path is None else Path(figure_k_path)
    for path in (lab_path, cen_path, ks_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    labels_frame(result).to_csv(lab_path, index=False)
    centroids_frame(result).to_csv(cen_path, index=False)
    k_selection_frame(result).to_csv(ks_path, index=False)
    write_persona_markdown(result, md_path)
    plot_clusters_3d(
        result.scores,
        result.labels,
        result.centroids,
        result.personas,
        result.job_factors,
        result.job_titles,
        result.factor_names,
        fig3d,
    )
    plot_k_selection(
        result.inertias,
        result.silhouette_by_k,
        result.k,
        result.elbow_k,
        result.best_silhouette_k,
        figk,
    )
    print_cluster_report(result, lab_path, cen_path, fig3d, figk)
    return result


def main() -> int:
    """CLI: python -m src.clustering.persona_kmeans"""
    run_persona_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())

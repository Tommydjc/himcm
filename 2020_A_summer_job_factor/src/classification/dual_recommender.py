"""LOOCV comparison: linear Softmax vs 2-layer ReLU MLP on 3 factor scores.

Features are Thompson scores F in R^{N x 3} only. The 15 Likert columns are
never loaded. Labels are GitHub ``work_choice`` ids in {0,...,7}, aligned to
score rows by sorted student_id (choice 0-index vs scores 1-index).

Model A is the CS231n-style white-box softmax from the HiMCM2020 mirror
(``softmax.py``), with W stored as (8, 3) so W[k, j] is job k's slope on
extracted factor j — not the designed wage axis.

Model B is Input(3)->Linear(16)->ReLU->Dropout(0.2)->Linear(8). Metrics are
strict leave-one-out; the NN curves are the mean over the 50 held-out folds.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.nn import functional as F

from src.entropy_weight.entropy_engine import FACTOR_COLUMNS, PACK_ROOT
from src.llm_factor.llm_factor_agent import JOB_TITLES

DEFAULT_SCORES: Final[Path] = PACK_ROOT / "results" / "student_factor_scores.csv"
DEFAULT_CHOICE: Final[Path] = PACK_ROOT / "data" / "raw" / "work_choice.csv"
DEFAULT_WEIGHTS: Final[Path] = PACK_ROOT / "results" / "entropy_weights.csv"
DEFAULT_METRICS: Final[Path] = PACK_ROOT / "results" / "model_comparison_metrics.csv"
DEFAULT_HEATMAP: Final[Path] = PACK_ROOT / "paper_figures" / "fig_softmax_weight_matrix_heatmap.png"
DEFAULT_CURVE: Final[Path] = PACK_ROOT / "paper_figures" / "fig_nn_training_loss_curve.png"
DEFAULT_W_CSV: Final[Path] = PACK_ROOT / "results" / "softmax_loocv_mean_weights.csv"
DEFAULT_CURVE_CSV: Final[Path] = PACK_ROOT / "results" / "nn_loocv_learning_curve.csv"

N_CLASSES: Final[int] = 8
N_FEATURES: Final[int] = 3
HIDDEN: Final[int] = 16
DROPOUT_P: Final[float] = 0.2
FIGURE_DPI: Final[int] = 300
RANDOM_STATE: Final[int] = 2020
LOG_CLIP: Final[float] = 1.0e-12

SOFTMAX_EPOCHS: Final[int] = 400
SOFTMAX_LR: Final[float] = 0.45
SOFTMAX_L2: Final[float] = 1.0e-3

NN_EPOCHS: Final[int] = 160
NN_LR: Final[float] = 0.04
NN_WEIGHT_DECAY: Final[float] = 1.0e-3

ANSI_RESET: Final[str] = "\033[0m"
ANSI_CYAN: Final[str] = "\033[36m"
ANSI_BOLD: Final[str] = "\033[1m"


@dataclass(frozen=True)
class FoldPrediction:
    """One LOOCV held-out student.

    Attributes
    ----------
    student_index:
        0-based row in the aligned (X, y) tables.
    y_true:
        Job id in {0,...,7}.
    proba:
        Length-8 class probabilities.
    """

    student_index: int
    y_true: int
    proba: np.ndarray


@dataclass(frozen=True)
class ModelMetrics:
    """Aggregate LOOCV metrics for one architecture."""

    name: str
    n_params: int
    top1: float
    top3: float
    log_loss: float
    n_folds: int


@dataclass(frozen=True)
class DualRecommenderResult:
    """Full Softmax vs MLP leave-one-out payload."""

    x: np.ndarray
    y: np.ndarray
    student_ids_scores: np.ndarray
    student_ids_choice: np.ndarray
    softmax_metrics: ModelMetrics
    mlp_metrics: ModelMetrics
    majority_top1: float
    mean_softmax_w: np.ndarray
    mean_softmax_b: np.ndarray
    nn_train_loss: np.ndarray
    nn_val_top1: np.ndarray
    factor_names: tuple[str, ...]
    job_titles: tuple[str, ...]


def _color(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{ANSI_RESET}"


def load_choice_labels(csv_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read ``work_choice.csv``; returns (student_id, job_choice_id)."""
    path = DEFAULT_CHOICE if csv_path is None else Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"work_choice missing: {path}")
    frame = pd.read_csv(path)
    if "job_choice_id" not in frame.columns:
        raise ValueError(f"{path} needs job_choice_id")
    id_col = "student_id" if "student_id" in frame.columns else frame.columns[0]
    ids = frame[id_col].to_numpy(dtype=np.int64)
    labels = frame["job_choice_id"].to_numpy(dtype=np.int64)
    if np.any((labels < 0) | (labels >= N_CLASSES)):
        raise ValueError(f"job_choice_id must be in 0..{N_CLASSES - 1}")
    return ids, labels


def align_features_and_labels(
    scores_path: Path | None = None,
    choice_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pair 3-D Thompson rows with work_choice by sorted student_id order.

    ``student_factor_scores.csv`` uses 1..N; ``work_choice.csv`` uses 0..N-1.
    After sorting each file there is no shared key, so alignment is positional
    (GitHub 0-index vs this pack's 1-index). Returns X (N, 3), y (N,), and both
    id vectors.
    """
    path = DEFAULT_SCORES if scores_path is None else Path(scores_path)
    if not path.is_file():
        raise FileNotFoundError(f"student factor scores missing: {path}")
    score_frame = pd.read_csv(path).sort_values("student_id", kind="mergesort")
    missing = [c for c in FACTOR_COLUMNS if c not in score_frame.columns]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    x = score_frame.loc[:, list(FACTOR_COLUMNS)].to_numpy(dtype=np.float64)
    score_ids = score_frame["student_id"].to_numpy(dtype=np.int64)
    if x.shape[1] != N_FEATURES:
        raise ValueError(f"refusing non-3-D features: {x.shape}")
    choice_ids, y = load_choice_labels(choice_path)
    order = np.argsort(choice_ids, kind="mergesort")
    choice_ids = choice_ids[order]
    y = y[order]
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"N mismatch: scores {x.shape[0]} vs choices {y.shape[0]}")
    return x, y, score_ids, choice_ids


def load_factor_names(weights_path: Path | None = None) -> tuple[str, ...]:
    """Extracted axis names from entropy_weights.csv when present."""
    path = DEFAULT_WEIGHTS if weights_path is None else Path(weights_path)
    if path.is_file():
        frame = pd.read_csv(path)
        if "name_en" in frame.columns and len(frame) >= 3:
            return tuple(str(v) for v in frame["name_en"].tolist()[:3])
    return tuple(FACTOR_COLUMNS)


def job_title_tuple() -> tuple[str, ...]:
    """Length-8 English job titles, index = class id."""
    return tuple(JOB_TITLES.get(k, f"job_{k}") for k in range(N_CLASSES))


def softmax_logits(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Z = X W^T + b, shapes (N, 3), (8, 3), (8,) -> (N, 8)."""
    return x @ weight.T + bias


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax with max subtraction."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def fit_linear_softmax(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = SOFTMAX_EPOCHS,
    step_size: float = SOFTMAX_LR,
    l2: float = SOFTMAX_L2,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-batch GD softmax, W in R^{8 x 3}, matching CS231n ``softmax.py``.

    Loss is mean cross-entropy plus (l2/2) ||W||_F^2. Bias is unregularized.
    """
    rng = np.random.default_rng(seed)
    n_obs, n_feat = x.shape
    weight = 0.01 * rng.normal(size=(N_CLASSES, n_feat))
    bias = np.zeros(N_CLASSES, dtype=np.float64)
    idx = np.arange(n_obs)
    for _ in range(epochs):
        logits = softmax_logits(x, weight, bias)
        probs = stable_softmax(logits)
        dscores = probs.copy()
        dscores[idx, y] -= 1.0
        dscores /= float(n_obs)
        d_w = dscores.T @ x + l2 * weight
        d_b = dscores.sum(axis=0)
        weight = weight - step_size * d_w
        bias = bias - step_size * d_b
    return weight, bias


class TwoLayerRecommender(nn.Module):
    """Input(3) -> Linear(16) -> ReLU -> Dropout(p) -> Linear(8).

    Softmax is applied in the loss / at inference, not as a layer that would
    drop the logit scale for log-loss.
    """

    def __init__(self, dropout_p: float = DROPOUT_P) -> None:
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, HIDDEN)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits, shape (N, 8)."""
        hidden = F.relu(self.fc1(x))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)

    def n_parameters(self) -> int:
        """Trainable scalar count (200 = 3*16+16 + 16*8+8)."""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


def fit_mlp_loocv_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: int,
    *,
    epochs: int = NN_EPOCHS,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train one MLP fold; return (proba_val, train_loss_path, val_top1_path)."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2**31 - 1))
    model = TwoLayerRecommender(DROPOUT_P)
    opt = torch.optim.Adam(model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY)
    x_tr = torch.tensor(x_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    x_va = torch.tensor(x_val.reshape(1, -1), dtype=torch.float32)
    train_loss = np.zeros(epochs, dtype=np.float64)
    val_top1 = np.zeros(epochs, dtype=np.float64)
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_tr)
        loss = F.cross_entropy(logits, y_tr)
        loss.backward()
        opt.step()
        train_loss[epoch] = float(loss.item())
        model.eval()
        with torch.no_grad():
            val_logits = model(x_va)
            pred = int(torch.argmax(val_logits, dim=1).item())
            val_top1[epoch] = float(pred == y_val)
    model.eval()
    with torch.no_grad():
        proba = F.softmax(model(x_va), dim=1).cpu().numpy().reshape(N_CLASSES)
    return proba, train_loss, val_top1


def top_k_hit(proba: np.ndarray, y_true: int, k: int) -> float:
    """1 if y_true is among the k largest probabilities."""
    order = np.argsort(proba)[::-1]
    return float(y_true in order[:k])


def nll(proba: np.ndarray, y_true: int) -> float:
    """Pointwise log-loss -ln p_y with probability clipping."""
    p = float(np.clip(proba[y_true], LOG_CLIP, 1.0))
    return float(-np.log(p))


def summarize_folds(name: str, n_params: int, folds: list[FoldPrediction]) -> ModelMetrics:
    """Mean Top-1, Top-3, and log-loss over LOOCV folds."""
    top1 = float(np.mean([top_k_hit(f.proba, f.y_true, 1) for f in folds]))
    top3 = float(np.mean([top_k_hit(f.proba, f.y_true, 3) for f in folds]))
    loss = float(np.mean([nll(f.proba, f.y_true) for f in folds]))
    return ModelMetrics(
        name=name,
        n_params=n_params,
        top1=top1,
        top3=top3,
        log_loss=loss,
        n_folds=len(folds),
    )


def majority_baseline_top1(y: np.ndarray) -> float:
    """LOOCV majority-class accuracy (train mode of the 49, test the held-out)."""
    hits = 0
    n_obs = y.shape[0]
    for i in range(n_obs):
        train = np.delete(y, i)
        values, counts = np.unique(train, return_counts=True)
        mode = int(values[int(np.argmax(counts))])
        hits += int(mode == int(y[i]))
    return float(hits / n_obs)


def run_loocv(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[list[FoldPrediction], list[FoldPrediction], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """50-fold LOOCV for both models. Returns fold lists, mean W, b, NN curves."""
    n_obs = x.shape[0]
    softmax_folds: list[FoldPrediction] = []
    mlp_folds: list[FoldPrediction] = []
    weights = np.zeros((n_obs, N_CLASSES, N_FEATURES), dtype=np.float64)
    biases = np.zeros((n_obs, N_CLASSES), dtype=np.float64)
    train_losses = np.zeros((n_obs, NN_EPOCHS), dtype=np.float64)
    val_accs = np.zeros((n_obs, NN_EPOCHS), dtype=np.float64)
    for i in range(n_obs):
        mask = np.ones(n_obs, dtype=bool)
        mask[i] = False
        x_tr, y_tr = x[mask], y[mask]
        x_te = x[i]
        y_te = int(y[i])
        weight, bias = fit_linear_softmax(x_tr, y_tr, seed=RANDOM_STATE + i)
        weights[i] = weight
        biases[i] = bias
        proba_s = stable_softmax(softmax_logits(x_te.reshape(1, -1), weight, bias))[0]
        softmax_folds.append(FoldPrediction(i, y_te, proba_s))
        proba_m, tr_path, va_path = fit_mlp_loocv_fold(
            x_tr,
            y_tr,
            x_te,
            y_te,
            seed=RANDOM_STATE + 17 * i,
        )
        mlp_folds.append(FoldPrediction(i, y_te, proba_m))
        train_losses[i] = tr_path
        val_accs[i] = va_path
    return (
        softmax_folds,
        mlp_folds,
        weights.mean(axis=0),
        biases.mean(axis=0),
        train_losses.mean(axis=0),
        val_accs.mean(axis=0),
    )


def plot_softmax_heatmap(
    weight: np.ndarray,
    job_titles: tuple[str, ...],
    factor_names: tuple[str, ...],
    out_path: Path,
) -> Path:
    """8 x 3 LOOCV-mean Softmax W heatmap at 300 DPI."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vmax = float(np.max(np.abs(weight)))
    vmax = max(vmax, 1.0e-6)
    fig, ax = plt.subplots(figsize=(8.4, 7.2), dpi=FIGURE_DPI)
    sns.heatmap(
        weight,
        vmin=-vmax,
        vmax=vmax,
        cmap="RdBu_r",
        annot=True,
        fmt=".2f",
        xticklabels=[f"$F_{j+1}$\n{factor_names[j]}" for j in range(N_FEATURES)],
        yticklabels=[f"{k} {job_titles[k]}" for k in range(N_CLASSES)],
        ax=ax,
        cbar_kws={"label": r"$W_{k j}$ (log-odds slope)"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_title(
        r"LOOCV-mean Softmax weights $W\in\mathbb{R}^{8\times 3}$"
        "\n(extracted factors; $W_{k,1}$ is not a wage slope on this battery)"
    )
    ax.set_xlabel("factor")
    ax.set_ylabel("job class $k$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_nn_curves(
    train_loss: np.ndarray,
    val_top1: np.ndarray,
    out_path: Path,
) -> Path:
    """Mean LOOCV train CE and held-out Top-1 vs epoch, 300 DPI."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, train_loss.size + 1)
    fig, ax1 = plt.subplots(figsize=(8.8, 5.0), dpi=FIGURE_DPI)
    color_loss = "#1f4e79"
    ax1.plot(epochs, train_loss, color=color_loss, linewidth=1.8, label="mean train CE")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("mean train cross-entropy", color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, axis="y", alpha=0.35)
    ax2 = ax1.twinx()
    color_acc = "#c0392b"
    ax2.plot(epochs, val_top1, color=color_acc, linewidth=1.8, label="mean LOOCV Top-1")
    ax2.set_ylabel("mean held-out Top-1", color=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(-0.05, 1.05)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper right", fontsize=8)
    ax1.set_title(
        r"2-layer ReLU MLP LOOCV learning curves ($N=50$, hidden $=16$, dropout $=0.2$)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def metrics_frame(
    softmax_m: ModelMetrics,
    mlp_m: ModelMetrics,
    majority_top1: float,
    n_obs: int,
) -> pd.DataFrame:
    """Comparison table written to ``model_comparison_metrics.csv``."""
    rows = [
        {
            "model": softmax_m.name,
            "features": "Thompson F (50 x 3)",
            "n_params": softmax_m.n_params,
            "protocol": "LOOCV",
            "n_folds": softmax_m.n_folds,
            "top1_accuracy": softmax_m.top1,
            "top3_hit_rate": softmax_m.top3,
            "log_loss": softmax_m.log_loss,
        },
        {
            "model": mlp_m.name,
            "features": "Thompson F (50 x 3)",
            "n_params": mlp_m.n_params,
            "protocol": "LOOCV",
            "n_folds": mlp_m.n_folds,
            "top1_accuracy": mlp_m.top1,
            "top3_hit_rate": mlp_m.top3,
            "log_loss": mlp_m.log_loss,
        },
        {
            "model": "majority_baseline_LOOCV",
            "features": "none (label mode of 49)",
            "n_params": 0,
            "protocol": "LOOCV",
            "n_folds": n_obs,
            "top1_accuracy": majority_top1,
            "top3_hit_rate": float("nan"),
            "log_loss": float("nan"),
        },
    ]
    return pd.DataFrame(rows)


def print_report(result: DualRecommenderResult, metrics_path: Path, heat: Path, curve: Path) -> None:
    """TTY comparison."""
    bar = "=" * 72
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))
    print(_color(ANSI_BOLD + ANSI_CYAN, "  Dual recommender LOOCV  |  Softmax vs 2-layer ReLU"))
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))
    print(f"  X shape              : {result.x.shape}  (3 factors only)")
    print(f"  y unique             : {sorted(set(int(v) for v in result.y))}")
    counts = {k: int(np.sum(result.y == k)) for k in range(N_CLASSES)}
    print(f"  class counts         : {counts}")
    print(f"  id alignment         : scores {result.student_ids_scores[0]}..{result.student_ids_scores[-1]}  "
          f"choice {result.student_ids_choice[0]}..{result.student_ids_choice[-1]} (positional)")
    print()
    for met in (result.softmax_metrics, result.mlp_metrics):
        print(
            f"  {met.name:28s}  params={met.n_params:4d}  "
            f"Top-1={met.top1:.3f}  Top-3={met.top3:.3f}  logloss={met.log_loss:.3f}"
        )
    print(f"  {'majority LOOCV':28s}  params={0:4d}  Top-1={result.majority_top1:.3f}")
    print()
    print(f"  metrics CSV          : {metrics_path}")
    print(f"  softmax heatmap      : {heat}")
    print(f"  NN curves            : {curve}")
    print(_color(ANSI_BOLD + ANSI_CYAN, bar))


def run_dual_recommender_pipeline(
    scores_path: Path | None = None,
    choice_path: Path | None = None,
    metrics_path: Path | None = None,
    heatmap_path: Path | None = None,
    curve_path: Path | None = None,
) -> DualRecommenderResult:
    """Align data, run LOOCV, write metrics CSV and 300 DPI figures."""
    x, y, score_ids, choice_ids = align_features_and_labels(scores_path, choice_path)
    factor_names = load_factor_names()
    titles = job_title_tuple()
    s_folds, m_folds, mean_w, mean_b, tr_curve, va_curve = run_loocv(x, y)
    softmax_m = summarize_folds(
        "linear_softmax",
        n_params=N_CLASSES * N_FEATURES + N_CLASSES,
        folds=s_folds,
    )
    mlp_m = summarize_folds("mlp_relu_16_dropout02", n_params=TwoLayerRecommender().n_parameters(), folds=m_folds)
    majority = majority_baseline_top1(y)
    result = DualRecommenderResult(
        x=x,
        y=y,
        student_ids_scores=score_ids,
        student_ids_choice=choice_ids,
        softmax_metrics=softmax_m,
        mlp_metrics=mlp_m,
        majority_top1=majority,
        mean_softmax_w=mean_w,
        mean_softmax_b=mean_b,
        nn_train_loss=tr_curve,
        nn_val_top1=va_curve,
        factor_names=factor_names,
        job_titles=titles,
    )
    met_path = DEFAULT_METRICS if metrics_path is None else Path(metrics_path)
    heat_path = DEFAULT_HEATMAP if heatmap_path is None else Path(heatmap_path)
    cur_path = DEFAULT_CURVE if curve_path is None else Path(curve_path)
    met_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame(softmax_m, mlp_m, majority, x.shape[0]).to_csv(met_path, index=False)
    w_csv = met_path.parent / "softmax_loocv_mean_weights.csv"
    curve_csv = met_path.parent / "nn_loocv_learning_curve.csv"
    w_frame = pd.DataFrame(mean_w, columns=list(FACTOR_COLUMNS))
    w_frame.insert(0, "job_id", np.arange(N_CLASSES))
    w_frame.insert(1, "title_en", list(titles))
    w_frame.to_csv(w_csv, index=False)
    pd.DataFrame(
        {
            "epoch": np.arange(1, tr_curve.size + 1),
            "mean_train_cross_entropy": tr_curve,
            "mean_loocv_top1": va_curve,
        }
    ).to_csv(curve_csv, index=False)
    plot_softmax_heatmap(mean_w, titles, factor_names, heat_path)
    plot_nn_curves(tr_curve, va_curve, cur_path)
    print_report(result, met_path, heat_path, cur_path)
    return result


def main() -> int:
    """CLI: python -m src.classification.dual_recommender"""
    run_dual_recommender_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())

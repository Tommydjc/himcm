#!/usr/bin/env python3
"""由 ``results/*.csv`` 出 300 DPI 图，并写出论文 ML 校验 LaTeX 片段。

数字只从 CSV 与当场 LOOCV 读取，不编造。Times New Roman 若本机缺失则回退 Times/STIX。

流水线::

    PYTHONPATH=. python -m src.ml.classifier && python -m src.ml.importance && python -m src.ml.plot_and_report

或::

    PYTHONPATH=. python -m src.ml
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from src.ml.classifier import run_loocv_dataset
from src.ml.dataset import build_inclusion_dataset

PRED_CSV: Path = PACK_ROOT / "results" / "ml_prediction_2032.csv"
AHP_CSV: Path = PACK_ROOT / "results" / "ahp_vs_ml_weights.csv"
MCDM_CSV: Path = PACK_ROOT / "results" / "brisbane_2032_ranking.csv"
FIG_PROB: Path = PACK_ROOT / "paper_figures" / "fig_ml_2032_probability.png"
FIG_DUMB: Path = PACK_ROOT / "paper_figures" / "fig_ahp_vs_ml_dumbbell.png"
TEX_OUT: Path = PACK_ROOT / "paper" / "sections" / "sec4_ml_validation.tex"


def configure_times_new_roman() -> str:
    """尽量启用 Times New Roman；返回实际族名。"""
    wanted = ("Times New Roman", "TimesNewRoman", "Times", "Nimbus Roman", "STIX")
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = "DejaVu Serif"
    for name in wanted:
        if name in available:
            chosen = name
            break
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [chosen, "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    return chosen


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def load_prediction_table(path: Path = PRED_CSV) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run python -m src.ml.classifier first")
    table = pd.read_csv(path)
    need = ["sde_name", "prob_logistic", "prob_rf", "ci_lower", "ci_upper", "ml_tier"]
    missing = [c for c in need if c not in table.columns]
    if missing:
        raise ValueError(f"{path.name} missing {missing}")
    return table


def load_ahp_ml_table(path: Path = AHP_CSV) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run python -m src.ml.importance first")
    table = pd.read_csv(path)
    need = [
        "Criterion",
        "AHP_Subjective_Weight",
        "ML_Objective_Weight",
        "Difference",
        "Bias_Interpretation",
    ]
    missing = [c for c in need if c not in table.columns]
    if missing:
        raise ValueError(f"{path.name} missing {missing}")
    return table


def plot_2032_probability(table: pd.DataFrame, dest: Path) -> None:
    """入选概率条形图 + 95% 误差棒 + P=0.50 虚线。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values("prob_logistic", ascending=False, kind="mergesort")
    names = ordered["sde_name"].astype(str).tolist()
    p_hat = ordered["prob_logistic"].to_numpy(dtype=float)
    low = ordered["ci_lower"].to_numpy(dtype=float)
    high = ordered["ci_upper"].to_numpy(dtype=float)
    yerr = np.vstack([p_hat - low, high - p_hat])
    yerr = np.clip(yerr, 0.0, None)
    x = np.arange(len(names), dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(
        x,
        p_hat,
        color="#0072B2",
        edgecolor="black",
        linewidth=0.6,
        width=0.62,
        zorder=2,
    )
    ax.errorbar(
        x,
        p_hat,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=4.0,
        zorder=3,
    )
    ax.axhline(0.50, color="#D55E00", linestyle="--", linewidth=1.3, label=r"$P=0.50$ decision line")
    ax.set_xticks(x, names)
    ax.set_ylabel(r"Bootstrap mean $P(y=1\mid X)$ (logistic)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Brisbane 2032 inclusion probability (B=500, 95% CI)")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ahp_ml_dumbbell(table: pd.DataFrame, dest: Path) -> None:
    """哑铃图：AHP 蓝点 vs ML 橙点。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values("AHP_Subjective_Weight", ascending=True, kind="mergesort")
    names = ordered["Criterion"].astype(str).tolist()
    y = np.arange(len(names), dtype=float)
    ahp = ordered["AHP_Subjective_Weight"].to_numpy(dtype=float)
    ml = ordered["ML_Objective_Weight"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for yi, a, m in zip(y, ahp, ml, strict=True):
        ax.plot([a, m], [yi, yi], color="#4d4d4d", linewidth=1.7, zorder=1)
    ax.scatter(ahp, y, s=70, color="#0072B2", zorder=2, label="AHP (subjective)")
    ax.scatter(ml, y, s=70, color="#E69F00", zorder=2, label=r"ML $|\beta|$ (objective)")
    ax.set_yticks(y, names)
    ax.set_xlabel("Normalized weight (six-criterion simplex)")
    ax.set_xlim(0.0, max(0.45, float(np.max(np.concatenate([ahp, ml]))) * 1.18))
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Dumbbell: expert AHP versus logistic criterion weights")
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _tex_escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def write_latex_section(
    pred: pd.DataFrame,
    ahp: pd.DataFrame,
    acc: float,
    auc: float,
    brier: float,
    mcdm_first: str,
    mcdm_score: float,
    dest: Path,
) -> None:
    """用 CSV/LOOCV 实测值写 booktabs 论述；不声称与数据相反的高准确率。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    by_name = pred.set_index("sde_name")
    cricket = by_name.loc["Cricket"]
    flag = by_name.loc["Flag football"]
    squash = by_name.loc["Squash"]
    ml_first = str(pred.sort_values("prob_logistic", ascending=False).iloc[0]["sde_name"])
    top_delta = ahp.sort_values("Difference", ascending=False, kind="mergesort").iloc[0]
    cosine = float(
        np.dot(ahp["AHP_Subjective_Weight"], ahp["ML_Objective_Weight"])
        / (
            np.linalg.norm(ahp["AHP_Subjective_Weight"])
            * np.linalg.norm(ahp["ML_Objective_Weight"])
        )
    )
    rows = []
    for _, rec in pred.iterrows():
        rows.append(
            f"{_tex_escape(str(rec['sde_name']))} & "
            f"{_fmt(rec['prob_logistic'])} & "
            f"{_fmt(rec['prob_rf'])} & "
            f"{_fmt(rec['ci_lower'])} & "
            f"{_fmt(rec['ci_upper'])} & "
            f"{_tex_escape(str(rec['ml_tier']))} \\\\"
        )
    pred_body = "\n".join(rows)
    w_rows = []
    for _, rec in ahp.iterrows():
        w_rows.append(
            f"{_tex_escape(str(rec['Criterion']))} & "
            f"{_fmt(rec['AHP_Subjective_Weight'])} & "
            f"{_fmt(rec['ML_Objective_Weight'])} & "
            f"{_fmt(rec['Difference'])} & "
            f"{_tex_escape(str(rec['Bias_Interpretation']))} \\\\"
        )
    w_body = "\n".join(w_rows)
    tex = rf"""% Auto-generated by src/ml/plot_and_report.py --- do not hand-edit numbers.
% Source CSVs: results/ml_prediction_2032.csv, results/ahp_vs_ml_weights.csv
% Nested under \subsection{Machine Learning Cross-Validation and Probability Prediction}
\label{{sec:ml-validation}}

We treat the locked AHP--SAW ranking as a \emph{{normative}} score (what the team
values) and the L2 logistic model as a \emph{{descriptive}} inclusion probability
(what a tiny labelled history of stay-versus-exit can support). The two exercises
share the same six IOC leaves; the classifier also includes inertia features that
are dropped before comparing \(|\beta|\) with AHP. All figures and tables in this
section are written from \texttt{{results/*.csv}} by \texttt{{src/ml/plot\_and\_report.py}}.

\subsubsection{{Leave-one-out diagnostics}}

On the sourced labelled panel (\(N=6\): four stay cases and two one-Games exits)
leave-one-out logistic predictions yield
accuracy \(={_fmt(acc)}\), ROC-AUC \(={_fmt(auc)}\), and Brier score
\(={_fmt(brier)}\) (Brier \(=\frac{{1}}{{N}}\sum_i(\hat p_i-y_i)^2\)).
A constant forecast \(\hat p=0.5\) has Brier \(0.25\); the observed Brier
\({_fmt(brier)}\) is therefore close to an uninformative baseline.
This is \emph{{not}} high LOOCV accuracy, and it does not by itself prove that the
criterion list is statistically sufficient. What it does show, honestly, is that
with six rows and eight raw columns the inclusion model is a specification check:
useful for reading \(|\beta|\) against AHP, not for claiming a validated classifier.
The random-forest LOOCV AUC on the same split is even weaker, so we keep logistic
as the interpretable primary and AHP--SAW as the decision engine.

\subsubsection{{Brisbane 2032 probabilities}}

Table~\ref{{tab:ml2032}} copies \texttt{{ml\_prediction\_2032.csv}}.
Logistic bootstrap means (\(B=500\)) are Cricket \({_fmt(cricket['prob_logistic'])}\)
(95\% CI \({_fmt(cricket['ci_lower'])}\)--\({_fmt(cricket['ci_upper'])}\), tier
\texttt{{{_tex_escape(str(cricket['ml_tier']))}}}),
Flag football \({_fmt(flag['prob_logistic'])}\)
(\({_fmt(flag['ci_lower'])}\)--\({_fmt(flag['ci_upper'])}\),
\texttt{{{_tex_escape(str(flag['ml_tier']))}}}),
and Squash \({_fmt(squash['prob_logistic'])}\)
(\({_fmt(squash['ci_lower'])}\)--\({_fmt(squash['ci_upper'])}\),
\texttt{{{_tex_escape(str(squash['ml_tier']))}}}).
Figure~\ref{{fig:ml2032}} draws the same intervals and the \(P=0.50\) line.
Every interval still covers values both below and above one-half, so no candidate
is a statistically sharp ``in'' or ``out'' call.

\begin{{table}}[htbp]
\centering
\caption{{2032 inclusion probabilities from \texttt{{ml\_prediction\_2032.csv}}
(logistic CI from bootstrap percentiles).}}
\label{{tab:ml2032}}
\begin{{tabular}}{{l r r r r l}}
\toprule
SDE & \(P_{{\mathrm{{logit}}}}\) & \(P_{{\mathrm{{RF}}}}\) & CI low & CI high & tier \\
\midrule
{pred_body}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.88\textwidth]{{fig_ml_2032_probability.png}}
\caption{{Logistic bootstrap means and 95\% intervals for Brisbane 2032 candidates.
Dashed line: \(P=0.50\). Source: \texttt{{paper\_figures/fig\_ml\_2032\_probability.png}}.}}
\label{{fig:ml2032}}
\end{{figure}}

\subsubsection{{AHP versus \(|\beta|\), and triangulation with SAW}}

Table~\ref{{tab:ahpml}} and Figure~\ref{{fig:dumbbell}} compare L1-normalized
logistic \(|\beta|\) on the six IOC leaves with the locked AHP vector restricted
to the same six names. The largest positive gap
\(\Delta w = w^{{\mathrm{{ML}}}}-w^{{\mathrm{{AHP}}}}\) is
\texttt{{{_tex_escape(str(top_delta['Criterion']))}}}
(\(\Delta w={_fmt(top_delta['Difference'])}\)):
the stay/exit labels put more mass on that leaf than the expert prior.
Youth appeal and gender parity go the other way---AHP is larger---and gender
has zero ML weight because all six labelled sports sit at a 50\% gender share.
The six-dimensional cosine between the two simplices is {_fmt(cosine)}.

Methodological triangulation is therefore only \emph{{partial}}.
AHP--SAW on the three-candidate universe ranks \textbf{{{_tex_escape(mcdm_first)}}}
first (composite \({_fmt(mcdm_score)}\) in \texttt{{brisbane\_2032\_ranking.csv}}).
The logistic bootstrap mean instead ranks \textbf{{{_tex_escape(ml_first)}}} first.
Flag football remains in the SAW gold medal and in the ML \texttt{{medium}} tier,
so both paradigms keep it above Squash; they do not jointly lock a unique 2032
add. We report that tension rather than force a ``strict first-place lock.''
Programme advice stays with the AHP--SAW ranking and the youth-weight sensitivity
already stated; the classifier is a second lens, not a trump card.

\begin{{table}}[htbp]
\centering
\caption{{Six-criterion AHP versus logistic \(|\beta|\) from
\texttt{{ahp\_vs\_ml\_weights.csv}}. Difference \(=w^{{\mathrm{{ML}}}}-w^{{\mathrm{{AHP}}}}\).}}
\label{{tab:ahpml}}
\begin{{tabular}}{{l r r r l}}
\toprule
Criterion & AHP & ML & \(\Delta w\) & reading \\
\midrule
{w_body}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{fig_ahp_vs_ml_dumbbell.png}}
\caption{{Dumbbell plot of subjective AHP weights (blue) and objective logistic
weights (orange). Source: \texttt{{paper\_figures/fig\_ahp\_vs\_ml\_dumbbell.png}}.}}
\label{{fig:dumbbell}}
\end{{figure}}
"""
    dest.write_text(tex, encoding="utf-8")


def main() -> int:
    """读 CSV、重算 LOOCV 指标、出图、写 LaTeX。"""
    font_name = configure_times_new_roman()
    pred = load_prediction_table()
    ahp = load_ahp_ml_table()
    data = build_inclusion_dataset()
    report = run_loocv_dataset(data.X, data.y)
    mcdm = pd.read_csv(MCDM_CSV)
    mcdm_first = str(mcdm.sort_values("Rank").iloc[0]["SDE_Name"])
    mcdm_score = float(mcdm.sort_values("Rank").iloc[0]["Score"])
    plot_2032_probability(pred, FIG_PROB)
    plot_ahp_ml_dumbbell(ahp, FIG_DUMB)
    write_latex_section(
        pred,
        ahp,
        report.acc_logit,
        report.auc_logit,
        report.brier_logit,
        mcdm_first,
        mcdm_score,
        TEX_OUT,
    )
    print(f"font.serif primary: {font_name}")
    print(f"wrote {FIG_PROB}")
    print(f"wrote {FIG_DUMB}")
    print(f"wrote {TEX_OUT}")
    print(
        "LOOCV logistic Accuracy={:.4f}  AUC={:.4f}  Brier={:.4f}".format(
            report.acc_logit, report.auc_logit, report.brier_logit
        )
    )
    print("LaTeX reports these values; it does not claim high LOOCV accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

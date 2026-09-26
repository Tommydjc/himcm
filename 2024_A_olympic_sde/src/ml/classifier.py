#!/usr/bin/env python3
"""小样本入席二分类：L2 Logistic（主）与浅层随机森林（对照）。

训练特征来自 ``src.ml.dataset``（IOC 六准则 + 惯性，折内 StandardScaler）。
当前有来源的标签集为 N=6（非编造的 30–40 行）。LOOCV 收集样本级 ``p̂_i``。
2032 候选概率的 95% 区间来自 B=500 的分层自助法（``random_state=42``），
不是向结果表填随机项目。

运行::

    PYTHONPATH=. python -m src.ml.classifier
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from src.indicators.loader import (
    BACKTEST_CODES,
    CANDIDATE_CODES,
    build_brisbane_candidates,
    build_sde_matrix,
    export_brisbane_candidates,
    export_sde_matrix,
)
from src.indicators.schema import INDICATORS, REQUIRED_INDICATOR_COLUMNS, validate_matrix
from src.mcda.score import RANGE_EPSILON
from src.ml.dataset import MATRIX_CSV, CANDIDATE_CSV, build_inclusion_dataset, get_loocv_folds

OUT_CSV: Path = PACK_ROOT / "results" / "ml_prediction_prob.csv"
OUT_CSV_2032: Path = PACK_ROOT / "results" / "ml_prediction_2032.csv"

LOGIT_C: float = 0.80
RF_DEPTH: int = 3
RF_TREES: int = 80
RF_LEAF: int = 2
RF_SEED: int = 42
BOOTSTRAP_B: int = 500
BOOTSTRAP_SEED: int = 42
DECISION_THRESHOLD: float = 0.50
Z_WALD: float = 1.959963984540054

FeatureBounds = dict[str, tuple[float, float, str]]


@dataclass(slots=True)
class LoocvReport:
    """留一法样本级概率与三项指标。"""

    p_logit: NDArray[np.float64]
    p_rf: NDArray[np.float64]
    y: NDArray[np.int64]
    acc_logit: float
    auc_logit: float
    brier_logit: float
    acc_rf: float
    auc_rf: float
    brier_rf: float
    confusion_logit: NDArray[np.int64]


@dataclass(slots=True)
class FittedInclusionModel:
    """兼容归因脚本的旧拟合容器（min-max 七因子）。"""

    name: str
    estimator: Any
    feature_cols: tuple[str, ...]
    bounds: FeatureBounds
    loocv_auc: float
    loocv_brier: float
    n_train: int
    jackknife_estimators: list[Any]
    jackknife_bounds: list[FeatureBounds]


def load_matrix(path: Path = MATRIX_CSV) -> pd.DataFrame:
    """读取 ``sde_matrix.csv``；缺失则现场导出。"""
    if not path.is_file():
        export_sde_matrix(build_sde_matrix(), path)
    return validate_matrix(pd.read_csv(path))


def load_candidates(path: Path = CANDIDATE_CSV) -> pd.DataFrame:
    """读取布里斯班三候选；缺失则现场导出。"""
    if not path.is_file():
        export_brisbane_candidates(build_brisbane_candidates(), path)
    return validate_matrix(pd.read_csv(path))


def inclusion_label(frame: pd.DataFrame) -> pd.Series:
    """由 ``Cohort`` 与 ``Events_Year`` 构造入席标签（归因脚本兼容）。"""
    if "Code" not in frame.columns:
        raise ValueError("frame must contain Code")
    if "Cohort" not in frame.columns or "Events_Year" not in frame.columns:
        raise ValueError("frame must contain Cohort and Events_Year")
    year = pd.to_numeric(frame["Events_Year"], errors="coerce")
    is_backtest = frame["Code"].astype(str).isin(BACKTEST_CODES)
    in_2028 = year >= 2028
    labels = np.where(is_backtest & in_2028, 1, 0).astype(int)
    return pd.Series(labels, index=frame.index, name="y_include")


def fit_feature_bounds(frame: pd.DataFrame) -> FeatureBounds:
    """在训练行上估计各因子 min/max 与方向（与 SAW 相同）。"""
    bounds: FeatureBounds = {}
    for item in INDICATORS:
        col = pd.to_numeric(frame[item.code], errors="coerce").to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(col)):
            raise ValueError(f"non-finite values in {item.code}")
        bounds[item.code] = (float(np.min(col)), float(np.max(col)), str(item.sense))
    return bounds


def transform_features(frame: pd.DataFrame, bounds: FeatureBounds) -> NDArray[np.float64]:
    """把原始因子映到 ``[0,1]``，列序与 ``INDICATORS`` 一致。"""
    cols: list[NDArray[np.float64]] = []
    for item in INDICATORS:
        if item.code not in bounds:
            raise KeyError(item.code)
        raw = pd.to_numeric(frame[item.code], errors="coerce").to_numpy(dtype=np.float64)
        col_min, col_max, sense = bounds[item.code]
        denom = (col_max - col_min) + RANGE_EPSILON
        if sense == "benefit":
            scaled = (raw - col_min) / denom
        elif sense == "cost":
            scaled = (col_max - raw) / denom
        else:
            raise ValueError(f"unknown sense {sense}")
        cols.append(np.clip(scaled, 0.0, 1.0))
    return np.column_stack(cols).astype(np.float64)


def _new_logit() -> LogisticRegression:
    """主模型：L2 逻辑回归，``P(y=1|X)=1/(1+exp(-(β0+β·x)))``。"""
    return LogisticRegression(
        penalty="l2",
        C=LOGIT_C,
        solver="liblinear",
        random_state=RF_SEED,
        max_iter=4000,
    )


def _new_forest() -> RandomForestClassifier:
    """对照：浅层森林，限制深度与叶节点最小样本。"""
    return RandomForestClassifier(
        n_estimators=RF_TREES,
        max_depth=RF_DEPTH,
        min_samples_leaf=RF_LEAF,
        random_state=RF_SEED,
        n_jobs=1,
    )


def _predict_positive(estimator: Any, x_norm: NDArray[np.float64]) -> NDArray[np.float64]:
    """返回 P(y=1|X)，形状 ``(m,)``。"""
    proba = estimator.predict_proba(x_norm)
    classes = list(estimator.classes_)
    if 1 not in classes:
        return np.zeros(x_norm.shape[0], dtype=np.float64)
    idx = classes.index(1)
    return np.asarray(proba[:, idx], dtype=np.float64)


def loocv_scores(
    raw: pd.DataFrame,
    y: NDArray[np.int64],
    factory: Any,
) -> tuple[float, float, NDArray[np.float64]]:
    """旧接口：DataFrame 七因子 + 折内 min-max（供单元测试与归因）。"""
    n = int(raw.shape[0])
    if n < 3:
        raise ValueError("LOOCV requires at least 3 labeled rows")
    from sklearn.model_selection import LeaveOneOut

    splitter = LeaveOneOut()
    oof = np.zeros(n, dtype=np.float64)
    x_index = np.arange(n)
    for train_idx, test_idx in splitter.split(x_index):
        train_frame = raw.iloc[train_idx]
        test_frame = raw.iloc[test_idx]
        y_train = y[train_idx]
        if int(np.unique(y_train).size) < 2:
            raise ValueError("a LOOCV fold lost a class; check label balance")
        bounds = fit_feature_bounds(train_frame)
        x_train = transform_features(train_frame, bounds)
        x_test = transform_features(test_frame, bounds)
        model = factory()
        model.fit(x_train, y_train)
        oof[test_idx] = _predict_positive(model, x_test)
    auc = float(roc_auc_score(y, oof))
    brier = float(brier_score_loss(y, oof))
    return auc, brier, oof


def fit_inclusion_model(
    raw: pd.DataFrame,
    y: NDArray[np.int64],
    name: str,
    factory: Any,
) -> FittedInclusionModel:
    """全样本拟合 + LOOCV（七因子 min-max），供 ``predict_candidate_probability``。"""
    auc, brier, _oof = loocv_scores(raw, y, factory)
    bounds = fit_feature_bounds(raw)
    x_all = transform_features(raw, bounds)
    estimator = factory()
    estimator.fit(x_all, y)
    jack_est: list[Any] = []
    jack_bounds: list[FeatureBounds] = []
    n = int(raw.shape[0])
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        train_frame = raw.iloc[mask]
        y_train = y[mask]
        b_i = fit_feature_bounds(train_frame)
        m_i = factory()
        m_i.fit(transform_features(train_frame, b_i), y_train)
        jack_est.append(m_i)
        jack_bounds.append(b_i)
    return FittedInclusionModel(
        name=name,
        estimator=estimator,
        feature_cols=tuple(REQUIRED_INDICATOR_COLUMNS),
        bounds=bounds,
        loocv_auc=auc,
        loocv_brier=brier,
        n_train=n,
        jackknife_estimators=jack_est,
        jackknife_bounds=jack_bounds,
    )


def _jackknife_interval(
    model: FittedInclusionModel,
    candidate_row: pd.DataFrame,
) -> tuple[float, float, float]:
    """全样本点估计 + Jackknife 标准误区间。"""
    x_full = transform_features(candidate_row, model.bounds)
    p_hat = float(_predict_positive(model.estimator, x_full)[0])
    theta = np.zeros(len(model.jackknife_estimators), dtype=np.float64)
    for i, est in enumerate(model.jackknife_estimators):
        x_i = transform_features(candidate_row, model.jackknife_bounds[i])
        theta[i] = float(_predict_positive(est, x_i)[0])
    n = float(theta.size)
    theta_bar = float(np.mean(theta))
    se = float(np.sqrt(((n - 1.0) / n) * float(np.sum((theta - theta_bar) ** 2))))
    low = float(np.clip(p_hat - Z_WALD * se, 0.0, 1.0))
    high = float(np.clip(p_hat + Z_WALD * se, 0.0, 1.0))
    if high < low:
        low, high = high, low
    return p_hat, low, high


def predict_candidate_probability(
    model: FittedInclusionModel,
    candidate_features: pd.DataFrame,
) -> pd.DataFrame:
    """旧接口：七因子候选概率。"""
    if candidate_features.shape[0] < 1:
        raise ValueError("candidate_features must have at least one row")
    missing = [c for c in REQUIRED_INDICATOR_COLUMNS if c not in candidate_features.columns]
    if missing:
        raise ValueError(f"candidate_features missing {missing}")
    rows: list[dict[str, object]] = []
    for pos in range(candidate_features.shape[0]):
        rec = candidate_features.iloc[[pos]]
        p_hat, low, high = _jackknife_interval(model, rec)
        name = str(rec["Discipline"].iloc[0]) if "Discipline" in rec.columns else str(rec["Code"].iloc[0])
        rows.append(
            {
                "Code": str(rec["Code"].iloc[0]),
                "SDE_Name": name,
                "Model": model.name,
                "P_include": p_hat,
                "CI_low": low,
                "CI_high": high,
                "N_train": int(model.n_train),
                "LOOCV_AUC": float(model.loocv_auc),
                "LOOCV_Brier": float(model.loocv_brier),
            }
        )
    return pd.DataFrame.from_records(rows)


def labeled_training_frame(matrix: pd.DataFrame) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """六项历史回测 (原表, y)。"""
    labeled = matrix.loc[matrix["Code"].astype(str).isin(BACKTEST_CODES)].copy()
    labeled = labeled.reset_index(drop=True)
    y = inclusion_label(labeled).to_numpy(dtype=np.int64)
    if int(np.unique(y).size) < 2:
        raise ValueError("training labels must contain both 0 and 1")
    return labeled, y


def assign_ml_tier(prob: float) -> str:
    """按主模型点估计划分档位。"""
    if prob >= 0.70:
        return "high"
    if prob >= 0.50:
        return "medium"
    return "low"


def run_loocv_dataset(x_raw: np.ndarray, y: np.ndarray) -> LoocvReport:
    """在 ``get_loocv_folds`` 上拟合双模型，收集 ``p̂_i``。"""
    n = int(np.asarray(y).reshape(-1).size)
    p_logit = np.zeros(n, dtype=np.float64)
    p_rf = np.zeros(n, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    for fold in get_loocv_folds(x_raw, y_arr):
        if int(np.unique(fold.y_train).size) < 2:
            raise ValueError("a LOOCV fold lost a class")
        logit = _new_logit()
        forest = _new_forest()
        logit.fit(fold.X_train, fold.y_train)
        forest.fit(fold.X_train, fold.y_train)
        idx = int(fold.val_index[0])
        p_logit[idx] = float(_predict_positive(logit, fold.X_val)[0])
        p_rf[idx] = float(_predict_positive(forest, fold.X_val)[0])
    y_hat = (p_logit >= DECISION_THRESHOLD).astype(np.int64)
    cm = confusion_matrix(y_arr, y_hat, labels=[0, 1]).astype(np.int64)
    return LoocvReport(
        p_logit=p_logit,
        p_rf=p_rf,
        y=y_arr,
        acc_logit=float(accuracy_score(y_arr, y_hat)),
        auc_logit=float(roc_auc_score(y_arr, p_logit)),
        brier_logit=float(brier_score_loss(y_arr, p_logit)),
        acc_rf=float(accuracy_score(y_arr, (p_rf >= DECISION_THRESHOLD).astype(int))),
        auc_rf=float(roc_auc_score(y_arr, p_rf)),
        brier_rf=float(brier_score_loss(y_arr, p_rf)),
        confusion_logit=cm,
    )


def _fit_scaled_models(
    x_raw: NDArray[np.float64],
    y: NDArray[np.int64],
) -> tuple[StandardScaler, LogisticRegression, RandomForestClassifier]:
    """在全部标签样本上 fit scaler 与双模型（不含候选行）。"""
    scaler = StandardScaler()
    x_s = np.asarray(scaler.fit_transform(x_raw), dtype=np.float64)
    logit = _new_logit()
    forest = _new_forest()
    logit.fit(x_s, y)
    forest.fit(x_s, y)
    return scaler, logit, forest


def bootstrap_candidate_probs(
    x_raw: NDArray[np.float64],
    y: NDArray[np.int64],
    x_cand: NDArray[np.float64],
    n_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """自助法：每次重抽样后重新 ``StandardScaler.fit``，预测候选。

    丢弃只有一个类别的抽次。返回形状 ``(n_boot, n_cand)`` 的 logistic / RF 概率。
    """
    rng = np.random.RandomState(seed)
    n = int(x_raw.shape[0])
    n_cand = int(x_cand.shape[0])
    logit_draws = np.zeros((n_boot, n_cand), dtype=np.float64)
    rf_draws = np.zeros((n_boot, n_cand), dtype=np.float64)
    filled = 0
    attempts = 0
    max_attempts = n_boot * 80
    while filled < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.choice(n, size=n, replace=True)
        y_b = y[idx]
        if int(np.unique(y_b).size) < 2:
            continue
        scaler = StandardScaler()
        x_tr = np.asarray(scaler.fit_transform(x_raw[idx]), dtype=np.float64)
        x_te = np.asarray(scaler.transform(x_cand), dtype=np.float64)
        logit = _new_logit()
        forest = _new_forest()
        logit.fit(x_tr, y_b)
        forest.fit(x_tr, y_b)
        logit_draws[filled] = _predict_positive(logit, x_te)
        rf_draws[filled] = _predict_positive(forest, x_te)
        filled += 1
    if filled < n_boot:
        raise RuntimeError(
            f"bootstrap obtained only {filled}/{n_boot} two-class resamples in {attempts} draws"
        )
    return logit_draws, rf_draws


def build_2032_table(
    names: list[str],
    logit_draws: NDArray[np.float64],
    rf_draws: NDArray[np.float64],
) -> pd.DataFrame:
    """均值概率 + logistic 的 2.5/97.5 分位区间 + 档位。"""
    rows: list[dict[str, object]] = []
    for j, name in enumerate(names):
        p_log = float(np.mean(logit_draws[:, j]))
        p_rf = float(np.mean(rf_draws[:, j]))
        low = float(np.quantile(logit_draws[:, j], 0.025))
        high = float(np.quantile(logit_draws[:, j], 0.975))
        rows.append(
            {
                "sde_name": name,
                "prob_logistic": p_log,
                "prob_rf": p_rf,
                "ci_lower": low,
                "ci_upper": high,
                "ml_tier": assign_ml_tier(p_log),
            }
        )
    return pd.DataFrame.from_records(rows)


def print_loocv_report(report: LoocvReport, codes: list[str]) -> None:
    """终端：混淆矩阵与 Accuracy / ROC-AUC / Brier。"""
    print("LOOCV (dataset features, scaler fit on training fold only)")
    print(f"  N={int(report.y.size)}  (sourced labeled SDEs; not a synthetic 30–40 panel)")
    print("  held-out rows:")
    detail = pd.DataFrame(
        {
            "Code": codes,
            "y": report.y,
            "p_logit": report.p_logit,
            "p_rf": report.p_rf,
        }
    )
    print(detail.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("  Logistic confusion matrix at p>=0.50  (rows=true 0/1, cols=pred 0/1)")
    cm = report.confusion_logit
    print(f"              pred_0  pred_1")
    print(f"     true_0   {int(cm[0, 0]):6d}  {int(cm[0, 1]):6d}")
    print(f"     true_1   {int(cm[1, 0]):6d}  {int(cm[1, 1]):6d}")
    print()
    print("  Logistic:  Accuracy={:.4f}  ROC-AUC={:.4f}  Brier={:.4f}".format(
        report.acc_logit, report.auc_logit, report.brier_logit
    ))
    print("  RandomForest: Accuracy={:.4f}  ROC-AUC={:.4f}  Brier={:.4f}".format(
        report.acc_rf, report.auc_rf, report.brier_rf
    ))
    print("  Brier = mean((p_hat - y)^2); lower is better calibration.")


def run_prediction_export(dest: Path = OUT_CSV) -> pd.DataFrame:
    """旧 CSV：七因子 Jackknife 接口，保持 ``ml_prediction_prob.csv``。"""
    matrix = load_matrix()
    candidates = load_candidates()
    candidates = candidates.loc[candidates["Code"].astype(str).isin(CANDIDATE_CODES)].copy()
    order = ["CKT", "SQU", "AFB"]
    candidates = candidates.set_index("Code").loc[order].reset_index()
    train_raw, y = labeled_training_frame(matrix)
    logit = fit_inclusion_model(train_raw, y, "l2_logistic", _new_logit)
    forest = fit_inclusion_model(train_raw, y, "random_forest_d3", _new_forest)
    table = pd.concat(
        [
            predict_candidate_probability(logit, candidates),
            predict_candidate_probability(forest, candidates),
        ],
        ignore_index=True,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False, encoding="utf-8")
    return table


def main() -> int:
    """LOOCV 报告 + 自助法 2032 概率表。"""
    data = build_inclusion_dataset()
    report = run_loocv_dataset(data.X, data.y)
    print_loocv_report(report, data.codes)
    print()
    logit_draws, rf_draws = bootstrap_candidate_probs(data.X, data.y, data.X_cand)
    table_2032 = build_2032_table(data.cand_names, logit_draws, rf_draws)
    OUT_CSV_2032.parent.mkdir(parents=True, exist_ok=True)
    table_2032.to_csv(OUT_CSV_2032, index=False, encoding="utf-8")
    print(f"wrote {OUT_CSV_2032}")
    print(table_2032.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("ci_lower/ci_upper = 2.5/97.5 percentiles of B=500 logistic bootstrap draws.")
    print("Inertia is a column in X (γ in the logit linear predictor), already scaled per fold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

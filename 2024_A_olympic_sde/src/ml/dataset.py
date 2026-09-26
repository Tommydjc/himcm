#!/usr/bin/env python3
"""无泄漏的入席学习表：IOC 六准则 + xlsx 历史惯性，折内 StandardScaler。

``historical_status`` 由 ``sde_matrix.csv``（loader 写入）读取：
``core_or_recent_add`` → y=1，``dropped_or_one_games`` → y=0。
2032 候选不进训练集。Cricket / Squash / Flag football 的惯性强制为 0。

棒垒球（BSB/SBL）在官方 xlsx 中计算惯性，但研究因子表尚无 IOC 六列，
因此不并入 ``X``（禁止用空值或编造联会会员数填表）。

运行::

    PYTHONPATH=. python -m src.ml.dataset
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from src.indicators.loader import (
    CANDIDATE_CODES,
    CORE_OR_RECENT_CODES,
    DROPPED_CODES,
    HISTORICAL_STATUS_BY_CODE,
    RAW_XLSX,
    _positive_event_count,
    _year_columns,
    build_brisbane_candidates,
    build_sde_matrix,
    export_brisbane_candidates,
    export_sde_matrix,
)
from src.indicators.schema import validate_matrix

MATRIX_CSV: Path = PACK_ROOT / "data" / "processed" / "sde_matrix.csv"
CANDIDATE_CSV: Path = PACK_ROOT / "data" / "processed" / "brisbane_candidates.csv"

IOC_FEATURE_COLS: tuple[str, ...] = (
    "GLOBAL_REACH",
    "GENDER_PARITY",
    "YOUTH_APPEAL",
    "INFRA_COST",
    "BROADCAST_VAL",
    "Events_Count",
)
INERTIA_COLS: tuple[str, ...] = ("Inertia_Streak", "Inertia_EventSum")
FEATURE_COLS: tuple[str, ...] = IOC_FEATURE_COLS + INERTIA_COLS

POSITIVE_STATUS: str = "core_or_recent_add"
NEGATIVE_STATUS: str = "dropped_or_one_games"
CANDIDATE_STATUS: str = "candidate_2032"

INERTIA_XLSX_CODES: tuple[str, ...] = (
    "ATH",
    "SWM",
    "SKB",
    "SRF",
    "BKG",
    "KTE",
    "BSB",
    "SBL",
    "CKT",
    "SQU",
    "AFB",
)


@dataclass(slots=True)
class LoocvFold:
    """单折：scaler 只在 ``X_train_raw`` 上 fit。"""

    train_index: NDArray[np.int64]
    val_index: NDArray[np.int64]
    X_train: NDArray[np.float64]
    X_val: NDArray[np.float64]
    y_train: NDArray[np.int64]
    y_val: NDArray[np.int64]
    scaler: StandardScaler


@dataclass(slots=True)
class InclusionDataset:
    """标签集与候选集共用 ``FEATURE_COLS``。"""

    codes: list[str]
    names: list[str]
    status: list[str]
    X: NDArray[np.float64]
    y: NDArray[np.int64]
    feature_cols: tuple[str, ...]
    cand_codes: list[str]
    cand_names: list[str]
    X_cand: NDArray[np.float64]
    frame: pd.DataFrame
    cand_frame: pd.DataFrame


def _assert_finite_block(block: pd.DataFrame, where: str) -> None:
    """断言无 NaN、无 inf、列为数值。"""
    numeric = block.apply(pd.to_numeric, errors="coerce")
    assert list(numeric.columns) == list(block.columns), f"{where}: column order changed"
    assert not numeric.isna().any().any(), f"{where}: NaN in {numeric.columns[numeric.isna().any()].tolist()}"
    arr = numeric.to_numpy(dtype=np.float64)
    assert np.all(np.isfinite(arr)), f"{where}: non-finite values"


def attach_historical_status(frame: pd.DataFrame) -> pd.DataFrame:
    """若缺 ``historical_status`` 则按 Code 映射补上，不覆盖已有非空值。"""
    out = frame.copy()
    mapped = out["Code"].astype(str).map(HISTORICAL_STATUS_BY_CODE).fillna("other")
    if "historical_status" not in out.columns:
        out["historical_status"] = mapped
    else:
        existing = out["historical_status"].astype(str)
        blank = existing.isin(["", "nan", "None", "<NA>"]) | out["historical_status"].isna()
        out.loc[blank, "historical_status"] = mapped.loc[blank]
    return out


def labels_from_status(status: pd.Series) -> NDArray[np.int64]:
    """``core_or_recent_add`` → 1，``dropped_or_one_games`` → 0。"""
    values: list[int] = []
    for raw in status.astype(str).tolist():
        if raw == POSITIVE_STATUS:
            values.append(1)
        elif raw == NEGATIVE_STATUS:
            values.append(0)
        else:
            raise ValueError(f"refusing to label status {raw!r} (candidates must be held out)")
    y = np.asarray(values, dtype=np.int64)
    assert set(y.tolist()).issubset({0, 1}), "labels must be binary"
    return y


def extract_inertia_table(
    xlsx_path: Path = RAW_XLSX,
    codes: tuple[str, ...] = INERTIA_XLSX_CODES,
) -> pd.DataFrame:
    """从官方 xlsx 统计连续参赛届数与累计小项数（禁止随机）。"""
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"missing contest workbook: {xlsx_path}")
    raw = pd.read_excel(xlsx_path)
    if "Code" not in raw.columns:
        raise ValueError("xlsx must contain column Code")
    raw["Sport"] = raw["Sport"].ffill()
    years = _year_columns(raw)
    year_keys = [int(str(col).replace("*", "")[:4]) for col in years]
    order = np.argsort(np.asarray(year_keys, dtype=np.int64))
    years_sorted = [years[int(i)] for i in order]
    rows: list[dict[str, object]] = []
    for code in codes:
        matched = raw.loc[raw["Code"].astype(str) == code]
        if matched.empty:
            raise ValueError(f"code {code} not found in {xlsx_path.name}")
        record = matched.iloc[0]
        present: list[bool] = []
        event_sum = 0.0
        for year_col in years_sorted:
            count = _positive_event_count(record[year_col])
            if count is None:
                present.append(False)
            else:
                present.append(True)
                event_sum += float(count)
        streak = 0
        for flag in reversed(present):
            if flag:
                streak += 1
            elif streak > 0:
                break
        rows.append(
            {
                "Code": str(code),
                "Inertia_Streak": int(streak),
                "Inertia_EventSum": float(event_sum),
            }
        )
    table = pd.DataFrame(rows)
    _assert_finite_block(table.loc[:, list(INERTIA_COLS)], "inertia xlsx")
    return table


def _zero_candidate_inertia(inertia: pd.DataFrame) -> pd.DataFrame:
    """新候选无在席红利：惯性列置 0。"""
    out = inertia.copy()
    mask = out["Code"].astype(str).isin(CANDIDATE_CODES)
    out.loc[mask, "Inertia_Streak"] = 0
    out.loc[mask, "Inertia_EventSum"] = 0.0
    return out


def _load_processed_matrix(path: Path = MATRIX_CSV) -> pd.DataFrame:
    if not path.is_file():
        export_sde_matrix(build_sde_matrix(), path)
    return attach_historical_status(validate_matrix(pd.read_csv(path)))


def _load_processed_candidates(path: Path = CANDIDATE_CSV) -> pd.DataFrame:
    if not path.is_file():
        export_brisbane_candidates(build_brisbane_candidates(), path)
    return attach_historical_status(validate_matrix(pd.read_csv(path)))


def _feature_block(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLS if c not in frame.columns]
    assert missing == [], f"missing feature columns: {missing}"
    block = frame.loc[:, list(FEATURE_COLS)].copy()
    _assert_finite_block(block, "feature block")
    return block


def build_inclusion_dataset() -> InclusionDataset:
    """合并 processed IOC 因子与 xlsx 惯性，切出训练标签集与 ``X_cand``。"""
    matrix = _load_processed_matrix()
    candidates = _load_processed_candidates()
    inertia = _zero_candidate_inertia(extract_inertia_table())
    labeled = matrix.loc[
        matrix["historical_status"].astype(str).isin({POSITIVE_STATUS, NEGATIVE_STATUS})
    ].copy()
    labeled = labeled.merge(inertia, on="Code", how="left", validate="one_to_one")
    labeled = labeled.reset_index(drop=True)
    assert set(labeled["Code"].astype(str)).issuperset(CORE_OR_RECENT_CODES)
    assert set(labeled["Code"].astype(str)).issuperset({"KTE", "BKG"})
    y = labels_from_status(labeled["historical_status"])
    assert int(np.sum(y)) == 4, "expected four positives (ATH, SWM, SKB, SRF)"
    assert int(y.size - np.sum(y)) >= 2, "expected at least Karate and Breaking as negatives"
    x_block = _feature_block(labeled)
    X = x_block.to_numpy(dtype=np.float64)
    assert X.shape == (labeled.shape[0], len(FEATURE_COLS))
    assert X.shape[0] == y.shape[0]

    cand = candidates.loc[candidates["Code"].astype(str).isin(CANDIDATE_CODES)].copy()
    cand = cand.merge(inertia, on="Code", how="left", validate="one_to_one")
    order = ["CKT", "SQU", "AFB"]
    cand = cand.set_index("Code").loc[order].reset_index()
    x_cand_block = _feature_block(cand)
    X_cand = x_cand_block.to_numpy(dtype=np.float64)
    assert X_cand.shape[1] == X.shape[1], "candidate feature width must match training X"
    assert np.allclose(cand.loc[:, list(INERTIA_COLS)].to_numpy(dtype=float), 0.0)
    baseball = inertia.loc[inertia["Code"].astype(str).isin({"BSB", "SBL"})]
    assert baseball.shape[0] == 2, "xlsx must contain Baseball and Softball codes"
    _assert_finite_block(baseball.loc[:, list(INERTIA_COLS)], "baseball inertia")

    return InclusionDataset(
        codes=labeled["Code"].astype(str).tolist(),
        names=labeled["Discipline"].astype(str).tolist(),
        status=labeled["historical_status"].astype(str).tolist(),
        X=X,
        y=y,
        feature_cols=FEATURE_COLS,
        cand_codes=cand["Code"].astype(str).tolist(),
        cand_names=cand["Discipline"].astype(str).tolist(),
        X_cand=X_cand,
        frame=labeled.reset_index(drop=True),
        cand_frame=cand.reset_index(drop=True),
    )


def get_loocv_folds(
    X: np.ndarray,
    y: np.ndarray,
) -> Iterator[LoocvFold]:
    """留一法生成器：每折在训练集上 ``StandardScaler.fit``，再变换验证行。

    禁止在循环外对全样本 ``fit``。
    """
    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    assert x_arr.ndim == 2, "X must be 2-D"
    assert x_arr.shape[0] == y_arr.shape[0], "X and y row mismatch"
    assert x_arr.shape[0] >= 3, "LOOCV needs at least 3 rows"
    assert not np.isnan(x_arr).any(), "X contains NaN before scaling"
    assert set(y_arr.tolist()).issubset({0, 1})
    splitter = LeaveOneOut()
    indices = np.arange(x_arr.shape[0], dtype=np.int64)
    for train_idx, val_idx in splitter.split(indices):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        x_train_raw = x_arr[train_idx]
        x_val_raw = x_arr[val_idx]
        scaler = StandardScaler()
        x_train = np.asarray(scaler.fit_transform(x_train_raw), dtype=np.float64)
        x_val = np.asarray(scaler.transform(x_val_raw), dtype=np.float64)
        expected_mean = np.mean(x_train_raw, axis=0)
        np.testing.assert_allclose(
            np.asarray(scaler.mean_, dtype=np.float64),
            expected_mean,
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg="StandardScaler must fit the training fold only",
        )
        assert x_train.shape[1] == x_val.shape[1] == x_arr.shape[1]
        assert x_val.shape[0] == 1
        yield LoocvFold(
            train_index=train_idx,
            val_index=val_idx,
            X_train=x_train,
            X_val=x_val,
            y_train=y_arr[train_idx],
            y_val=y_arr[val_idx],
            scaler=scaler,
        )


def main() -> int:
    """自检并打印训练/候选特征对齐报告。"""
    data = build_inclusion_dataset()
    print("labeled panel")
    print(
        pd.DataFrame(
            {
                "Code": data.codes,
                "Discipline": data.names,
                "historical_status": data.status,
                "y": data.y,
            }
        ).to_string(index=False)
    )
    print(f"X shape {data.X.shape}  feature_cols={list(data.feature_cols)}")
    print(f"X_cand shape {data.X_cand.shape}  codes={data.cand_codes}")
    n_folds = 0
    for _fold in get_loocv_folds(data.X, data.y):
        n_folds += 1
    assert n_folds == int(data.y.size)
    print(f"LOOCV folds={n_folds} (scaler fit per training fold only)")
    print("candidate inertia forced to 0:")
    print(data.cand_frame.loc[:, ["Code", *INERTIA_COLS]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

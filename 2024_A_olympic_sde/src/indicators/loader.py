#!/usr/bin/env python3
"""从赛题 xlsx 与研究因子表清洗 SDE 矩阵，并导出布里斯班 2032 候选表。

运行::

    PYTHONPATH=. python -m src.indicators.loader

写出 ``data/processed/sde_matrix.csv`` 与 ``data/processed/brisbane_candidates.csv``，
二者列头均为 ``MATRIX_EXPORT_COLUMNS``（见 ``schema.py``）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from src.indicators.schema import (
    HOST_POPULARITY_COLUMN,
    MATRIX_EXPORT_COLUMNS,
    REQUIRED_INDICATOR_COLUMNS,
    validate_matrix,
)

RAW_XLSX: Path = PACK_ROOT / "data" / "raw" / "HiMCM_Olympic_Data.xlsx"
RESEARCH_CSV: Path = PACK_ROOT / "data" / "raw" / "sde_factors_research.csv"
PROCESSED_DIR: Path = PACK_ROOT / "data" / "processed"
PROCESSED_CSV: Path = PROCESSED_DIR / "sde_matrix.csv"
BRISBANE_CSV: Path = PROCESSED_DIR / "brisbane_candidates.csv"

# 回测 6 项 + 2032 候选 3 项（Code 与官方 xlsx 三字母码一致）。
TARGET_CODES: tuple[str, ...] = (
    "ATH",
    "SWM",
    "SKB",
    "SRF",
    "BKG",
    "KTE",
    "CKT",
    "SQU",
    "AFB",
)
BACKTEST_CODES: frozenset[str] = frozenset({"ATH", "SWM", "SKB", "SRF", "BKG", "KTE"})
CANDIDATE_CODES: frozenset[str] = frozenset({"CKT", "SQU", "AFB"})
DROPPED_CODES: frozenset[str] = frozenset({"KTE", "BKG", "BSB", "SBL"})
CORE_OR_RECENT_CODES: frozenset[str] = frozenset({"ATH", "SWM", "SKB", "SRF"})

# 写入 processed 表，供 ML 标签使用（正例核心/近增，负例移出或单届）。
HISTORICAL_STATUS_BY_CODE: dict[str, str] = {
    "ATH": "core_or_recent_add",
    "SWM": "core_or_recent_add",
    "SKB": "core_or_recent_add",
    "SRF": "core_or_recent_add",
    "BKG": "dropped_or_one_games",
    "KTE": "dropped_or_one_games",
    "BSB": "dropped_or_one_games",
    "SBL": "dropped_or_one_games",
    "CKT": "candidate_2032",
    "SQU": "candidate_2032",
    "AFB": "candidate_2032",
}

RESEARCH_FACTORS: tuple[str, ...] = (
    "GLOBAL_REACH",
    "GENDER_PARITY",
    "YOUTH_APPEAL",
    "INFRA_COST",
    "BROADCAST_VAL",
)

# 布里斯班 2032 三候选：团队锁定情景分（非随机、非 IF 会员表原值）。
BRISBANE_CANDIDATE_FACTORS: dict[str, dict[str, float]] = {
    "CKT": {
        "GLOBAL_REACH": 108.0,
        "GENDER_PARITY": 45.0,
        "YOUTH_APPEAL": 8.0,
        "INFRA_COST": 5.5,
        "BROADCAST_VAL": 9.5,
        HOST_POPULARITY_COLUMN: 9.8,
    },
    "SQU": {
        "GLOBAL_REACH": 150.0,
        "GENDER_PARITY": 48.0,
        "YOUTH_APPEAL": 6.5,
        "INFRA_COST": 3.0,
        "BROADCAST_VAL": 6.0,
        HOST_POPULARITY_COLUMN: 7.5,
    },
    "AFB": {
        "GLOBAL_REACH": 75.0,
        "GENDER_PARITY": 50.0,
        "YOUTH_APPEAL": 8.5,
        "INFRA_COST": 3.5,
        "BROADCAST_VAL": 7.5,
        HOST_POPULARITY_COLUMN: 6.5,
    },
}


def _year_columns(frame: pd.DataFrame) -> list[object]:
    """xlsx 中表示奥运年份的列（含 ``1906*`` 届间运动会）。"""
    years: list[object] = []
    for col in frame.columns:
        text = str(col).replace("*", "")[:4]
        if text.isdigit() and 1896 <= int(text) <= 2032:
            years.append(col)
    return years


def _positive_event_count(value: object) -> float | None:
    """把单元格解析为正的小项数；bullet/空为未正式设项。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"", "•", "*", "nan"}:
            return None
        try:
            number = float(stripped)
        except ValueError:
            return None
        return number if number > 0.0 else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number) or number <= 0.0:
        return None
    return float(number)


def extract_history(xlsx_path: Path) -> pd.DataFrame:
    """从官方历史底表提取 Appearances 与最新一届 Events_Count。"""
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"missing contest workbook: {xlsx_path}")
    raw = pd.read_excel(xlsx_path)
    if "Code" not in raw.columns:
        raise ValueError("xlsx must contain column Code")
    raw["Sport"] = raw["Sport"].ffill()
    years = _year_columns(raw)
    rows: list[dict[str, object]] = []
    for code in TARGET_CODES:
        matched = raw.loc[raw["Code"].astype(str) == code]
        if matched.empty:
            raise ValueError(f"code {code} not found in {xlsx_path.name}")
        record = matched.iloc[0]
        discipline = record["Discipline"]
        if pd.isna(discipline) or str(discipline).strip() == "":
            discipline = record["Sport"]
        appearances = 0
        events_latest: float | None = None
        latest_year = -1
        for year_col in years:
            count = _positive_event_count(record[year_col])
            if count is None:
                continue
            appearances += 1
            year_key = int(str(year_col).replace("*", "")[:4])
            if year_key >= latest_year:
                latest_year = year_key
                events_latest = count
        if events_latest is None:
            events_latest = 0.0
        cohort = "backtest" if code in BACKTEST_CODES else "candidate_2032"
        if code not in CANDIDATE_CODES and code not in BACKTEST_CODES:
            cohort = "other"
        rows.append(
            {
                "Code": str(code),
                "Sport": str(record["Sport"]),
                "Discipline": str(discipline),
                "GoverningBody": str(record.get("Sports Governing Body", "")),
                "Appearances": int(appearances),
                "Events_Count": float(events_latest),
                "Events_Year": int(latest_year) if latest_year > 0 else pd.NA,
                "Cohort": cohort,
                "historical_status": HISTORICAL_STATUS_BY_CODE.get(str(code), "other"),
            }
        )
    return pd.DataFrame(rows)


def load_research_factors(csv_path: Path) -> pd.DataFrame:
    """读取联会会员数与 Saaty 研究赋分（禁止随机生成）。"""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"missing {csv_path}; provide IF statistics in data/raw/sde_factors_research.csv"
        )
    research = pd.read_csv(csv_path)
    if "Code" not in research.columns:
        raise ValueError("research CSV must contain Code")
    missing = [c for c in RESEARCH_FACTORS if c not in research.columns]
    if missing:
        raise ValueError(f"research CSV missing factor columns: {missing}")
    keep = ["Code", *RESEARCH_FACTORS]
    return research.loc[:, keep].copy()


def apply_brisbane_factor_overrides(frame: pd.DataFrame) -> pd.DataFrame:
    """用布里斯班 2032 情景分覆盖三候选的 IOC 因子与 ``HOST_POPULARITY``。"""
    out = frame.copy()
    if HOST_POPULARITY_COLUMN not in out.columns:
        out[HOST_POPULARITY_COLUMN] = pd.NA
    numeric_cols = [*RESEARCH_FACTORS, HOST_POPULARITY_COLUMN]
    for column in numeric_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    for code, factors in BRISBANE_CANDIDATE_FACTORS.items():
        mask = out["Code"].astype(str) == code
        if not bool(mask.any()):
            continue
        for column, value in factors.items():
            out.loc[mask, column] = float(value)
    return out


def align_export_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """按 ``MATRIX_EXPORT_COLUMNS`` 对齐列头；缺 ``HOST_POPULARITY`` 则补空列。"""
    out = frame.copy()
    if HOST_POPULARITY_COLUMN not in out.columns:
        out[HOST_POPULARITY_COLUMN] = pd.NA
    missing = [c for c in MATRIX_EXPORT_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"cannot export; missing columns: {missing}")
    return out.loc[:, list(MATRIX_EXPORT_COLUMNS)].copy()


def build_sde_matrix(xlsx_path: Path = RAW_XLSX, research_path: Path = RESEARCH_CSV) -> pd.DataFrame:
    """合并历史出场与 IOC 因子，覆盖布里斯班候选情景分，校验后返回。"""
    history = extract_history(xlsx_path)
    research = load_research_factors(research_path)
    merged = history.merge(research, on="Code", how="left", validate="one_to_one")
    ordered = merged.set_index("Code").loc[list(TARGET_CODES)].reset_index()
    overridden = apply_brisbane_factor_overrides(ordered)
    aligned = align_export_columns(overridden)
    return validate_matrix(aligned)


def build_brisbane_candidates(
    xlsx_path: Path = RAW_XLSX,
    research_path: Path = RESEARCH_CSV,
) -> pd.DataFrame:
    """只保留 Cricket / Squash / Flag football 三行，列头与 ``sde_matrix`` 完全一致。"""
    matrix = build_sde_matrix(xlsx_path=xlsx_path, research_path=research_path)
    order = [code for code in ("CKT", "SQU", "AFB") if code in CANDIDATE_CODES]
    candidates = matrix.set_index("Code").loc[order].reset_index()
    aligned = align_export_columns(candidates)
    if aligned[HOST_POPULARITY_COLUMN].isna().any():
        raise ValueError("Brisbane candidates must have HOST_POPULARITY for every row")
    return validate_matrix(aligned)


def export_processed_csv(frame: pd.DataFrame, dest: Path) -> Path:
    """确保 ``data/processed/`` 存在后写入 utf-8 CSV。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False, encoding="utf-8")
    return dest


def export_sde_matrix(
    frame: pd.DataFrame,
    dest: Path = PROCESSED_CSV,
) -> Path:
    """写入历史/全量 processed 矩阵。"""
    return export_processed_csv(align_export_columns(frame), dest)


def export_brisbane_candidates(
    frame: pd.DataFrame,
    dest: Path = BRISBANE_CSV,
) -> Path:
    """写入布里斯班 2032 候选表。"""
    return export_processed_csv(align_export_columns(frame), dest)


def main() -> int:
    """命令行入口：清洗、校验、同时导出历史矩阵与布里斯班候选表。"""
    matrix = build_sde_matrix()
    candidates = build_brisbane_candidates()
    matrix_path = export_sde_matrix(matrix)
    candidates_path = export_brisbane_candidates(candidates)
    print("sde_matrix export report")
    print(f"  path: {matrix_path}")
    print(f"  rows: {matrix.shape[0]}")
    print(f"  columns ({matrix.shape[1]}): {list(matrix.columns)}")
    print(f"  indicator fields: {list(REQUIRED_INDICATOR_COLUMNS)}")
    print(
        matrix.loc[:, ["Code", "Discipline", "Cohort", *REQUIRED_INDICATOR_COLUMNS]].to_string(
            index=False
        )
    )
    print()
    print("brisbane_candidates export report")
    print(f"  path: {candidates_path}")
    print(f"  rows: {candidates.shape[0]}")
    print(f"  columns match sde_matrix: {list(candidates.columns) == list(matrix.columns)}")
    show = ["Code", "Discipline", *REQUIRED_INDICATOR_COLUMNS, HOST_POPULARITY_COLUMN]
    print(candidates.loc[:, show].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

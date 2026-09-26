"""因子元数据与因子矩阵校验。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import pandas as pd

Sense = Literal["benefit", "cost"]
Scale = Literal["quantitative", "saaty_1_9"]

REQUIRED_ID_COLUMNS: tuple[str, ...] = ("Discipline", "Code")

# 管道导出列：xlsx 派生 + 研究表 IOC 因子。
REQUIRED_INDICATOR_COLUMNS: tuple[str, ...] = (
    "Appearances",
    "Events_Count",
    "GLOBAL_REACH",
    "GENDER_PARITY",
    "YOUTH_APPEAL",
    "INFRA_COST",
    "BROADCAST_VAL",
)

# 2032 东道主情景列；与 AHP 七准则解耦，不进入 ``INDICATORS``。
HOST_POPULARITY_COLUMN: str = "HOST_POPULARITY"

# ``sde_matrix.csv`` 与 ``brisbane_candidates.csv`` 共用表头（顺序固定）。
MATRIX_EXPORT_COLUMNS: tuple[str, ...] = (
    "Code",
    "Sport",
    "Discipline",
    "GoverningBody",
    "Appearances",
    "Events_Count",
    "Events_Year",
    "Cohort",
    "historical_status",
    "GLOBAL_REACH",
    "GENDER_PARITY",
    "YOUTH_APPEAL",
    "INFRA_COST",
    "BROADCAST_VAL",
    HOST_POPULARITY_COLUMN,
)


@dataclass(frozen=True, slots=True)
class Indicator:
    """单个评价因子。

    Parameters
    ----------
    code :
        与数据表列名一致的英文代号。
    sense :
        ``benefit`` 越大越好；``cost`` 越小越好。
    ioc_criterion :
        对应 IOC 准则（可空）。
    scale :
        定量或 Saaty 1–9。
    unit :
        物理单位；Saaty 用 ``saaty``。
    source :
        可追溯出处。
    """

    code: str
    sense: Sense
    ioc_criterion: str = ""
    scale: Scale = "quantitative"
    unit: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if self.sense not in ("benefit", "cost"):
            raise ValueError(f"invalid sense {self.sense!r}")
        if self.scale not in ("quantitative", "saaty_1_9"):
            raise ValueError(f"invalid scale {self.scale!r}")


INDICATORS: tuple[Indicator, ...] = (
    Indicator(
        code="Appearances",
        sense="benefit",
        ioc_criterion="Popularity and Accessibility",
        scale="quantitative",
        unit="games",
        source="HiMCM_Olympic_Data.xlsx",
    ),
    Indicator(
        code="Events_Count",
        sense="cost",
        ioc_criterion="Popularity and Accessibility",
        scale="quantitative",
        unit="events",
        source="HiMCM_Olympic_Data.xlsx (latest programmed Games with a numeric cell)",
    ),
    Indicator(
        code="GLOBAL_REACH",
        sense="benefit",
        ioc_criterion="Inclusivity",
        scale="quantitative",
        unit="countries",
        source="data/raw/sde_factors_research.csv",
    ),
    Indicator(
        code="GENDER_PARITY",
        sense="benefit",
        ioc_criterion="Gender Equity",
        scale="quantitative",
        unit="percent",
        source="data/raw/sde_factors_research.csv",
    ),
    Indicator(
        code="YOUTH_APPEAL",
        sense="benefit",
        ioc_criterion="Relevance and Innovation",
        scale="saaty_1_9",
        unit="saaty",
        source="data/raw/sde_factors_research.csv",
    ),
    Indicator(
        code="INFRA_COST",
        sense="cost",
        ioc_criterion="Sustainability",
        scale="saaty_1_9",
        unit="saaty",
        source="data/raw/sde_factors_research.csv",
    ),
    Indicator(
        code="BROADCAST_VAL",
        sense="benefit",
        ioc_criterion="Popularity and Accessibility",
        scale="saaty_1_9",
        unit="saaty",
        source="data/raw/sde_factors_research.csv",
    ),
)

HOST_POPULARITY_INDICATOR: Indicator = Indicator(
    code=HOST_POPULARITY_COLUMN,
    sense="benefit",
    ioc_criterion="Popularity and Accessibility",
    scale="quantitative",
    unit="score_0_10",
    source="Brisbane 2032 host-popularity scenario in src/indicators/loader.py",
)

INDICATOR_BY_CODE: dict[str, Indicator] = {ind.code: ind for ind in INDICATORS}


def validate_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """校验因子表：必需列齐全，无空值、无负数；Saaty 列落在 [1, 9]（``BROADCAST_VAL`` 允许到 10）。

    Parameters
    ----------
    df :
        行 = SDE。必须含 ``REQUIRED_INDICATOR_COLUMNS``。

    Returns
    -------
    pd.DataFrame
        输入的副本。

    Raises
    ------
    TypeError, ValueError
        类型不对或表不满足契约。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be pandas.DataFrame, got {type(df)!r}")
    if df.shape[0] < 1:
        raise ValueError("factor matrix must have at least one row (SDE)")
    missing_id = [c for c in REQUIRED_ID_COLUMNS if c not in df.columns]
    if missing_id:
        raise ValueError(f"missing id columns: {missing_id}")
    missing = [c for c in REQUIRED_INDICATOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required indicator columns: {missing}")

    sub = df.loc[:, list(REQUIRED_INDICATOR_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if sub.isna().any().any():
        bad = sub.columns[sub.isna().any()].tolist()
        raise ValueError(f"invalid nulls or non-numeric values in: {bad}")
    if (sub < 0).any().any():
        bad = sub.columns[(sub < 0).any()].tolist()
        raise ValueError(f"negative values not allowed in indicator columns: {bad}")

    for code in REQUIRED_INDICATOR_COLUMNS:
        spec = INDICATOR_BY_CODE[code]
        col = sub[code]
        if spec.scale == "saaty_1_9":
            # BROADCAST_VAL 允许 1–10 情景分（布里斯班候选可出现 9.5）；其余 Saaty 列仍为 [1, 9]。
            upper = 10.0 if code == "BROADCAST_VAL" else 9.0
            if (col < 1.0).any() or (col > upper).any():
                raise ValueError(f"{code} must lie in [1, {upper:g}]")
        if code == "GENDER_PARITY" and (col > 100.0).any():
            raise ValueError("GENDER_PARITY is a percent in [0, 100]")

    if HOST_POPULARITY_COLUMN in df.columns:
        host = pd.to_numeric(df[HOST_POPULARITY_COLUMN], errors="coerce")
        filled = host.notna()
        if filled.any():
            vals = host.loc[filled]
            if (vals < 0.0).any() or (vals > 10.0).any():
                raise ValueError("HOST_POPULARITY must lie in [0, 10] when provided")

    return df.copy()

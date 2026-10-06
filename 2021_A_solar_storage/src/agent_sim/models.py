r"""Pydantic output contract for household load-shedding negotiation."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class LoadSheddingAgreement(BaseModel):
    """Structured family consensus produced by the EMS crew (Task 3)."""

    consensus_reached: bool = Field(description="是否达成全家共识")
    shed_appliances: List[str] = Field(
        description="决定关停或推迟的家电名称列表"
    )
    hvac_temp_offset_c: float = Field(
        description="暖气调低度数, 例如 2.5"
    )
    negotiated_reduction_kW: float = Field(
        description="本次协商实际削减的总千瓦数"
    )
    rationale: str = Field(description="家庭妥协背后的核心逻辑简述")

    @field_validator("hvac_temp_offset_c")
    @classmethod
    def _hvac_non_negative(cls, value: float) -> float:
        """Offset is degrees *lowered*, so it must be >= 0."""

        return max(float(value), 0.0)

    @field_validator("negotiated_reduction_kW")
    @classmethod
    def _cut_non_negative(cls, value: float) -> float:
        return max(float(value), 0.0)

    @field_validator("shed_appliances")
    @classmethod
    def _strip_names(cls, value: List[str]) -> List[str]:
        return [str(item).strip() for item in value if str(item).strip()]

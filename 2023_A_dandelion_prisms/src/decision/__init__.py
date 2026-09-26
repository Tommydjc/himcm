"""影响因子（官方 Req 2）与割草/PRISMS 权衡（课程扩展）。"""

from src.decision.impact import cover_to_spread_score, score_species
from src.decision.pareto import PolicyPoint, mowing_cost, nondominated
from src.decision.tradeoff_model import (
    EcoWeights,
    StrategyScore,
    nondominated_benefit_cost,
    run_four_strategies,
)

__all__ = [
    "EcoWeights",
    "PolicyPoint",
    "StrategyScore",
    "cover_to_spread_score",
    "mowing_cost",
    "nondominated",
    "nondominated_benefit_cost",
    "run_four_strategies",
    "score_species",
]

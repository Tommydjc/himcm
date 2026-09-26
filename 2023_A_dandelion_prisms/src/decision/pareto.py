"""割草双目标：残存盖度 vs 相对成本。

.. math::

    C=n_{\\mathrm{mow}}\\,(1+\\tfrac12\\eta),
    \\qquad
    J=(\\mathrm{cover}_{12},\\,C).

非支配：不存在另一政策在两项上都不差且至少一项严格更好。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PolicyPoint:
    """一条割草政策的目标值。"""

    interval_weeks: int
    eta: float
    n_mows: int
    cover: float
    cost: float
    n_plants: float


def mowing_cost(n_mows: int, eta: float) -> float:
    """相对成本 \(C=n(1+\\eta/2)\)，无货币单位。"""
    if n_mows < 0:
        raise ValueError("n_mows must be nonnegative")
    intensity = min(max(float(eta), 0.0), 1.0)
    return float(n_mows) * (1.0 + 0.5 * intensity)


def nondominated(points: list[PolicyPoint]) -> list[PolicyPoint]:
    """二维最小化帕累托集，按 cost 升序。"""
    front: list[PolicyPoint] = []
    for cand in points:
        dominated = False
        for other in points:
            if other is cand:
                continue
            le = other.cover <= cand.cover + 1.0e-15 and other.cost <= cand.cost + 1.0e-15
            sl = other.cover < cand.cover - 1.0e-15 or other.cost < cand.cost - 1.0e-15
            if le and sl:
                dominated = True
                break
        if not dominated:
            front.append(cand)
    return sorted(front, key=lambda p: (p.cost, p.cover))

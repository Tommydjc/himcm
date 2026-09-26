"""环境层：建筑拓扑图与灾害动力学（与规划器 / RL 解耦）。"""

from .hazards import HazardModel
from .layouts import (
    get_daycare_layout,
    get_office_layout,
    get_warehouse_layout,
    render_layout,
)

__all__ = [
    "HazardModel",
    "get_daycare_layout",
    "get_office_layout",
    "get_warehouse_layout",
    "render_layout",
]

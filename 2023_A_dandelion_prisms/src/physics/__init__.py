"""微观涡环沉降与风散核（WALD 羽流 + 混合高斯/上升核）。"""

from src.physics.dispersal_kernel import (
    GerminationParams,
    GridConfig,
    PappusAeroParams,
    build_hybrid_kernel,
    calculate_germination_modifier,
    compute_effective_seed_flux,
    compute_fall_time,
    deposit_hybrid_from_point,
    evaluate_bivariate_kernel,
)
from src.physics.plume import (
    build_plume_kernel,
    deposit_from_point,
    kernel_second_moment,
    rotate_offsets,
)
from src.physics.wald import wald_mean_shape, wald_pdf

__all__ = [
    "GerminationParams",
    "GridConfig",
    "PappusAeroParams",
    "build_hybrid_kernel",
    "build_plume_kernel",
    "calculate_germination_modifier",
    "compute_effective_seed_flux",
    "compute_fall_time",
    "deposit_from_point",
    "deposit_hybrid_from_point",
    "evaluate_bivariate_kernel",
    "kernel_second_moment",
    "rotate_offsets",
    "wald_mean_shape",
    "wald_pdf",
]

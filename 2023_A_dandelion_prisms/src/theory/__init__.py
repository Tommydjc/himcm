"""Fisher–KPP 连续对照。"""

from src.theory.fisher_kpp import (
    diffusion_from_hybrid_kernel,
    diffusion_from_kernel,
    diffusion_from_second_moment,
    diffusion_from_wind_sigma,
    intrinsic_rate,
    intrinsic_rate_daily,
    regress_front_speed,
    relative_wave_error,
    solve_fisher_1d,
    traveling_wave_speed,
)

__all__ = [
    "diffusion_from_hybrid_kernel",
    "diffusion_from_kernel",
    "diffusion_from_second_moment",
    "diffusion_from_wind_sigma",
    "intrinsic_rate",
    "intrinsic_rate_daily",
    "regress_front_speed",
    "relative_wave_error",
    "solve_fisher_1d",
    "traveling_wave_speed",
]

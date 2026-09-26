"""生物物理与气候契约的只读加载器。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACK_ROOT: Path = Path(__file__).resolve().parents[1]
PARAM_DIR: Path = PACK_ROOT / "data" / "parameters"


@dataclass(frozen=True, slots=True)
class Biophysics:
    """冠毛阻力、生活史与格网契约。

    Parameters
    ----------
    H_m :
        释放高度 \(H\)（m）。
    v_t_mps :
        冠毛终端沉降速度 \(v_t\)（m/s）。
    kappa_turb :
        湍流强度 \(\kappa=\sigma_w/\bar U\)（无量纲）。
    sigma0_m, gamma_plume :
        侧向羽流 \(\sigma_c(x')=\sigma_0+\gamma\sqrt{x'_+}\) 的截距与系数。
    u_floor_mps :
        风速地板，避免 WALD 的 \(\mu\to 0\)。
    n_seed_per_capitulum :
        单头落籽数（粒/头）。
    n_capitula_per_adult_week_bloom :
        花期每周每成株头数。
    local_retention :
        就近滞留比例 \(\varepsilon\in[0,1]\)，其余进羽流核。
    K_max_per_m2 :
        自疏上限 \(K\)（株/m²，莲座+成株）。
    K_smi_exp :
        \(K_{\mathrm{eff}}=K\cdot\mathrm{SMI}^{a}\)。
    T_min_c, T_opt_c, T_sigma_c :
        温度窗（℃）。
    smi_hill, smi_hill_n :
        水分 Hill 半饱和与指数。
    surv_*_week, g_*_base :
        周存活与基线阶段转移。
    source_x_m, source_y_m :
        地块外绒球源点（m），\(x<0\) 表示西缘外侧。
    n_seed_initial_puffball :
        第 0 周一次释放的籽数。
    source_persists :
        源株是否按花期持续供种。
    grid_n, dx_m :
        格点数与间距；默认 \(100\times 1\,\mathrm{m}=1\,\mathrm{ha}\)。
    kernel_half_m :
        核半宽（m）。
    weeks_per_year :
        年周数，默认 52。
    """

    H_m: float
    v_t_mps: float
    kappa_turb: float
    sigma0_m: float
    gamma_plume: float
    u_floor_mps: float
    n_seed_per_capitulum: float
    n_capitula_per_adult_week_bloom: float
    local_retention: float
    K_max_per_m2: float
    K_smi_exp: float
    T_min_c: float
    T_opt_c: float
    T_sigma_c: float
    smi_hill: float
    smi_hill_n: float
    surv_seed_week: float
    surv_seedling_week: float
    surv_rosette_week: float
    surv_adult_week: float
    g_seed_base: float
    g_seedling_base: float
    g_rosette_base: float
    source_x_m: float
    source_y_m: float
    n_seed_initial_puffball: float
    source_persists: bool
    grid_n: int
    dx_m: float
    kernel_half_m: int
    weeks_per_year: int


@dataclass(frozen=True, slots=True)
class MonthForcing:
    """单月情景强迫。"""

    month: int
    T_c: float
    wind_mps: float
    wind_from_deg: float
    smi: float


@dataclass(frozen=True, slots=True)
class ClimateSpec:
    """一种气候的花期与 12 个月强迫。"""

    name: str
    label: str
    bloom_months: tuple[int, ...]
    months: tuple[MonthForcing, ...]


def load_biophysics(path: Path | None = None) -> Biophysics:
    """读取 ``biophysics.json`` 并校验正性约束。"""
    target = path or (PARAM_DIR / "biophysics.json")
    raw: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    params = Biophysics(
        H_m=float(raw["H_m"]),
        v_t_mps=float(raw["v_t_mps"]),
        kappa_turb=float(raw["kappa_turb"]),
        sigma0_m=float(raw["sigma0_m"]),
        gamma_plume=float(raw["gamma_plume"]),
        u_floor_mps=float(raw["u_floor_mps"]),
        n_seed_per_capitulum=float(raw["n_seed_per_capitulum"]),
        n_capitula_per_adult_week_bloom=float(raw["n_capitula_per_adult_week_bloom"]),
        local_retention=float(raw["local_retention"]),
        K_max_per_m2=float(raw["K_max_per_m2"]),
        K_smi_exp=float(raw["K_smi_exp"]),
        T_min_c=float(raw["T_min_c"]),
        T_opt_c=float(raw["T_opt_c"]),
        T_sigma_c=float(raw["T_sigma_c"]),
        smi_hill=float(raw["smi_hill"]),
        smi_hill_n=float(raw["smi_hill_n"]),
        surv_seed_week=float(raw["surv_seed_week"]),
        surv_seedling_week=float(raw["surv_seedling_week"]),
        surv_rosette_week=float(raw["surv_rosette_week"]),
        surv_adult_week=float(raw["surv_adult_week"]),
        g_seed_base=float(raw["g_seed_base"]),
        g_seedling_base=float(raw["g_seedling_base"]),
        g_rosette_base=float(raw["g_rosette_base"]),
        source_x_m=float(raw["source_x_m"]),
        source_y_m=float(raw["source_y_m"]),
        n_seed_initial_puffball=float(raw["n_seed_initial_puffball"]),
        source_persists=bool(raw["source_persists"]),
        grid_n=int(raw["grid_n"]),
        dx_m=float(raw["dx_m"]),
        kernel_half_m=int(raw["kernel_half_m"]),
        weeks_per_year=int(raw["weeks_per_year"]),
    )
    if params.H_m <= 0.0 or params.v_t_mps <= 0.0:
        raise ValueError("H_m and v_t_mps must be positive")
    if not 0.0 <= params.local_retention <= 1.0:
        raise ValueError("local_retention must lie in [0, 1]")
    if params.grid_n <= 0 or params.dx_m <= 0.0:
        raise ValueError("grid_n and dx_m must be positive")
    return params


def load_climates(path: Path | None = None) -> dict[str, ClimateSpec]:
    """读取三气候月表。"""
    target = path or (PARAM_DIR / "climate_monthly.json")
    raw: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    out: dict[str, ClimateSpec] = {}
    for name, block in raw["climates"].items():
        months = tuple(
            MonthForcing(
                month=int(row["month"]),
                T_c=float(row["T_c"]),
                wind_mps=float(row["wind_mps"]),
                wind_from_deg=float(row["wind_from_deg"]),
                smi=float(row["smi"]),
            )
            for row in block["months"]
        )
        if len(months) != 12:
            raise ValueError(f"{name} must have 12 months")
        bloom = tuple(int(m) for m in block["bloom_months"])
        out[name] = ClimateSpec(
            name=name,
            label=str(block["label"]),
            bloom_months=bloom,
            months=months,
        )
    return out


def load_impact_contract(path: Path | None = None) -> dict[str, Any]:
    """读取官方 Req 2 的物种属性与 SAW 权。"""
    target = path or (PARAM_DIR / "impact_species.json")
    return json.loads(target.read_text(encoding="utf-8"))

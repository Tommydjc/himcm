r"""Household multi-agent EMS: butler warning + two-role bargain + DSR.

Protocol (deterministic, contest-reproducible; no live LLM)
    1. EMS Butler watches SOC vs a 20\% line and a look-ahead blizzard
       forecast, then posts a cut target \(\alpha\) (e.g. 45\%).
    2. Role A (survival parent) and Role B (WFH/comfort) alternate
       concessions until a feasible agreement.
    3. The family contract is a structured action
       ``{shed_appliances, hvac_temp_offset, power_cut_kW}``.
    4. The 8760 h engine applies the contract on the storm mask and
       re-simulates SOC (demand-side response, DSR).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.battery_model.battery_dynamics import CommercialBattery, step_soc
from src.load_profiler.mcmc_load import (
    DEFAULT_CATALOG,
    HVAC_NAME,
    LoadSynthParams,
    _hvac_layer,
    _hour_in_peaks,
    _load_catalog,
)
from src.optimization.sizing_milp import (
    DISPATCH_CSV,
    LOAD_CSV,
    PACK_ROOT,
    RESULTS_DIR,
    SIZING_CSV,
    evaluate_full_year_dispatch,
    load_pv_load_table,
    milp_battery_catalog,
)

FIGURE_DIR: Path = PACK_ROOT / "paper_figures"
IRRADIANCE_CSV: Path = PACK_ROOT / "data" / "raw" / "solar_irradiance_annual.csv"
AGREEMENT_JSON: Path = RESULTS_DIR / "family_agreement.json"
STORM_CSV: Path = RESULTS_DIR / "blizzard_dsr_dispatch.csv"
STORM_FIG: Path = FIGURE_DIR / "fig_blizzard_dsr_soc.png"

SOC_WARN: float = 0.20
STORM_HOURS: int = 72
PROTECTED: tuple[str, ...] = (
    "Refrigerator_Freezer",
    "WiFi_Router_Comm",
    "Home_Office_Laptop",
    "Well_Water_Pump",
)
PARENT_SHED: tuple[str, ...] = (
    "Electric_Clothes_Dryer",
    "Dishwasher",
    "Washing_Machine",
    "Microwave_Oven",
    "Electric_Water_Heater",
    "Induction_Cooktop",
    HVAC_NAME,
)
WFH_SHED: tuple[str, ...] = (
    "Electric_Clothes_Dryer",
    "Dishwasher",
)


@dataclass(frozen=True, slots=True)
class BlizzardAlert:
    r"""Butler look-ahead.

    Attributes
    ----------
    level :
        1 = watch, 2 = warning, 3 = blizzard / SOC-line breach.
    t0, t1 :
        Inclusive hour indices of the forecast window.
    soc_now :
        SOC at \(t_0\).
    soc_min_forecast :
        Greedy min SOC over the window with no DSR.
    target_cut_frac :
        Requested load reduction \(\alpha\in[0,1]\).
    """

    level: int
    t0: int
    t1: int
    soc_now: float
    soc_min_forecast: float
    target_cut_frac: float
    mean_pv_kw: float
    mean_temp_c: float
    reason: str


@dataclass(frozen=True, slots=True)
class FamilyAction:
    """Structured action contract (diagram JSON)."""

    shed_appliances: tuple[str, ...]
    hvac_temp_offset: float
    power_cut_kW: float
    light_dim_frac: float = 0.0
    protected_appliances: tuple[str, ...] = PROTECTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "shed_appliances": list(self.shed_appliances),
            "hvac_temp_offset": float(self.hvac_temp_offset),
            "power_cut_kW": float(self.power_cut_kW),
            "light_dim_frac": float(self.light_dim_frac),
            "protected_appliances": list(self.protected_appliances),
        }


def _fleet_from_sizing(path: Path | None = None) -> CommercialBattery:
    """Build the installed equivalent pack from the MILP BOM CSV."""

    csv_path = Path(path) if path is not None else SIZING_CSV
    packs = milp_battery_catalog()
    x_hat: dict[str, int] = {p.model_name: 0 for p in packs}
    if csv_path.is_file():
        bom = pd.read_csv(csv_path)
        for _, row in bom.iterrows():
            x_hat[str(row["model_name"])] = int(row["units"])
    else:
        x_hat["FREEDOH_Module"] = 3
        x_hat["Lead_Carbon_Array"] = 1
    e_cap = 0.0
    p_max = 0.0
    eta_w = 0.0
    soc_min = 0.0
    soc_max = 0.0
    cycles_w = 0.0
    for p in packs:
        n_u = int(x_hat.get(p.model_name, 0))
        if n_u <= 0:
            continue
        e_cap += n_u * p.capacity_kWh
        p_max += n_u * p.continuous_power_kW
        eta_w += n_u * p.capacity_kWh * p.roundtrip_eff
        soc_min += n_u * p.capacity_kWh * p.soc_min
        soc_max += n_u * p.capacity_kWh * p.soc_max
        cycles_w += n_u * p.capacity_kWh * p.cycle_life
    if e_cap <= 1.0e-9:
        raise ValueError("BOM has zero storage; run sizing MILP first")
    return CommercialBattery(
        model_name="installed_fleet",
        capacity_kWh=e_cap,
        continuous_power_kW=p_max,
        roundtrip_eff=eta_w / e_cap,
        cost_usd=0.0,
        cycle_life=max(int(cycles_w / e_cap), 1),
        chemistry="LFP",
        soc_min=soc_min / e_cap,
        soc_max=soc_max / e_cap,
    )


def _simulate_soc_path(
    pv: np.ndarray,
    load: np.ndarray,
    battery: CommercialBattery,
    soc0: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy EMS: returns SOC, deficit kW, applied discharge kW."""

    n = int(pv.size)
    soc = np.zeros(n, dtype=np.float64)
    def_kw = np.zeros(n, dtype=np.float64)
    dis_kw = np.zeros(n, dtype=np.float64)
    s = 0.5 * (battery.soc_min + battery.soc_max) if soc0 is None else float(soc0)
    soh = 1.0
    for t in range(n):
        soc[t] = s
        surplus = float(pv[t] - load[t])
        if surplus >= 0.0:
            s, soh, _pch, _pdis = step_soc(s, surplus, 0.0, battery, soh=soh)
        else:
            need = -surplus
            s, soh, _pch, pdis = step_soc(s, 0.0, need, battery, soh=soh)
            dis_kw[t] = pdis
            def_kw[t] = need - pdis
    return soc, def_kw, dis_kw


def find_blizzard_window(
    pv_kw: np.ndarray,
    t_amb_c: np.ndarray,
    width_h: int = STORM_HOURS,
) -> tuple[int, int]:
    """Lowest-PV cold window: argmin of rolling PV sum among cold hours."""

    pv = np.asarray(pv_kw, dtype=np.float64)
    temp = np.asarray(t_amb_c, dtype=np.float64)
    n = int(pv.size)
    w = min(int(width_h), n)
    csum = np.concatenate(([0.0], np.cumsum(pv)))
    best_i = 0
    best_score = np.inf
    for i in range(0, n - w + 1):
        e_pv = csum[i + w] - csum[i]
        t_mean = float(temp[i : i + w].mean())
        # Prefer dark and cold; small PV dominates.
        score = e_pv + 0.35 * max(t_mean, -5.0) * w
        if score < best_score:
            best_score = score
            best_i = i
    return best_i, best_i + w - 1


def classify_alert(
    soc_now: float,
    soc_min_forecast: float,
    target_cut_frac: float,
    mean_pv_kw: float,
    mean_temp_c: float,
) -> tuple[int, str]:
    """Map SOC / weather into butler levels 1–3."""

    if soc_now <= SOC_WARN or soc_min_forecast <= SOC_WARN:
        if mean_temp_c < 14.0:
            return 3, (
                f"Level-3 blizzard: SOC line {SOC_WARN:.0%} breached in forecast "
                f"(now {soc_now:.1%}, min {soc_min_forecast:.1%}); "
                f"request {target_cut_frac:.0%} load cut."
            )
        return 2, (
            f"Level-2 SOC warning: forecast min {soc_min_forecast:.1%} "
            f"vs {SOC_WARN:.0%} line."
        )
    if mean_pv_kw < 0.50 and mean_temp_c < 10.0:
        return 2, "Level-2 winter-storm watch (low PV, cold)."
    return 1, "Level-1 watch."


def butler_alert(
    pv: np.ndarray,
    load: np.ndarray,
    t_amb_c: np.ndarray,
    battery: CommercialBattery,
    soc_path: np.ndarray,
    t0: int,
    t1: int,
) -> BlizzardAlert:
    """Look-ahead cut target so forecast min SOC stays on the 20% line."""

    soc_now = float(soc_path[t0])
    sl = slice(t0, t1 + 1)
    soc_w, def_w, _dis = _simulate_soc_path(pv[sl], load[sl], battery, soc0=soc_now)
    soc_min = float(np.min(soc_w))
    load_e = float(np.sum(load[sl]))
    def_e = float(np.sum(def_w))
    # Energy that must be shed to hold SOC_WARN (plus the already-unserved kWh).
    e_cap = battery.capacity_kWh
    gap = max(SOC_WARN - soc_min, 0.0) * e_cap + def_e
    alpha = float(np.clip(gap / max(load_e, 1.0e-6), 0.15, 0.60))
    if soc_min <= SOC_WARN:
        alpha = max(alpha, 0.45)
    mean_pv = float(np.mean(pv[sl]))
    mean_t = float(np.mean(t_amb_c[sl]))
    level, reason = classify_alert(soc_now, soc_min, alpha, mean_pv, mean_t)
    return BlizzardAlert(
        level=level,
        t0=int(t0),
        t1=int(t1),
        soc_now=soc_now,
        soc_min_forecast=soc_min,
        target_cut_frac=alpha,
        mean_pv_kw=mean_pv,
        mean_temp_c=mean_t,
        reason=reason,
    )


def _appliance_kw_profile(
    name: str,
    hour: np.ndarray,
    catalog: pd.DataFrame,
) -> np.ndarray:
    """Hourly coincidence kW used for bargaining (catalog × peak duty)."""

    if name not in catalog.index:
        return np.zeros(hour.shape, dtype=np.float64)
    row = catalog.loc[name]
    rated = float(row["rated_power_W"]) / 1000.0
    duty = float(row["duty_cycle"])
    in_peak = _hour_in_peaks(hour, str(row["peak_hours"]))
    return np.where(in_peak, rated * duty, rated * duty * 0.12)


def _lights_kw(hour: np.ndarray, catalog: pd.DataFrame) -> np.ndarray:
    if "LED_Lighting_WholeHouse" not in catalog.index:
        return np.zeros(hour.shape, dtype=np.float64)
    rated = float(catalog.loc["LED_Lighting_WholeHouse"]["rated_power_W"]) / 1000.0
    evening = (hour >= 18) & (hour <= 23)
    return np.where(evening, rated, 0.08 * rated)


def estimate_cut_series(
    action: FamilyAction,
    hour: np.ndarray,
    t_amb_c: np.ndarray,
    hvac_kw: np.ndarray,
    catalog: pd.DataFrame,
    params: LoadSynthParams,
    storm_mask: np.ndarray,
) -> np.ndarray:
    r"""Non-negative kW shed; HVAC via \(T^\star\) shift, appliances via catalog."""

    n = int(hour.size)
    cut = np.zeros(n, dtype=np.float64)
    if abs(action.hvac_temp_offset) > 1.0e-9:
        shifted = replace(params, t_star_c=params.t_star_c + action.hvac_temp_offset)
        hvac_new = _hvac_layer(t_amb_c, catalog, shifted)
        cut += np.maximum(hvac_kw - hvac_new, 0.0)
    for name in action.shed_appliances:
        if name == HVAC_NAME:
            continue
        cut += _appliance_kw_profile(name, hour, catalog)
    if action.light_dim_frac > 0.0:
        cut += action.light_dim_frac * _lights_kw(hour, catalog)
    cut = np.where(storm_mask, cut, 0.0)
    return np.maximum(cut, 0.0)


def mean_cut_kw(
    action: FamilyAction,
    hour: np.ndarray,
    t_amb_c: np.ndarray,
    hvac_kw: np.ndarray,
    catalog: pd.DataFrame,
    params: LoadSynthParams,
    storm_mask: np.ndarray,
) -> float:
    series = estimate_cut_series(
        action, hour, t_amb_c, hvac_kw, catalog, params, storm_mask
    )
    m = storm_mask.astype(np.float64)
    denom = float(np.sum(m))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(series, m) / denom)


def negotiate(
    alert: BlizzardAlert,
    hour: np.ndarray,
    t_amb_c: np.ndarray,
    load_kw: np.ndarray,
    hvac_kw: np.ndarray,
    catalog: pd.DataFrame,
    params: LoadSynthParams,
) -> tuple[FamilyAction, list[dict[str, str]]]:
    """Three-round parent ↔ WFH bargain. Protected loads never enter the shed list."""

    storm = np.zeros(hour.shape, dtype=bool)
    storm[alert.t0 : alert.t1 + 1] = True
    load_mean = float(np.mean(load_kw[storm])) if np.any(storm) else float(np.mean(load_kw))
    target_kw = alert.target_cut_frac * load_mean

    parent = FamilyAction(
        shed_appliances=tuple(a for a in PARENT_SHED if a not in PROTECTED),
        hvac_temp_offset=-6.0,
        power_cut_kW=0.0,
        light_dim_frac=0.70,
    )
    wfh = FamilyAction(
        shed_appliances=WFH_SHED,
        hvac_temp_offset=-1.0,
        power_cut_kW=0.0,
        light_dim_frac=0.45,
    )
    deal = FamilyAction(
        shed_appliances=WFH_SHED,
        hvac_temp_offset=-2.5,
        power_cut_kW=0.0,
        light_dim_frac=0.45,
    )
    extra: list[str] = []
    if mean_cut_kw(deal, hour, t_amb_c, hvac_kw, catalog, params, storm) < 0.85 * target_kw:
        extra.append("Washing_Machine")
        deal = FamilyAction(
            shed_appliances=tuple(list(WFH_SHED) + extra),
            hvac_temp_offset=-2.5,
            power_cut_kW=0.0,
            light_dim_frac=0.45,
        )
    for cand in (parent, wfh, deal):
        for name in cand.shed_appliances:
            if name in PROTECTED:
                raise RuntimeError(f"protected load {name} cannot be shed")
    parent = replace(
        parent,
        power_cut_kW=mean_cut_kw(parent, hour, t_amb_c, hvac_kw, catalog, params, storm),
    )
    wfh = replace(
        wfh,
        power_cut_kW=mean_cut_kw(wfh, hour, t_amb_c, hvac_kw, catalog, params, storm),
    )
    deal = replace(
        deal,
        power_cut_kW=mean_cut_kw(deal, hour, t_amb_c, hvac_kw, catalog, params, storm),
    )

    pct = 100.0 * alert.target_cut_frac
    level_zh = {1: "一级观察", 2: "二级预警", 3: "三级暴雪缺电预警"}[alert.level]
    dialogue = [
        {
            "agent": "EMS_Butler",
            "text_zh": (
                f"{level_zh}：SOC 警戒线 {SOC_WARN:.0%} 将在预报窗内被击穿"
                f"（当前 {alert.soc_now:.1%}，无需求响应最低 {alert.soc_min_forecast:.1%}）。"
                f"窗内均温 {alert.mean_temp_c:.1f}°C、光伏 {alert.mean_pv_kw:.2f} kW。"
                f"请家庭在未来 {alert.t1 - alert.t0 + 1} h 削减约 {pct:.0f}% 功率"
                f"（约 {target_kw:.1f} kW）。"
            ),
            "text_en": alert.reason,
        },
        {
            "agent": "Role_A_Survival_Parent",
            "text_zh": (
                "精打细算家长：关闭地暖、拔掉非必要插头保命。冰箱和水井泵留下，"
                f"烘干机/洗碗机/洗衣机/热水器全部停。估计可削 {parent.power_cut_kW:.1f} kW。"
            ),
            "text_en": (
                "Shut floor heating and unplug non-essentials; keep fridge and well pump."
            ),
        },
        {
            "agent": "Role_B_WFH",
            "text_zh": (
                "居家办公成员：我可以不烘干衣服、调暗灯光，但我的工作电脑和路由器不能断。"
                f"洗碗机可推迟。估计可削 {wfh.power_cut_kW:.1f} kW。"
            ),
            "text_en": (
                "I can skip the dryer and dim lights, but the work PC and router stay on."
            ),
        },
        {
            "agent": "Family_Agreement",
            "text_zh": (
                f"妥协：停 {', '.join(deal.shed_appliances)}；地暖设定 {deal.hvac_temp_offset:.1f}°C；"
                f"灯光调暗 {deal.light_dim_frac:.0%}；电脑/路由/冰箱受保护。"
                f"合约削负荷 {deal.power_cut_kW:.1f} kW。"
            ),
            "text_en": (
                f"shed={list(deal.shed_appliances)}, HVAC {deal.hvac_temp_offset} K, "
                f"cut {deal.power_cut_kW:.1f} kW."
            ),
        },
    ]
    return deal, dialogue


def apply_family_action(
    load_kw: np.ndarray,
    hour: np.ndarray,
    t_amb_c: np.ndarray,
    hvac_kw: np.ndarray,
    action: FamilyAction,
    catalog: pd.DataFrame,
    params: LoadSynthParams,
    storm_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract DSR cut; never go negative. Returns load_dsr, cut_kw."""

    cut = estimate_cut_series(
        action, hour, t_amb_c, hvac_kw, catalog, params, storm_mask
    )
    # Cap cut so residual load stays non-negative (no invented energy).
    cut = np.minimum(cut, np.maximum(load_kw, 0.0))
    load_dsr = np.maximum(load_kw - cut, 0.0)
    return load_dsr, cut


def plot_storm_soc(
    hours: np.ndarray,
    soc_base: np.ndarray,
    soc_dsr: np.ndarray,
    t0: int,
    t1: int,
    path: Path | None = None,
) -> Path:
    """SOC with / without DSR across the blizzard window."""

    if path is None:
        path = STORM_FIG
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sl = slice(max(t0 - 12, 0), min(t1 + 13, len(hours)))
    x = np.arange(sl.start, sl.stop)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, soc_base[sl], color="#c45911", lw=1.8, label="No DSR")
    ax.plot(x, soc_dsr[sl], color="#1f4e79", lw=1.8, label="Family DSR contract")
    ax.axhline(SOC_WARN, color="#c00000", ls="--", lw=1.2, label="20% SOC line")
    ax.axvspan(t0, t1, color="#9dc3e6", alpha=0.25, label="Blizzard window")
    ax.set_xlabel("Hour index (8760 series)", fontsize=11)
    ax.set_ylabel("SOC", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Blizzard SOC: butler + household DSR", fontsize=12)
    ax.grid(True, ls="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def run_blizzard_agents(
    *,
    load_csv: Path | None = None,
    sizing_csv: Path | None = None,
    irradiance_csv: Path | None = None,
    agreement_json: Path | None = None,
    storm_csv: Path | None = None,
    figure_path: Path | None = None,
) -> dict[str, Any]:
    """End-to-end: detect storm, bargain, apply DSR, write JSON/CSV/figure."""

    table = load_pv_load_table(load_csv)
    irr = pd.read_csv(irradiance_csv if irradiance_csv is not None else IRRADIANCE_CSV)
    if len(irr) != len(table):
        raise ValueError("irradiance and PV/load tables must both be 8760 rows")
    t_amb = irr["temperature_c"].to_numpy(dtype=np.float64)
    pv = table["pv_output_kW"].to_numpy(dtype=np.float64)
    load = table["load_demand_kW"].to_numpy(dtype=np.float64)
    hour = table["hour"].to_numpy(dtype=np.int64)
    catalog = _load_catalog(DEFAULT_CATALOG)
    params = LoadSynthParams()
    hvac = _hvac_layer(t_amb, catalog, params)
    battery = _fleet_from_sizing(sizing_csv)
    soc_year, def_year, _dis = _simulate_soc_path(pv, load, battery)
    t0, t1 = find_blizzard_window(pv, t_amb)
    alert = butler_alert(pv, load, t_amb, battery, soc_year, t0, t1)
    action, dialogue = negotiate(alert, hour, t_amb, load, hvac, catalog, params)
    storm = np.zeros(len(table), dtype=bool)
    storm[t0 : t1 + 1] = True
    load_dsr, cut = apply_family_action(
        load, hour, t_amb, hvac, action, catalog, params, storm
    )
    soc_dsr, def_dsr, _ = _simulate_soc_path(pv, load_dsr, battery)
    payload = {
        "alert": {
            "level": alert.level,
            "t0": alert.t0,
            "t1": alert.t1,
            "soc_now": alert.soc_now,
            "soc_min_forecast": alert.soc_min_forecast,
            "target_cut_frac": alert.target_cut_frac,
            "mean_pv_kw": alert.mean_pv_kw,
            "mean_temp_c": alert.mean_temp_c,
            "reason": alert.reason,
        },
        "dialogue": dialogue,
        "action": action.to_dict(),
        "metrics": {
            "soc_min_no_dsr": float(np.min(soc_year[t0 : t1 + 1])),
            "soc_min_dsr": float(np.min(soc_dsr[t0 : t1 + 1])),
            "def_kwh_no_dsr": float(np.sum(def_year[t0 : t1 + 1])),
            "def_kwh_dsr": float(np.sum(def_dsr[t0 : t1 + 1])),
            "mean_cut_kw": float(np.mean(cut[storm])),
            "e_capacity_kWh": battery.capacity_kWh,
        },
    }
    out_json = Path(agreement_json) if agreement_json is not None else AGREEMENT_JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    storm_frame = pd.DataFrame(
        {
            "t": np.arange(len(table)),
            "hour": hour,
            "month": table["month"].to_numpy(),
            "temperature_c": t_amb,
            "pv_output_kW": pv,
            "load_demand_kW": load,
            "load_dsr_kW": load_dsr,
            "dsr_cut_kW": cut,
            "soc_no_dsr": soc_year,
            "soc_dsr": soc_dsr,
            "p_def_no_dsr_kW": def_year,
            "p_def_dsr_kW": def_dsr,
            "storm_mask": storm.astype(int),
        }
    )
    out_csv = Path(storm_csv) if storm_csv is not None else STORM_CSV
    storm_frame.to_csv(out_csv, index=False)
    fig = plot_storm_soc(np.arange(len(table)), soc_year, soc_dsr, t0, t1, figure_path)
    print(dialogue[0]["text_zh"])
    print(dialogue[1]["text_zh"])
    print(dialogue[2]["text_zh"])
    print(dialogue[3]["text_zh"])
    print(
        f"storm h={t0}-{t1}  SOC min {payload['metrics']['soc_min_no_dsr']:.3f} → "
        f"{payload['metrics']['soc_min_dsr']:.3f}  "
        f"def {payload['metrics']['def_kwh_no_dsr']:.1f} → "
        f"{payload['metrics']['def_kwh_dsr']:.1f} kWh"
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    print(f"wrote {fig}")
    payload["paths"] = {
        "agreement_json": str(out_json.resolve()),
        "storm_csv": str(out_csv.resolve()),
        "figure": str(fig.resolve()),
    }
    _ = DISPATCH_CSV
    _ = LOAD_CSV
    return payload


def main() -> None:
    """CLI: Level-3 blizzard bargain on the installed MILP fleet."""

    run_blizzard_agents()


if __name__ == "__main__":
    main()

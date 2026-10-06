r"""Budget Pareto scan and emerging-storage LCOS (HiMCM 2021 A, Task 4).

LCOS (discounted, \(N\) years, rate \(r\))
\[
\mathrm{LCOS}
=
\frac{
  \mathrm{CapEx}
  +\sum_{y=1}^{N}
    \dfrac{\mathrm{OpEx}_y+\mathrm{Replacement}_y}{(1+r)^y}
}{
  \sum_{y=1}^{N}
    \dfrac{E_{\mathrm{discharged},y}}{(1+r)^y}
}.
\]

Throughput \(E_{\mathrm{discharged},1}\) is taken from the same 8760 h
PV/load CSV and greedy EMS used in sizing — not invented. Capex and
cycle parameters for Na-ion / VRFB / cement are literature-order
assumptions documented on ``EmergingTech``, not measured lab data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.battery_model.battery_dynamics import CommercialBattery
from src.optimization.sizing_milp import (
    LOAD_CSV,
    PACK_ROOT,
    RESULTS_DIR,
    MilpCostParams,
    build_and_solve_milp,
    evaluate_full_year_dispatch,
    load_pv_load_table,
    milp_battery_catalog,
    typical_month_days,
)

FIGURE_DIR: Path = PACK_ROOT / "paper_figures"
PARETO_CSV: Path = RESULTS_DIR / "budget_pareto_front.csv"
LCOS_CSV: Path = RESULTS_DIR / "lcos_emerging_tech.csv"
DISPATCH_CSV: Path = RESULTS_DIR / "annual_power_dispatch.csv"
SIZING_CSV: Path = RESULTS_DIR / "optimal_sizing_solution.csv"

BUDGET_MIN_USD: float = 4000.0
BUDGET_MAX_USD: float = 18000.0
BUDGET_STEP_USD: float = 1000.0
KNEE_BUDGETS_USD: tuple[float, float] = (8000.0, 10000.0)

LCOS_YEARS: int = 20
LCOS_DISCOUNT: float = 0.07
REF_CAPACITY_KWH: float = 13.5
REF_POWER_KW: float = 5.0


@dataclass(frozen=True, slots=True)
class EmergingTech:
    r"""One storage family for Task-4 LCOS.

    Attributes
    ----------
    tech_id, name :
        Short code and display name.
    trl :
        NASA/DOE technology readiness level (integer 1–9).
    capex_usd_per_kwh :
        Installed energy-side capex (USD/kWh).
    roundtrip_eff :
        AC–AC \(\eta_{\mathrm{rt}}\).
    cycle_life :
        Equivalent full cycles to ~10\% fade.
    calendar_years :
        Warranty / electrolyte calendar life.
    opex_frac_capex :
        Annual O&M as a fraction of initial CapEx.
    replacement_frac :
        Fraction of CapEx paid at each end-of-life event.
    energy_density_wh_kg :
        Gravimetric energy density (Wh/kg).
    env_score :
        Qualitative 0–10 (resource / recyclability; higher is better).
    c_rate :
        Continuous power / energy (1/h).
    """

    tech_id: str
    name: str
    trl: int
    capex_usd_per_kwh: float
    roundtrip_eff: float
    cycle_life: int
    calendar_years: float
    opex_frac_capex: float
    replacement_frac: float
    energy_density_wh_kg: float
    env_score: float
    c_rate: float
    notes: str


def emerging_catalog() -> list[EmergingTech]:
    """Four families: LFP baseline, Na-ion, VRFB, cement structural."""

    # LFP $/kWh matches Tesla_Powerwall_Plus in PACK_BASELINES (10500/13.5).
    return [
        EmergingTech(
            tech_id="A_LFP",
            name="LFP commercial baseline",
            trl=9,
            capex_usd_per_kwh=10500.0 / 13.5,
            roundtrip_eff=0.90,
            cycle_life=4000,
            calendar_years=15.0,
            opex_frac_capex=0.010,
            replacement_frac=0.70,
            energy_density_wh_kg=140.0,
            env_score=7.0,
            c_rate=5.0 / 13.5,
            notes="TRL9 pack quote aligned to Powerwall+ capex in the MILP catalog",
        ),
        EmergingTech(
            tech_id="B_NaIon",
            name="Sodium-ion",
            trl=7,
            capex_usd_per_kwh=300.0,
            roundtrip_eff=0.90,
            cycle_life=4000,
            calendar_years=15.0,
            opex_frac_capex=0.012,
            replacement_frac=0.70,
            energy_density_wh_kg=120.0,
            env_score=8.5,
            c_rate=0.40,
            notes="Literature-order 2024–25 pack $/kWh; abundant Na, better low-T",
        ),
        EmergingTech(
            tech_id="C_VRFB",
            name="Vanadium redox flow",
            trl=8,
            capex_usd_per_kwh=600.0,
            roundtrip_eff=0.75,
            cycle_life=15000,
            calendar_years=25.0,
            opex_frac_capex=0.015,
            replacement_frac=0.30,
            energy_density_wh_kg=20.0,
            env_score=8.0,
            c_rate=0.25,
            notes="Long-duration; stack replacement fraction 0.3 (electrolyte reused)",
        ),
        EmergingTech(
            tech_id="D_Cement",
            name="Cement-based structural storage",
            trl=4,
            capex_usd_per_kwh=1800.0,
            roundtrip_eff=0.60,
            cycle_life=8000,
            calendar_years=50.0,
            opex_frac_capex=0.005,
            replacement_frac=0.20,
            energy_density_wh_kg=0.8,
            env_score=9.0,
            c_rate=0.10,
            notes="TRL4 research-order Wh/kg; building-integrated, not a drop-in pack",
        ),
    ]


def count_outage_events(def_kw: np.ndarray, threshold_kw: float = 1.0e-4) -> int:
    r"""Count contiguous outage runs where \(P_{\mathrm{def}}>\tau\)."""

    flag = np.asarray(def_kw, dtype=np.float64) > float(threshold_kw)
    if flag.size == 0:
        return 0
    prev = np.concatenate(([False], flag[:-1]))
    return int(np.sum(flag & ~prev))


def _bom_string(x_hat: dict[str, int]) -> str:
    parts = [f"{n}×{name}" for name, n in x_hat.items() if int(n) > 0]
    return "+".join(parts) if parts else "none"


def default_budgets() -> np.ndarray:
    r"""Inclusive grid $4000–$18000 step $1000."""

    n = int(round((BUDGET_MAX_USD - BUDGET_MIN_USD) / BUDGET_STEP_USD)) + 1
    return np.linspace(BUDGET_MIN_USD, BUDGET_MAX_USD, n)


def scan_budget_pareto(
    budgets: Iterable[float] | None = None,
    *,
    costs: MilpCostParams | None = None,
    load_csv: Path | None = None,
    out_csv: Path | None = None,
    evaluate_8760: bool = True,
    time_limit_s: int = 25,
) -> pd.DataFrame:
    """Solve the sizing MILP on each budget; write ``budget_pareto_front.csv``.

    Does not overwrite ``optimal_sizing_solution.csv``. Outage hours and
    event counts on the 8760 h greedy trajectory when ``evaluate_8760``.
    """

    if costs is None:
        costs = MilpCostParams()
    if out_csv is None:
        out_csv = PARETO_CSV
    packs = milp_battery_catalog()
    table = load_pv_load_table(load_csv)
    pv, load, w = typical_month_days(table)
    grid = default_budgets() if budgets is None else np.asarray(list(budgets), dtype=np.float64)
    rows: list[dict[str, object]] = []
    for budget in grid:
        milp = build_and_solve_milp(
            pv_kw=pv,
            load_kw=load,
            hour_weights=w,
            packs=packs,
            budget_limit=float(budget),
            max_allowed_outages=None,
            costs=costs,
            wrap="daily",
            time_limit_s=time_limit_s,
            name=f"pareto_{int(budget)}",
        )
        x_hat = milp["x"]
        assert isinstance(x_hat, dict)
        lpsp_8760 = float("nan")
        out_h = float("nan")
        out_e = float("nan")
        def_e = float("nan")
        if evaluate_8760:
            year = evaluate_full_year_dispatch(table, packs, x_hat, costs)
            lpsp_8760 = float(year["lpsp_8760"])
            out_h = float(year["outage_hours_8760"])
            disp = year["dispatch_8760"]
            assert isinstance(disp, pd.DataFrame)
            out_e = float(count_outage_events(disp["p_def_kW"].to_numpy()))
            def_e = float(year["def_energy_kWh_8760"])
        rows.append(
            {
                "budget_usd": float(budget),
                "capex_usd": float(milp["capex_usd"]),
                "tco_usd": float(milp["tco_usd"]),
                "lpsp_typical": float(milp["lpsp"]),
                "lpsp_8760": lpsp_8760,
                "outage_hours_8760": out_h,
                "outage_events_8760": out_e,
                "def_energy_kWh_8760": def_e,
                "e_capacity_kWh": float(milp["e_capacity_kWh"]),
                "p_capacity_kW": float(milp["p_capacity_kW"]),
                "bom": _bom_string(x_hat),
                "n_Tesla_Powerwall_Plus": int(x_hat.get("Tesla_Powerwall_Plus", 0)),
                "n_Enphase_IQ_10T": int(x_hat.get("Enphase_IQ_10T", 0)),
                "n_FREEDOH_Module": int(x_hat.get("FREEDOH_Module", 0)),
                "n_Lead_Carbon_Array": int(x_hat.get("Lead_Carbon_Array", 0)),
                "lpsp_relaxed": bool(milp.get("lpsp_relaxed", False)),
            }
        )
        print(
            f"budget=${float(budget):,.0f}  capex=${float(milp['capex_usd']):,.0f}  "
            f"LPSP8760={lpsp_8760:.4f}  outage_h={out_h:.0f}  "
            f"events={out_e:.0f}  {_bom_string(x_hat)}"
        )
    frame = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def compute_lcos(
    capex_usd: float,
    e_discharged_y1_kwh: float,
    *,
    n_years: int = LCOS_YEARS,
    discount_rate: float = LCOS_DISCOUNT,
    opex_usd_year: float = 0.0,
    replacement_usd: float = 0.0,
    lifetime_years: float = 15.0,
    fade_to: float = 0.80,
) -> dict[str, float]:
    r"""Discounted LCOS. Year-0 CapEx is undiscounted; cash flows at year-end.

    Throughput in year \(y\) is \(E_1\) times a linear SOH fade to
    ``fade_to`` at replacement, then resets. Replacement cash occurs at
    integer years \(k\cdot\lfloor\mathrm{life}\rfloor\) for \(k\ge 1\).
    """

    if n_years < 1:
        raise ValueError("n_years must be >= 1")
    life = max(float(lifetime_years), 1.0)
    npv_cost = float(capex_usd)
    npv_e = 0.0
    for y in range(1, n_years + 1):
        df = 1.0 / (1.0 + float(discount_rate)) ** y
        age = (y - 1) % int(round(life))
        span = max(int(round(life)), 1)
        soh = 1.0 - (1.0 - float(fade_to)) * (age / span)
        e_y = max(float(e_discharged_y1_kwh), 0.0) * soh
        repl = 0.0
        if int(round(life)) >= 1 and y % int(round(life)) == 0 and y < n_years:
            repl = float(replacement_usd)
        npv_cost += (float(opex_usd_year) + repl) * df
        npv_e += e_y * df
    lcos = float("inf") if npv_e <= 1.0e-9 else npv_cost / npv_e
    return {
        "npv_cost_usd": npv_cost,
        "npv_energy_kwh": npv_e,
        "lcos_usd_per_kwh": lcos,
        "lifetime_years": life,
    }


def _simulate_ref_throughput(tech: EmergingTech, table: pd.DataFrame) -> dict[str, float]:
    """Greedy 8760 h dispatch of a 13.5 kWh reference pack with this chemistry."""

    pack = CommercialBattery(
        model_name=tech.tech_id,
        capacity_kWh=REF_CAPACITY_KWH,
        continuous_power_kW=max(REF_CAPACITY_KWH * tech.c_rate, 0.5),
        roundtrip_eff=tech.roundtrip_eff,
        cost_usd=tech.capex_usd_per_kwh * REF_CAPACITY_KWH,
        cycle_life=tech.cycle_life,
        chemistry=tech.tech_id,
        soc_min=0.10 if tech.tech_id != "D_Cement" else 0.05,
        soc_max=0.95 if tech.tech_id != "D_Cement" else 0.95,
    )
    year = evaluate_full_year_dispatch(
        table,
        [pack],
        {pack.model_name: 1},
        MilpCostParams(),
    )
    disp = year["dispatch_8760"]
    assert isinstance(disp, pd.DataFrame)
    e_dis = float(disp["p_dis_kW"].sum())
    e_ch = float(disp["p_ch_kW"].sum())
    efc = 0.5 * (e_ch * pack.eta_ch + e_dis / pack.eta_dis) / pack.capacity_kWh
    return {
        "e_discharged_y1_kwh": e_dis,
        "e_charged_y1_kwh": e_ch,
        "efc_year": float(efc),
        "lpsp_8760": float(year["lpsp_8760"]),
    }


def build_lcos_table(
    *,
    load_csv: Path | None = None,
    out_csv: Path | None = None,
    n_years: int = LCOS_YEARS,
    discount_rate: float = LCOS_DISCOUNT,
    techs: Sequence[EmergingTech] | None = None,
) -> pd.DataFrame:
    """Compute LCOS for the four emerging families; write CSV."""

    if techs is None:
        techs = emerging_catalog()
    if out_csv is None:
        out_csv = LCOS_CSV
    table = load_pv_load_table(load_csv)
    rows: list[dict[str, object]] = []
    for tech in techs:
        capex = tech.capex_usd_per_kwh * REF_CAPACITY_KWH
        sim = _simulate_ref_throughput(tech, table)
        efc = max(sim["efc_year"], 1.0e-6)
        cycle_life_y = tech.cycle_life / efc
        life = min(float(tech.calendar_years), float(cycle_life_y))
        lcos = compute_lcos(
            capex,
            sim["e_discharged_y1_kwh"],
            n_years=n_years,
            discount_rate=discount_rate,
            opex_usd_year=tech.opex_frac_capex * capex,
            replacement_usd=tech.replacement_frac * capex,
            lifetime_years=life,
        )
        rows.append(
            {
                "tech_id": tech.tech_id,
                "name": tech.name,
                "trl": tech.trl,
                "capex_usd_per_kwh": tech.capex_usd_per_kwh,
                "capex_ref_usd": capex,
                "roundtrip_eff": tech.roundtrip_eff,
                "cycle_life": tech.cycle_life,
                "calendar_years": tech.calendar_years,
                "lifetime_years_used": lcos["lifetime_years"],
                "energy_density_wh_kg": tech.energy_density_wh_kg,
                "env_score": tech.env_score,
                "e_discharged_y1_kwh": sim["e_discharged_y1_kwh"],
                "efc_year": sim["efc_year"],
                "lpsp_8760_ref_pack": sim["lpsp_8760"],
                "npv_cost_usd": lcos["npv_cost_usd"],
                "npv_energy_kwh": lcos["npv_energy_kwh"],
                "lcos_usd_per_kwh": lcos["lcos_usd_per_kwh"],
                "discount_rate": discount_rate,
                "horizon_years": n_years,
                "notes": tech.notes,
            }
        )
    frame = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=9)


def plot_budget_outage_pareto(
    pareto: pd.DataFrame,
    path: Path | None = None,
    knee_usd: Sequence[float] = KNEE_BUDGETS_USD,
) -> Path:
    """Fig. 1: budget vs outage hours, LPSP on the twin axis; knee markers."""

    if path is None:
        path = FIGURE_DIR / "fig_budget_outage_pareto.png"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = pareto["budget_usd"].to_numpy(dtype=np.float64)
    y_h = pareto["outage_hours_8760"].to_numpy(dtype=np.float64)
    y_p = pareto["lpsp_8760"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, y_h, color="#1f4e79", marker="o", lw=2.0, ms=5.5, label="Outage hours")
    ax.set_xlabel("Capital budget (USD)", fontsize=11)
    ax.set_ylabel("Annual outage hours", color="#1f4e79", fontsize=11)
    ax.tick_params(axis="y", colors="#1f4e79")
    _style_axes(ax)

    ax2 = ax.twinx()
    ax2.plot(x, y_p, color="#c45911", marker="s", lw=1.6, ms=4.5, label="LPSP")
    ax2.set_ylabel("LPSP (8760 h)", color="#c45911", fontsize=11)
    ax2.tick_params(axis="y", colors="#c45911")
    ax2.spines["top"].set_visible(False)

    y_lo, y_hi = float(np.nanmin(y_h)), float(np.nanmax(y_h))
    for k, budget in enumerate(knee_usd):
        ax.axvline(budget, color="#548235", ls=":", lw=1.4, alpha=0.9)
        ax.annotate(
            f"Knee ${budget:,.0f}",
            xy=(budget, y_lo + 0.12 * (y_hi - y_lo) * (1 + 0.35 * k)),
            xytext=(8, 12 + 10 * k),
            textcoords="offset points",
            fontsize=8.5,
            color="#375623",
            arrowprops={"arrowstyle": "->", "color": "#375623", "lw": 0.8},
        )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=8.5)
    ax.set_title("Budget–reliability Pareto (MILP sizing + 8760 h EMS)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_soc_carpet(
    dispatch: pd.DataFrame,
    e_capacity_kwh: float,
    path: Path | None = None,
) -> Path:
    """Fig. 2: 365×24 SOC heatmap from the 8760 h trajectory."""

    if path is None:
        path = FIGURE_DIR / "fig_8760h_dispatch_carpet.png"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    e_bat = dispatch["e_bat_kWh"].to_numpy(dtype=np.float64)
    n = min(int(e_bat.size), 8760)
    n_days = n // 24
    n_use = n_days * 24
    cap = max(float(e_capacity_kwh), 1.0e-9)
    soc = (e_bat[:n_use] / cap).reshape(n_days, 24)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    mesh = ax.imshow(
        soc,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_ylabel("Day of year (time order)", fontsize=11)
    ax.set_xticks([0, 6, 12, 18, 23])
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("SOC  $E(t)/E_{cap}$", fontsize=10)
    ax.set_title("8760 h battery SOC carpet", fontsize=12)
    ax.annotate(
        "summer high-SOC band",
        xy=(14, int(0.45 * n_days)),
        xytext=(16, int(0.18 * n_days)),
        fontsize=8,
        color="white",
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 0.8},
    )
    ax.annotate(
        "winter draw-down",
        xy=(4, int(0.85 * n_days)),
        xytext=(8, int(0.70 * n_days)),
        fontsize=8,
        color="white",
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 0.8},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _radar_scores(lcos_table: pd.DataFrame) -> tuple[list[str], np.ndarray, list[str]]:
    """Map five metrics onto [0, 1]; LCOS is inverted (lower cost → higher score)."""

    axes = ["LCOS (inv.)", "Energy density", "Cycle life", "Environment", "Round-trip"]
    labels = lcos_table["name"].tolist()
    lcos = lcos_table["lcos_usd_per_kwh"].to_numpy(dtype=np.float64)
    dens = lcos_table["energy_density_wh_kg"].to_numpy(dtype=np.float64)
    cyc = lcos_table["cycle_life"].to_numpy(dtype=np.float64)
    env = lcos_table["env_score"].to_numpy(dtype=np.float64)
    eta = lcos_table["roundtrip_eff"].to_numpy(dtype=np.float64)

    def unit(v: np.ndarray, invert: bool = False) -> np.ndarray:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if hi - lo < 1.0e-12:
            out = np.ones_like(v)
        else:
            out = (v - lo) / (hi - lo)
        return 1.0 - out if invert else out

    scores = np.vstack(
        [
            unit(lcos, invert=True),
            unit(np.log10(np.maximum(dens, 1.0e-3))),
            unit(np.log10(np.maximum(cyc, 1.0))),
            unit(env),
            unit(eta),
        ]
    ).T
    scores = 0.12 + 0.88 * np.clip(scores, 0.0, 1.0)
    return axes, scores, labels


def plot_lcos_radar(
    lcos_table: pd.DataFrame,
    path: Path | None = None,
) -> Path:
    """Fig. 3: four-technology radar on five normalized axes."""

    if path is None:
        path = FIGURE_DIR / "fig_emerging_tech_lcos_radar.png"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    axes, scores, labels = _radar_scores(lcos_table)
    n_ax = len(axes)
    ang = np.linspace(0.0, 2.0 * np.pi, n_ax, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    colors = ("#1f4e79", "#ed7d31", "#548235", "#833c0c")
    fig, ax = plt.subplots(figsize=(6.6, 6.2), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(ang), axes, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    for i, name in enumerate(labels):
        row = np.concatenate([scores[i], scores[i, :1]])
        ax.plot(ang_c, row, color=colors[i % 4], lw=1.8, label=name)
        ax.fill(ang_c, row, color=colors[i % 4], alpha=0.12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8, frameon=False)
    ax.set_title("Emerging storage: five-axis comparison", fontsize=12, pad=16)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _capacity_from_sizing(path: Path) -> float:
    if not path.is_file():
        return REF_CAPACITY_KWH
    bom = pd.read_csv(path)
    if "capacity_kWh_total" in bom.columns:
        return float(bom["capacity_kWh_total"].sum())
    return REF_CAPACITY_KWH


def run_techno_economic(
    *,
    budgets: Iterable[float] | None = None,
    evaluate_8760: bool = True,
) -> dict[str, Path]:
    """Scan budgets, compute LCOS, write three 300 DPI figures."""

    pareto = scan_budget_pareto(budgets=budgets, evaluate_8760=evaluate_8760)
    lcos = build_lcos_table()
    fig1 = plot_budget_outage_pareto(pareto)
    if DISPATCH_CSV.is_file():
        dispatch = pd.read_csv(DISPATCH_CSV)
    else:
        packs = milp_battery_catalog()
        table = load_pv_load_table()
        x_hat = {
            "Tesla_Powerwall_Plus": 0,
            "Enphase_IQ_10T": 0,
            "FREEDOH_Module": 3,
            "Lead_Carbon_Array": 1,
        }
        year = evaluate_full_year_dispatch(table, packs, x_hat, MilpCostParams())
        dispatch = year["dispatch_8760"]
        assert isinstance(dispatch, pd.DataFrame)
    e_cap = _capacity_from_sizing(SIZING_CSV)
    if e_cap <= 1.0e-9:
        e_cap = float(np.nanmax(dispatch["e_bat_kWh"].to_numpy())) / 0.95
    fig2 = plot_soc_carpet(dispatch, e_cap)
    fig3 = plot_lcos_radar(lcos)
    print(f"wrote {PARETO_CSV}")
    print(f"wrote {LCOS_CSV}")
    print(f"wrote {fig1}")
    print(f"wrote {fig2}")
    print(f"wrote {fig3}")
    print(lcos[["tech_id", "trl", "lcos_usd_per_kwh", "e_discharged_y1_kwh"]].to_string(index=False))
    return {
        "pareto_csv": PARETO_CSV,
        "lcos_csv": LCOS_CSV,
        "fig_pareto": fig1,
        "fig_carpet": fig2,
        "fig_radar": fig3,
    }


def main() -> None:
    r"""CLI: $4k–$18k Pareto, LCOS table, three paper figures."""

    run_techno_economic()


if __name__ == "__main__":
    main()

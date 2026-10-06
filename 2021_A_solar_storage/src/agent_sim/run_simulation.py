r"""Event-driven blizzard A/B test: passive house vs crewAI DSR.

Window: days 40–42 of the 8760 h archive (72 hourly steps), taken from
the Open-Meteo irradiance / PV–load CSVs — not invented.

    Group A — rigid load, ignore the SOC line.
    Group B — when \(SOC<0.20\) and a dark-hour streak is on, call
    ``run_family_negotiation`` once and apply ``AdaptiveLoadAdapter``
    for the next 48 h.

SOC updates use ``step_soc`` (charge/discharge efficiencies, clamps).
No path creates energy: \(P_{\mathrm{dsr}}\le P_{\mathrm{raw}}\).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.agent_sim.crew_engine import SOC_LINE, run_family_negotiation
from src.agent_sim.load_adapter import AdaptiveLoadAdapter
from src.agent_sim.models import LoadSheddingAgreement
from src.battery_model.battery_dynamics import CommercialBattery, step_soc
from src.optimization.sizing_milp import (
    LOAD_CSV,
    PACK_ROOT,
    RESULTS_DIR,
    SIZING_CSV,
    load_pv_load_table,
    milp_battery_catalog,
)

IRRADIANCE_CSV: Path = PACK_ROOT / "data" / "raw" / "solar_irradiance_annual.csv"
OUT_CSV: Path = RESULTS_DIR / "crewai_adaptive_vs_passive.csv"

STORM_DAY_START: int = 40  # 1-indexed day-of-year in the 8760 archive
STORM_N_DAYS: int = 3
DSR_HORIZON_H: int = 48
DARK_GHI_WM2: float = 15.0
DARK_STREAK_H: int = 3
DEF_EPS_KW: float = 1.0e-4


def storm_slice_hours(
    n_year: int = 8760,
    day_start: int = STORM_DAY_START,
    n_days: int = STORM_N_DAYS,
) -> tuple[int, int]:
    """Inclusive hour indices for 1-indexed days ``day_start`` .. ``day_start+n_days-1``."""

    t0 = (int(day_start) - 1) * 24
    t1 = t0 + int(n_days) * 24 - 1
    if t0 < 0 or t1 >= n_year:
        raise ValueError(f"storm window [{t0}, {t1}] outside 0..{n_year - 1}")
    return t0, t1


def installed_battery(sizing_csv: Path | None = None) -> CommercialBattery:
    """Equivalent pack from the MILP BOM (zero units → 3×FREEDOH + 1×lead-carbon)."""

    csv_path = Path(sizing_csv) if sizing_csv is not None else SIZING_CSV
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
    for pack in packs:
        n_u = int(x_hat.get(pack.model_name, 0))
        if n_u <= 0:
            continue
        e_cap += n_u * pack.capacity_kWh
        p_max += n_u * pack.continuous_power_kW
        eta_w += n_u * pack.capacity_kWh * pack.roundtrip_eff
        soc_min += n_u * pack.capacity_kWh * pack.soc_min
        soc_max += n_u * pack.capacity_kWh * pack.soc_max
        cycles_w += n_u * pack.capacity_kWh * pack.cycle_life
    if e_cap <= 1.0e-9:
        raise ValueError("installed storage is zero; run sizing MILP first")
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


def _step_hour(
    soc: float,
    soh: float,
    pv_kw: float,
    load_kw: float,
    battery: CommercialBattery,
) -> tuple[float, float, float, float, float]:
    """One EMS hour. Returns soc_next, soh_next, p_ch, p_dis, p_def."""

    surplus = float(pv_kw) - float(load_kw)
    if surplus >= 0.0:
        soc_n, soh_n, p_ch, p_dis = step_soc(soc, surplus, 0.0, battery, soh=soh)
        return soc_n, soh_n, p_ch, p_dis, 0.0
    need = -surplus
    soc_n, soh_n, p_ch, p_dis = step_soc(soc, 0.0, need, battery, soh=soh)
    return soc_n, soh_n, p_ch, p_dis, max(need - p_dis, 0.0)


def simulate_passive(
    pv: np.ndarray,
    load: np.ndarray,
    battery: CommercialBattery,
    t_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy SOC / deficit on the raw load through hour ``t_end`` inclusive."""

    n = int(t_end) + 1
    soc = np.zeros(n, dtype=np.float64)
    def_kw = np.zeros(n, dtype=np.float64)
    s = 0.5 * (battery.soc_min + battery.soc_max)
    soh = 1.0
    for t in range(n):
        soc[t] = s
        s, soh, _c, _d, def_kw[t] = _step_hour(s, soh, float(pv[t]), float(load[t]), battery)
    return soc, def_kw


def _dark_streak(ghi_wm2: np.ndarray, t: int, streak_h: int = DARK_STREAK_H) -> bool:
    lo = max(0, t - streak_h + 1)
    window = ghi_wm2[lo : t + 1]
    return int(window.size) >= streak_h and bool(np.all(window < DARK_GHI_WM2))


def simulate_adaptive(
    pv: np.ndarray,
    load: np.ndarray,
    hour: np.ndarray,
    ghi_wm2: np.ndarray,
    battery: CommercialBattery,
    adapter: AdaptiveLoadAdapter,
    t0: int,
    t1: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LoadSheddingAgreement | None, int]:
    """Passive until trigger; then adapter for ``DSR_HORIZON_H`` hours.

    Returns
    -------
    soc, def_kw, load_dsr, agreement, trigger_t
        ``trigger_t`` is -1 if the crew never fired.
    """

    n = int(t1) + 1
    soc = np.zeros(n, dtype=np.float64)
    def_kw = np.zeros(n, dtype=np.float64)
    load_dsr = np.array(load[:n], dtype=np.float64, copy=True)
    s = 0.5 * (battery.soc_min + battery.soc_max)
    soh = 1.0
    agreement: LoadSheddingAgreement | None = None
    trigger_t = -1
    dsr_until = -1
    for t in range(n):
        soc[t] = s
        in_storm = t0 <= t <= t1
        if (
            in_storm
            and agreement is None
            and s < SOC_LINE
            and _dark_streak(ghi_wm2, t)
        ):
            agreement = run_family_negotiation(
                soc=float(s),
                target_cut_frac=0.45,
            )
            trigger_t = t
            dsr_until = min(t + DSR_HORIZON_H - 1, t1)
            print(
                f"crewAI trigger t={t} SOC={s:.3f}  "
                f"DSR hours {t}–{dsr_until}  "
                f"shed={agreement.shed_appliances}"
            )
        if agreement is not None and trigger_t <= t <= dsr_until:
            load_dsr[t] = float(
                adapter.apply_curtailment(
                    float(load[t]),
                    agreement,
                    int(hour[t]),
                )
            )
        s, soh, _c, _d, def_kw[t] = _step_hour(
            s, soh, float(pv[t]), float(load_dsr[t]), battery
        )
    return soc, def_kw, load_dsr, agreement, trigger_t


def _first_empty_hour(
    soc: np.ndarray,
    def_kw: np.ndarray,
    soc_min: float,
    t0: int,
    t1: int,
) -> int | None:
    """First storm hour where usable SOC is exhausted and load is unserved."""

    for t in range(t0, t1 + 1):
        if soc[t] <= soc_min + 1.0e-6 and def_kw[t] > DEF_EPS_KW:
            return int(t)
    return None


def _outage_hours(def_kw: np.ndarray, t0: int, t1: int) -> int:
    return int(np.sum(def_kw[t0 : t1 + 1] > DEF_EPS_KW))


def _longest_outage_run(def_kw: np.ndarray, t0: int, t1: int) -> int:
    longest = 0
    run = 0
    for t in range(t0, t1 + 1):
        if def_kw[t] > DEF_EPS_KW:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return int(longest)


def run_blizzard_ab(
    *,
    load_csv: Path | None = None,
    irradiance_csv: Path | None = None,
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """Warm-start from hour 0, then A/B on days 40–42; write the comparison CSV."""

    table = load_pv_load_table(load_csv)
    irr = pd.read_csv(irradiance_csv if irradiance_csv is not None else IRRADIANCE_CSV)
    if len(irr) != len(table):
        raise ValueError("irradiance and PV/load tables must both be 8760 rows")
    pv = table["pv_output_kW"].to_numpy(dtype=np.float64)
    load = table["load_demand_kW"].to_numpy(dtype=np.float64)
    hour = table["hour"].to_numpy(dtype=np.int64)
    ghi = irr["ghi_W_m2"].to_numpy(dtype=np.float64)
    stamps = irr["time"].astype(str).to_numpy()
    t0, t1 = storm_slice_hours(len(table))
    battery = installed_battery()
    adapter = AdaptiveLoadAdapter()

    soc_a, def_a = simulate_passive(pv, load, battery, t1)
    soc_b, def_b, load_b, agreement, trigger_t = simulate_adaptive(
        pv, load, hour, ghi, battery, adapter, t0, t1
    )

    empty_a = _first_empty_hour(soc_a, def_a, battery.soc_min, t0, t1)
    empty_b = _first_empty_hour(soc_b, def_b, battery.soc_min, t0, t1)
    frame = pd.DataFrame(
        {
            "timestamp": stamps[t0 : t1 + 1],
            "hour_index": np.arange(t0, t1 + 1),
            "hour_of_day": hour[t0 : t1 + 1],
            "ghi_W_m2": ghi[t0 : t1 + 1],
            "pv_output_kW": pv[t0 : t1 + 1],
            "raw_load_kW": load[t0 : t1 + 1],
            "negotiated_load_kW": load_b[t0 : t1 + 1],
            "soc_passive": soc_a[t0 : t1 + 1],
            "soc_adaptive": soc_b[t0 : t1 + 1],
            "p_def_passive_kW": def_a[t0 : t1 + 1],
            "p_def_adaptive_kW": def_b[t0 : t1 + 1],
        }
    )
    path = Path(out_csv) if out_csv is not None else OUT_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

    print(
        f"storm days {STORM_DAY_START}–{STORM_DAY_START + STORM_N_DAYS - 1}  "
        f"hours {t0}–{t1}  ({stamps[t0]} → {stamps[t1]})"
    )
    print(
        f"GHI mean={float(ghi[t0:t1+1].mean()):.1f} W/m2  "
        f"T mean={float(irr['temperature_c'].to_numpy()[t0:t1+1].mean()):.1f} C  "
        f"PV mean={float(pv[t0:t1+1].mean()):.3f} kW"
    )
    print(
        f"A passive: SOC min={float(np.min(soc_a[t0:t1+1])):.3f}  "
        f"empty_at={empty_a}  outage_h={_outage_hours(def_a, t0, t1)}  "
        f"longest_run={_longest_outage_run(def_a, t0, t1)} h  "
        f"def={float(def_a[t0:t1+1].sum()):.2f} kWh"
    )
    print(
        f"B adaptive: SOC min={float(np.min(soc_b[t0:t1+1])):.3f}  "
        f"empty_at={empty_b}  outage_h={_outage_hours(def_b, t0, t1)}  "
        f"longest_run={_longest_outage_run(def_b, t0, t1)} h  "
        f"def={float(def_b[t0:t1+1].sum()):.2f} kWh  trigger_t={trigger_t}"
    )
    if agreement is not None:
        print(agreement.model_dump_json(indent=2))
    print(f"wrote {path}")
    return frame


def main() -> None:
    """CLI: ``python -m src.agent_sim.run_simulation``."""

    run_blizzard_ab()


if __name__ == "__main__":
    main()

r"""MILP capacity sizing + EMS dispatch (SciPy HiGHS).

Decision variables
    \(x_k\in\{0,1,2,3,4\}\) units of commercial pack \(k\);
    hourly \(P_{\mathrm{ch},k}(t),P_{\mathrm{dis},k}(t),E_{k}(t)\);
    bus \(P_{\mathrm{def}}(t),P_{\mathrm{curt}}(t)\).

Per-pack energy (keeps \(\eta_k\) linear; no \(x_k\eta_k\) product)
    \[
    E_k(t{+}1)
    = E_k(t)+P_{\mathrm{ch},k}(t)\eta_{\mathrm{ch},k}
      -P_{\mathrm{dis},k}(t)/\eta_{\mathrm{dis},k}.
    \]

Bus balance
    \[
    P_{\mathrm{PV}}(t)-P_{\mathrm{curt}}(t)
    +\sum_k P_{\mathrm{dis},k}(t)-\sum_k P_{\mathrm{ch},k}(t)
    +P_{\mathrm{def}}(t)=P_{\mathrm{load}}(t).
    \]

Objective (10-year TCO)
    \[
    \min\sum_k x_k\mathrm{Cost}_k
    +H\sum_t w_t\bigl(C_{\mathrm{outage}}P_{\mathrm{def}}(t)
      +C_{\mathrm{waste}}P_{\mathrm{curt}}(t)\bigr).
    \]

Default horizon is 12 monthly typical days (\(T=288\)) with weights
\(w_t=365/12\) so the year is represented without a 5-nested pack
enumeration. Pass ``horizon="full"`` for a true 8760-hour MILP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from src.battery_model.battery_dynamics import (
    PACK_BASELINES,
    CommercialBattery,
)

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
RESULTS_DIR: Path = PACK_ROOT / "results"
LOAD_CSV: Path = RESULTS_DIR / "annual_pv_load_8760h.csv"
SIZING_CSV: Path = RESULTS_DIR / "optimal_sizing_solution.csv"
DISPATCH_CSV: Path = RESULTS_DIR / "annual_power_dispatch.csv"

X_MAX: int = 4
HORIZON_YEARS: float = 10.0
EPS_LPSP_DEFAULT: float = 0.05
SOLVER_TIME_TYPICAL_S: int = 90
SOLVER_TIME_FULL_S: int = 180


@dataclass(frozen=True, slots=True)
class MilpCostParams:
    r"""Economic and reliability knobs.

    Attributes
    ----------
    c_outage_usd_per_kwh :
        Value of lost load \(C_{\mathrm{outage}}\) (USD/kWh).
    c_waste_usd_per_kwh :
        Curtailment penalty \(C_{\mathrm{waste}}\) (USD/kWh).
    epsilon_lpsp :
        Hard LPSP cap \(\varepsilon_{\mathrm{allow}}\).
    horizon_years :
        Operating years \(H\) in the TCO sum.
    """

    c_outage_usd_per_kwh: float = 4.0
    c_waste_usd_per_kwh: float = 0.05
    epsilon_lpsp: float = EPS_LPSP_DEFAULT
    horizon_years: float = HORIZON_YEARS


def milp_battery_catalog() -> list[CommercialBattery]:
    """Four brief packs: Powerwall+, Enphase IQ, FREEDOH module, lead-carbon."""

    packs: list[CommercialBattery] = []
    for row in PACK_BASELINES:
        packs.append(
            CommercialBattery(
                model_name=str(row["model_name"]),
                capacity_kWh=float(row["capacity_kWh"]),
                continuous_power_kW=float(row["continuous_power_kW"]),
                roundtrip_eff=float(row["roundtrip_eff"]),
                cost_usd=float(row["cost_usd"]),
                cycle_life=int(row["cycle_life"]),
                chemistry=str(row["chemistry"]),
                soc_min=float(row["soc_min"]),
                soc_max=float(row["soc_max"]),
            )
        )
    return packs


def load_pv_load_table(path: Path | None = None) -> pd.DataFrame:
    """Read 8760-hour PV/load CSV, generating it if missing."""

    csv_path = Path(path) if path is not None else LOAD_CSV
    if not csv_path.is_file():
        from src.scripts.run_pv_load import run_annual_pv_load

        run_annual_pv_load(out_csv=csv_path)
    table = pd.read_csv(csv_path)
    need = {"hour", "month", "pv_output_kW", "load_demand_kW"}
    missing = need.difference(table.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns {missing}")
    return table


def typical_month_days(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One median-load day per month: PV, load (kW) and hour-weights (h/year)."""

    pv_parts: list[np.ndarray] = []
    load_parts: list[np.ndarray] = []
    weights: list[float] = []
    n_year = len(table)
    for month in range(1, 13):
        block = table.loc[table["month"] == month].reset_index(drop=True)
        if block.empty:
            continue
        n_days = max(len(block) // 24, 1)
        daily = (
            block["load_demand_kW"]
            .to_numpy(dtype=np.float64)[: n_days * 24]
            .reshape(n_days, 24)
            .sum(axis=1)
        )
        pick = int(np.argsort(daily)[len(daily) // 2])
        sl = slice(pick * 24, pick * 24 + 24)
        pv_parts.append(block["pv_output_kW"].to_numpy(dtype=np.float64)[sl])
        load_parts.append(block["load_demand_kW"].to_numpy(dtype=np.float64)[sl])
        w = float(len(block)) / 24.0  # days represented by this month
        weights.extend([w] * 24)
    pv = np.concatenate(pv_parts)
    load = np.concatenate(load_parts)
    w_arr = np.asarray(weights, dtype=np.float64)
    if abs(w_arr.sum() - float(n_year)) > 2.0:
        w_arr *= float(n_year) / max(w_arr.sum(), 1.0)
    return pv, load, w_arr


def _next_index(t: int, n: int, wrap: Literal["horizon", "daily"]) -> int:
    if wrap == "daily":
        day = t // 24
        return day * 24 + (t + 1) % 24
    return (t + 1) % n


def build_and_solve_milp(
    pv_kw: np.ndarray,
    load_kw: np.ndarray,
    hour_weights: np.ndarray,
    packs: Sequence[CommercialBattery],
    budget_limit: float,
    max_allowed_outages: float | None,
    costs: MilpCostParams,
    wrap: Literal["horizon", "daily"] = "daily",
    time_limit_s: int = SOLVER_TIME_TYPICAL_S,
    name: str = "offgrid_milp",
    enforce_lpsp: bool = True,
) -> dict[str, object]:
    """Build a sparse MILP and solve with HiGHS. No pack-count enumeration."""

    _ = name
    pv = np.asarray(pv_kw, dtype=np.float64)
    load = np.asarray(load_kw, dtype=np.float64)
    w = np.asarray(hour_weights, dtype=np.float64)
    if pv.shape != load.shape or pv.shape != w.shape:
        raise ValueError("pv, load, and weights must share shape")
    n = int(pv.size)
    k_n = len(packs)
    load_year = float(np.dot(w, load))
    pv_year = float(np.dot(w, pv))
    if load_year <= 0.0:
        raise ValueError("annual load energy must be positive")
    min_lpsp = max(0.0, 1.0 - pv_year / load_year)
    eps_lpsp = float(costs.epsilon_lpsp)

    use_u = max_allowed_outages is not None
    # layout: x[k] | ch[k,t] | dis[k,t] | e[k,t] | def[t] | curt[t] | (u[t])
    off_x = 0
    off_ch = k_n
    off_dis = off_ch + k_n * n
    off_e = off_dis + k_n * n
    off_def = off_e + k_n * n
    off_curt = off_def + n
    off_u = off_curt + n
    n_vars = off_u + (n if use_u else 0)

    def ix_x(k: int) -> int:
        return off_x + k

    def ix_ch(k: int, t: int) -> int:
        return off_ch + k * n + t

    def ix_dis(k: int, t: int) -> int:
        return off_dis + k * n + t

    def ix_e(k: int, t: int) -> int:
        return off_e + k * n + t

    def ix_def(t: int) -> int:
        return off_def + t

    def ix_curt(t: int) -> int:
        return off_curt + t

    def ix_u(t: int) -> int:
        return off_u + t

    c = np.zeros(n_vars, dtype=np.float64)
    for k, pack in enumerate(packs):
        c[ix_x(k)] = pack.cost_usd
    for t in range(n):
        c[ix_def(t)] = costs.horizon_years * w[t] * costs.c_outage_usd_per_kwh
        c[ix_curt(t)] = costs.horizon_years * w[t] * costs.c_waste_usd_per_kwh

    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    for k, pack in enumerate(packs):
        ub[ix_x(k)] = float(X_MAX)
        p_cap = pack.continuous_power_kW * X_MAX
        e_cap = pack.capacity_kWh * pack.soc_max * X_MAX
        for t in range(n):
            ub[ix_ch(k, t)] = p_cap
            ub[ix_dis(k, t)] = p_cap
            ub[ix_e(k, t)] = e_cap
    for t in range(n):
        ub[ix_def(t)] = float(load[t])
        ub[ix_curt(t)] = float(pv[t])
        if use_u:
            ub[ix_u(t)] = 1.0

    integrality = np.zeros(n_vars, dtype=np.int8)
    for k in range(k_n):
        integrality[ix_x(k)] = 1
    if use_u:
        for t in range(n):
            integrality[ix_u(t)] = 1

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    con_lb: list[float] = []
    con_ub: list[float] = []
    row_i = 0

    def add_row(indices: list[int], coeffs: list[float], lo: float, hi: float) -> None:
        nonlocal row_i
        rows.extend([row_i] * len(indices))
        cols.extend(indices)
        data.extend(coeffs)
        con_lb.append(lo)
        con_ub.append(hi)
        row_i += 1

    # budget
    add_row(
        [ix_x(k) for k in range(k_n)],
        [packs[k].cost_usd for k in range(k_n)],
        0.0,
        float(budget_limit),
    )
    if enforce_lpsp:
        add_row(
            [ix_def(t) for t in range(n)],
            [float(w[t]) for t in range(n)],
            0.0,
            eps_lpsp * load_year,
        )
    # power balance: -sum ch + sum dis + def - curt = load - pv
    for t in range(n):
        idx = [ix_ch(k, t) for k in range(k_n)] + [ix_dis(k, t) for k in range(k_n)]
        idx += [ix_def(t), ix_curt(t)]
        coef = [-1.0] * k_n + [1.0] * k_n + [1.0, -1.0]
        rhs = float(load[t] - pv[t])
        add_row(idx, coef, rhs, rhs)
    # pack dynamics and limits
    for k, pack in enumerate(packs):
        eta_c = pack.eta_ch
        inv_eta_d = 1.0 / pack.eta_dis
        e_lo = pack.capacity_kWh * pack.soc_min
        e_hi = pack.capacity_kWh * pack.soc_max
        p_max = pack.continuous_power_kW
        for t in range(n):
            t_next = _next_index(t, n, wrap)
            # e_next - e - eta_c ch + (1/eta_d) dis = 0
            add_row(
                [ix_e(k, t_next), ix_e(k, t), ix_ch(k, t), ix_dis(k, t)],
                [1.0, -1.0, -eta_c, inv_eta_d],
                0.0,
                0.0,
            )
            add_row([ix_e(k, t), ix_x(k)], [1.0, -e_hi], -np.inf, 0.0)
            add_row([ix_e(k, t), ix_x(k)], [1.0, -e_lo], 0.0, np.inf)
            add_row(
                [ix_ch(k, t), ix_dis(k, t), ix_x(k)],
                [1.0, 1.0, -p_max],
                -np.inf,
                0.0,
            )
    if use_u:
        p_m = float(np.max(load)) + 1.0
        for t in range(n):
            add_row([ix_def(t), ix_u(t)], [1.0, -p_m], -np.inf, 0.0)
        add_row(
            [ix_u(t) for t in range(n)],
            [float(w[t]) for t in range(n)],
            0.0,
            float(max_allowed_outages),
        )

    a_mat = sparse.coo_matrix((data, (rows, cols)), shape=(row_i, n_vars)).tocsr()
    constraint = LinearConstraint(a_mat, np.asarray(con_lb), np.asarray(con_ub))
    bounds = Bounds(lb, ub)
    result = milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraint,
        options={"time_limit": float(time_limit_s), "disp": False},
    )
    if result.x is None:
        if enforce_lpsp:
            fallback = build_and_solve_milp(
                pv_kw=pv,
                load_kw=load,
                hour_weights=w,
                packs=packs,
                budget_limit=budget_limit,
                max_allowed_outages=max_allowed_outages,
                costs=costs,
                wrap=wrap,
                time_limit_s=time_limit_s,
                name=name,
                enforce_lpsp=False,
            )
            fallback["lpsp_relaxed"] = True
            fallback["message"] = (
                f"LPSP≤{eps_lpsp:.4f} infeasible under budget=${budget_limit:.0f}; "
                f"returned TCO-optimum (LPSP={float(fallback['lpsp']):.4f})"
            )
            return fallback
        raise RuntimeError(
            f"MILP infeasible or unsolved: success={result.success} {result.message}"
        )
    x_sol = result.x
    status = "Optimal" if result.success else "Feasible"
    x_hat = {
        packs[k].model_name: int(round(float(x_sol[ix_x(k)])))
        for k in range(k_n)
    }
    capex_v = float(sum(x_hat[packs[k].model_name] * packs[k].cost_usd for k in range(k_n)))
    def_kw = np.array([float(x_sol[ix_def(t)]) for t in range(n)])
    curt_kw = np.array([float(x_sol[ix_curt(t)]) for t in range(n)])
    ch_kw = np.zeros(n, dtype=np.float64)
    dis_kw = np.zeros(n, dtype=np.float64)
    e_tot = np.zeros(n, dtype=np.float64)
    for t in range(n):
        ch_kw[t] = sum(float(x_sol[ix_ch(k, t)]) for k in range(k_n))
        dis_kw[t] = sum(float(x_sol[ix_dis(k, t)]) for k in range(k_n))
        e_tot[t] = sum(float(x_sol[ix_e(k, t)]) for k in range(k_n))
    def_energy = float(np.dot(w, def_kw))
    curt_energy = float(np.dot(w, curt_kw))
    lpsp = def_energy / load_year
    opex_v = float(
        np.dot(w, costs.c_outage_usd_per_kwh * def_kw + costs.c_waste_usd_per_kwh * curt_kw)
    )
    tco = capex_v + costs.horizon_years * opex_v
    e_cap = sum(x_hat[packs[k].model_name] * packs[k].capacity_kWh for k in range(k_n))
    p_cap = sum(
        x_hat[packs[k].model_name] * packs[k].continuous_power_kW for k in range(k_n)
    )
    outage_hours = float(np.dot(w, (def_kw > 1.0e-4).astype(np.float64)))
    balance = pv - curt_kw + dis_kw - ch_kw + def_kw - load
    return {
        "status": status,
        "message": str(result.message),
        "x": x_hat,
        "capex_usd": capex_v,
        "opex_year_usd": opex_v,
        "tco_usd": tco,
        "lpsp": lpsp,
        "epsilon_lpsp_requested": float(costs.epsilon_lpsp),
        "epsilon_lpsp_applied": (float(lpsp) if not enforce_lpsp else eps_lpsp),
        "min_lpsp_energy_gap": min_lpsp,
        "lpsp_relaxed": (not enforce_lpsp),
        "pv_energy_kWh_year": pv_year,
        "def_energy_kWh_year": def_energy,
        "curt_energy_kWh_year": curt_energy,
        "load_energy_kWh_year": load_year,
        "outage_hours_year": outage_hours,
        "e_capacity_kWh": e_cap,
        "p_capacity_kW": p_cap,
        "balance_max_abs_kw": float(np.max(np.abs(balance))),
        "dispatch": pd.DataFrame(
            {
                "t": np.arange(n),
                "pv_kW": pv,
                "load_kW": load,
                "p_ch_kW": ch_kw,
                "p_dis_kW": dis_kw,
                "p_def_kW": def_kw,
                "p_curt_kW": curt_kw,
                "e_bat_kWh": e_tot,
                "weight_h": w,
            }
        ),
    }


def evaluate_full_year_dispatch(
    table: pd.DataFrame,
    packs: Sequence[CommercialBattery],
    x_hat: dict[str, int],
    costs: MilpCostParams,
    time_limit_s: int = 60,
) -> dict[str, object]:
    r"""Fix \(x_k\) and run a causal greedy EMS on 8760 h (SOC from battery_dynamics).

    The MILP already optimized dispatch on typical days. This pass reports
    annual LPSP without a second 43k-variable LP. ``time_limit_s`` is unused
    and kept for call-site compatibility.
    """

    from src.battery_model.battery_dynamics import step_soc

    _ = (costs, time_limit_s)
    pv = table["pv_output_kW"].to_numpy(dtype=np.float64)
    load = table["load_demand_kW"].to_numpy(dtype=np.float64)
    n = len(pv)
    e_cap = 0.0
    p_max = 0.0
    eta_w = 0.0
    soc_min_acc = 0.0
    soc_max_acc = 0.0
    cycles_w = 0.0
    for p in packs:
        n_u = int(x_hat.get(p.model_name, 0))
        if n_u <= 0:
            continue
        e_cap += n_u * p.capacity_kWh
        p_max += n_u * p.continuous_power_kW
        eta_w += n_u * p.capacity_kWh * p.roundtrip_eff
        soc_min_acc += n_u * p.capacity_kWh * p.soc_min
        soc_max_acc += n_u * p.capacity_kWh * p.soc_max
        cycles_w += n_u * p.capacity_kWh * p.cycle_life
    ch_kw = np.zeros(n)
    dis_kw = np.zeros(n)
    def_kw = np.zeros(n)
    curt_kw = np.zeros(n)
    e_tot = np.zeros(n)
    if e_cap <= 1.0e-9:
        def_kw = load.copy()
        curt_kw = pv.copy()
    else:
        eta_rt = eta_w / e_cap
        soc_min = soc_min_acc / e_cap
        soc_max = soc_max_acc / e_cap
        eq = CommercialBattery(
            model_name="fleet_equivalent",
            capacity_kWh=e_cap,
            continuous_power_kW=p_max,
            roundtrip_eff=eta_rt,
            cost_usd=0.0,
            cycle_life=max(int(cycles_w / e_cap), 1),
            chemistry="LFP",
            soc_min=soc_min,
            soc_max=soc_max,
        )
        soc = 0.5 * (soc_min + soc_max)
        soh = 1.0
        for t in range(n):
            e_tot[t] = soc * e_cap
            surplus = float(pv[t] - load[t])
            if surplus >= 0.0:
                soc, soh, pch, _pdis = step_soc(soc, surplus, 0.0, eq, soh=soh)
                ch_kw[t] = pch
                curt_kw[t] = surplus - pch
            else:
                need = -surplus
                soc, soh, _pch, pdis = step_soc(soc, 0.0, need, eq, soh=soh)
                dis_kw[t] = pdis
                def_kw[t] = need - pdis
    def_e = float(def_kw.sum())
    load_e = float(load.sum())
    return {
        "lpsp_8760": def_e / max(load_e, 1.0e-9),
        "outage_hours_8760": float(np.sum(def_kw > 1.0e-4)),
        "def_energy_kWh_8760": def_e,
        "curt_energy_kWh_8760": float(curt_kw.sum()),
        "dispatch_status": "greedy_ems",
        "dispatch_8760": pd.DataFrame(
            {
                "hour": table["hour"].to_numpy(),
                "month": table["month"].to_numpy(),
                "pv_output_kW": pv,
                "load_demand_kW": load,
                "p_ch_kW": ch_kw,
                "p_dis_kW": dis_kw,
                "p_def_kW": def_kw,
                "p_curt_kW": curt_kw,
                "e_bat_kWh": e_tot,
            }
        ),
    }


def _write_sizing_csv(
    packs: Sequence[CommercialBattery],
    x_hat: dict[str, int],
    summary: dict[str, object],
    path: Path,
) -> pd.DataFrame:
    rows = []
    for p in packs:
        n_u = int(x_hat.get(p.model_name, 0))
        rows.append(
            {
                "model_name": p.model_name,
                "units": n_u,
                "capacity_kWh_each": p.capacity_kWh,
                "power_kW_each": p.continuous_power_kW,
                "cost_usd_each": p.cost_usd,
                "capacity_kWh_total": n_u * p.capacity_kWh,
                "power_kW_total": n_u * p.continuous_power_kW,
                "capex_usd": n_u * p.cost_usd,
                "chemistry": p.chemistry,
                "roundtrip_eff": p.roundtrip_eff,
            }
        )
    frame = pd.DataFrame(rows)
    frame["budget_usd"] = float(summary.get("budget_limit", np.nan))
    frame["tco_usd"] = float(summary["tco_usd"])
    frame["lpsp"] = float(summary["lpsp"])
    frame["epsilon_lpsp_applied"] = float(summary.get("epsilon_lpsp_applied", np.nan))
    frame["lpsp_relaxed"] = bool(summary.get("lpsp_relaxed", False))
    frame["outage_hours_year"] = float(summary["outage_hours_year"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def solve_optimal_storage(
    budget_limit: float,
    max_allowed_outages: float | None = None,
    *,
    horizon: Literal["typical", "full"] = "typical",
    costs: MilpCostParams | None = None,
    load_csv: Path | None = None,
    sizing_csv: Path | None = None,
    dispatch_csv: Path | None = None,
    evaluate_8760: bool = True,
    time_limit_s: int | None = None,
) -> dict[str, object]:
    r"""Size packs under a budget and outage limit; write BOM + dispatch CSVs.

    Args:
        budget_limit:
            Capex ceiling \(\sum_k x_k\mathrm{Cost}_k\le B\) (USD).
        max_allowed_outages:
            Annual hours with \(P_{\mathrm{def}}>0\) (weighted); ``None``
            drops the binary outage-count constraint (LPSP hard cap remains).
        horizon:
            ``typical`` = 12×24 h MILP; ``full`` = 8760 h MILP.
        evaluate_8760:
            After typical-day sizing, freeze \(x_k\) and LP-dispatch 8760 h.

    Returns:
        Dict with ``x``, TCO, LPSP, hardware totals, and DataFrames.
    """

    if costs is None:
        costs = MilpCostParams()
    if sizing_csv is None:
        sizing_csv = SIZING_CSV
    if dispatch_csv is None:
        dispatch_csv = DISPATCH_CSV
    packs = milp_battery_catalog()
    table = load_pv_load_table(load_csv)

    if horizon == "full":
        pv = table["pv_output_kW"].to_numpy(dtype=np.float64)
        load = table["load_demand_kW"].to_numpy(dtype=np.float64)
        w = np.ones(len(table), dtype=np.float64)
        wrap: Literal["horizon", "daily"] = "horizon"
        tlim = time_limit_s if time_limit_s is not None else SOLVER_TIME_FULL_S
    else:
        pv, load, w = typical_month_days(table)
        wrap = "daily"
        tlim = time_limit_s if time_limit_s is not None else SOLVER_TIME_TYPICAL_S

    milp = build_and_solve_milp(
        pv_kw=pv,
        load_kw=load,
        hour_weights=w,
        packs=packs,
        budget_limit=float(budget_limit),
        max_allowed_outages=max_allowed_outages,
        costs=costs,
        wrap=wrap,
        time_limit_s=tlim,
    )
    milp["budget_limit"] = float(budget_limit)
    milp["horizon"] = horizon
    bom = _write_sizing_csv(packs, milp["x"], milp, Path(sizing_csv))  # type: ignore[arg-type]
    milp["bom"] = bom

    dispatch_out = milp["dispatch"]
    assert isinstance(dispatch_out, pd.DataFrame)
    if evaluate_8760 and horizon == "typical":
        year = evaluate_full_year_dispatch(
            table,
            packs,
            milp["x"],  # type: ignore[arg-type]
            costs,
            time_limit_s=min(tlim, 90),
        )
        milp.update(year)
        dispatch_out = year["dispatch_8760"]  # type: ignore[assignment]
        assert isinstance(dispatch_out, pd.DataFrame)
        dispatch_out.to_csv(dispatch_csv, index=False)
        milp["dispatch_path"] = str(Path(dispatch_csv).resolve())
    elif horizon == "full":
        dispatch_out.to_csv(dispatch_csv, index=False)
        milp["dispatch_path"] = str(Path(dispatch_csv).resolve())
        milp["lpsp_8760"] = milp["lpsp"]
        milp["outage_hours_8760"] = milp["outage_hours_year"]

    milp["sizing_path"] = str(Path(sizing_csv).resolve())
    _print_report(milp, packs)
    return milp


def _print_report(milp: dict[str, object], packs: Sequence[CommercialBattery]) -> None:
    x_hat = milp["x"]
    assert isinstance(x_hat, dict)
    print("=== optimal storage BOM ===")
    for p in packs:
        n_u = int(x_hat.get(p.model_name, 0))
        if n_u:
            print(
                f"  {n_u} × {p.model_name}: "
                f"{n_u * p.capacity_kWh:.1f} kWh, "
                f"{n_u * p.continuous_power_kW:.1f} kW, "
                f"${n_u * p.cost_usd:,.0f}"
            )
    print(
        f"capex=${float(milp['capex_usd']):,.0f}  "
        f"TCO({float(MilpCostParams().horizon_years):.0f}y)="
        f"${float(milp['tco_usd']):,.0f}"
    )
    print(
        f"typical-day LPSP={float(milp['lpsp']):.4f}  "
        f"(ε_applied={float(milp.get('epsilon_lpsp_applied', 0.0)):.4f})  "
        f"outage-h≈{float(milp['outage_hours_year']):.1f}  "
        f"E={float(milp['e_capacity_kWh']):.1f} kWh  "
        f"P={float(milp['p_capacity_kW']):.1f} kW"
    )
    if "lpsp_8760" in milp:
        print(
            f"8760 LPSP={float(milp['lpsp_8760']):.4f}  "
            f"outage-h={float(milp['outage_hours_8760']):.1f}  "
            f"def={float(milp.get('def_energy_kWh_8760', milp['def_energy_kWh_year'])):.1f} kWh"
        )
    print(f"wrote {milp.get('sizing_path', '')}")


def main() -> None:
    r"""CLI: default $10k budget, LPSP 5% (energy-gap floor if infeasible)."""

    solve_optimal_storage(budget_limit=10000.0, max_allowed_outages=None)


if __name__ == "__main__":
    main()

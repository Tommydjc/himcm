"""
Pareto sizing (CapEx vs mean wait) + thunderstorm delay stress tests.

Reads autonomous planning candidates, densifies with a feasible grid,
exports ``results/optimal_kiosk_sizing.csv``, and writes three 300 DPI
paper figures under ``paper_figures/``.

Run (pack root):
  PYTHONPATH=src python -m optimization.pareto_sizing
  # or:
  python -m src.optimization.pareto_sizing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.sankey import Sankey

PACK_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PACK_ROOT / "src"
for _p in (str(SRC_ROOT), str(PACK_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from des_env.airport_sim import STORM_ONSET_MIN, run_airport_simulation
from mcp_server.server import (
    AREA_CAP_M2,
    BUDGET_CAP_USD,
    validate_infrastructure_budget,
)

PLANNING_JSON = PACK_ROOT / "results" / "autonomous_planning_summary.json"
PARETO_CSV = PACK_ROOT / "results" / "optimal_kiosk_sizing.csv"
STRESS_CSV = PACK_ROOT / "results" / "thunderstorm_stress_comparison.csv"
FIG_DIR = PACK_ROOT / "paper_figures"

BASELINE_EMPIRICAL = {"c_fast": 12, "c_slow": 15, "c_bank": 2}


def load_planning_candidates(
    path: Path = PLANNING_JSON,
) -> List[Dict[str, Any]]:
    """Load candidate points from autonomous planning summary."""
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    cands = list(payload.get("all_candidates") or [])
    if not cands:
        for row in payload.get("subtask_results") or []:
            if row.get("status") == "ok" and "candidate" in row:
                cands.append(row["candidate"])
    return cands


def is_dominated_2d(
    cost_a: float,
    wait_a: float,
    cost_b: float,
    wait_b: float,
) -> bool:
    """True if A dominates B on min-cost / min-wait."""
    le = cost_a <= cost_b + 1e-12 and wait_a <= wait_b + 1e-12
    lt = cost_a < cost_b - 1e-12 or wait_a < wait_b - 1e-12
    return bool(le and lt)


def extract_pareto_front_2d(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Non-dominated sort on (total_cost, mean_wait_time)."""
    front: List[Dict[str, Any]] = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if is_dominated_2d(
                float(b["total_cost"]),
                float(b["mean_wait_time"]),
                float(a["total_cost"]),
                float(a["mean_wait_time"]),
            ):
                dominated = True
                break
        if not dominated:
            front.append(dict(a))
    front.sort(key=lambda r: (float(r["total_cost"]), float(r["mean_wait_time"])))
    return front


def evaluate_design(
    c_fast: int,
    c_slow: int,
    c_bank: int,
    *,
    pricing_mode: float = 1.0,
    seed: int = 42,
    storm_injection: bool = False,
    scenario: Optional[str] = None,
    lambda_base: Optional[float] = None,
    peak_multiplier: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate budget/area then run DES; return KPI row."""
    budget = validate_infrastructure_budget(
        c_fast, c_slow, c_bank, budget_cap_usd=BUDGET_CAP_USD, area_cap_m2=AREA_CAP_M2
    )
    kwargs: Dict[str, Any] = {
        "c_fast": c_fast,
        "c_slow": c_slow,
        "c_bank": c_bank,
        "pricing_mode": pricing_mode,
        "seed": seed,
        "storm_injection": storm_injection,
    }
    if scenario is not None:
        kwargs["scenario"] = scenario
    if lambda_base is not None:
        kwargs["lambda_base"] = float(lambda_base)
    if peak_multiplier is not None:
        kwargs["peak_multiplier"] = float(peak_multiplier)
    metrics = run_airport_simulation(**kwargs)
    return {
        "c_fast": c_fast,
        "c_slow": c_slow,
        "c_bank": c_bank,
        "total_cost": float(budget["total_cost"]),
        "total_area": float(budget["total_area"]),
        "budget_valid": bool(budget["is_valid"]),
        "mean_wait_time": float(metrics["mean_wait_time"]),
        "reneging_rate": float(metrics["reneging_rate"]),
        "max_queue_len": int(metrics["max_queue_len"]),
        "max_system_load": int(metrics.get("max_system_load", 0)),
        "utilization_fast": float(metrics["utilization_fast"]),
        "utilization_slow": float(metrics["utilization_slow"]),
        "served_fast": int(metrics["served_fast"]),
        "served_slow": int(metrics["served_slow"]),
        "served_powerbank": int(metrics["served_powerbank"]),
        "n_arrivals": int(metrics["n_arrivals"]),
        "n_anxious": int(metrics["anxious_passengers"]),
        "n_storm_delayed": int(metrics.get("n_storm_delayed", 0)),
        "flow_calm": int(metrics.get("flow_calm", 0)),
        "flow_to_fast": int(metrics.get("flow_to_fast", 0)),
        "flow_to_slow": int(metrics.get("flow_to_slow", 0)),
        "flow_to_powerbank": int(metrics.get("flow_to_powerbank", 0)),
        "flow_to_abort": int(metrics.get("flow_to_abort", 0)),
        "storm_injection": bool(metrics.get("storm_injection", storm_injection)),
        "pricing_mode": float(pricing_mode),
        "queue_timeline": metrics.get("queue_timeline", []),
    }


def densify_design_grid(
    *,
    budget_cap: float = BUDGET_CAP_USD,
    seed: int = 7,
    lambda_base: float = 0.55,
    peak_multiplier: float = 4.0,
) -> List[Dict[str, Any]]:
    """
    Evaluate a coarse feasible grid to densify the CapEx–wait front.

    Uses a morning-rush-like intensity so mean wait meaningfully trades off
    against CapEx (off-peak λ leaves most designs with wait≈0).
    """
    rows: List[Dict[str, Any]] = []
    for c_fast in range(2, 21, 2):
        for c_slow in range(0, 17, 4):
            for c_bank in range(0, 5):
                info = validate_infrastructure_budget(
                    c_fast,
                    c_slow,
                    c_bank,
                    budget_cap_usd=budget_cap,
                    area_cap_m2=AREA_CAP_M2,
                )
                if not info["is_valid"]:
                    continue
                # Skip degenerate packs with no slow/bank redundancy for kiosk study.
                if c_fast + c_slow + c_bank < 4:
                    continue
                row = evaluate_design(
                    c_fast,
                    c_slow,
                    c_bank,
                    pricing_mode=1.0,
                    seed=seed,
                    lambda_base=lambda_base,
                    peak_multiplier=peak_multiplier,
                )
                row["source"] = "grid_rush"
                row.pop("queue_timeline", None)
                rows.append(row)
    return rows


def select_knee_near_budget(
    pareto: Sequence[Dict[str, Any]],
    budget_mark: float = BUDGET_CAP_USD,
    *,
    max_reneging: float = 0.05,
) -> Dict[str, Any]:
    """
    Pick the Pareto point nearest the $80k mark with acceptable reneging.

    Preference order:
    1. cost ≤ budget_mark and reneging ≤ max_reneging → min wait, then max cost
       (spend remaining budget to harden service);
    2. else reneging-feasible points → nearest cost to budget_mark;
    3. else any Pareto → nearest cost.
    """
    if not pareto:
        raise ValueError("empty Pareto front")
    feasible = [
        p for p in pareto if float(p.get("reneging_rate", 1.0)) <= max_reneging + 1e-12
    ]
    pool = feasible if feasible else list(pareto)
    under = [p for p in pool if float(p["total_cost"]) <= budget_mark + 1e-9]
    if under:
        return min(
            under,
            key=lambda p: (float(p["mean_wait_time"]), -float(p["total_cost"])),
        )
    return min(pool, key=lambda p: abs(float(p["total_cost"]) - budget_mark))


def plot_pareto_cost_vs_wait(
    all_rows: Sequence[Dict[str, Any]],
    pareto: Sequence[Dict[str, Any]],
    knee: Dict[str, Any],
    out_path: Path,
) -> None:
    """Figure 1: CapEx vs mean wait with Pareto curve and $80k knee."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=300)

    xs = [float(r["total_cost"]) for r in all_rows]
    ys = [float(r["mean_wait_time"]) for r in all_rows]
    ax.scatter(xs, ys, s=28, c="#95a5a6", alpha=0.55, label="Evaluated designs", zorder=2)

    px = [float(r["total_cost"]) for r in pareto]
    py = [float(r["mean_wait_time"]) for r in pareto]
    ax.plot(px, py, "-o", color="#2980b9", linewidth=2.0, markersize=7, label="Pareto front", zorder=3)

    ax.axvline(
        BUDGET_CAP_USD,
        color="#e67e22",
        linestyle="--",
        linewidth=1.4,
        label="Budget cap $80,000",
    )
    ax.scatter(
        [float(knee["total_cost"])],
        [float(knee["mean_wait_time"])],
        s=140,
        c="#c0392b",
        marker="*",
        zorder=4,
        label=(
            f"Knee @ ${knee['total_cost']:,.0f} "
            f"({knee['c_fast']}/{knee['c_slow']}/{knee['c_bank']})"
        ),
    )
    ax.set_xlabel("Capital expenditure CapEx (USD)")
    ax.set_ylabel("Mean wait time (minutes)")
    ax.set_title("Pareto front: CapEx vs mean charging wait")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_queue_delay_stress(
    baseline_tl: Sequence[Dict[str, Any]],
    pareto_tl: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    baseline_label: str,
    pareto_label: str,
) -> None:
    """Figure 2: 16h queue/system-load under storm; red band after 17:00."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=300)

    def _series(
        tl: Sequence[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.asarray([row["time_minutes"] for row in tl], dtype=float) / 60.0 + 6.0
        q = np.asarray(
            [row.get("queue_length", row.get("system_load", 0.0)) for row in tl],
            dtype=float,
        )
        load = np.asarray(
            [row.get("system_load", row.get("queue_length", 0.0)) for row in tl],
            dtype=float,
        )
        return t, load, q

    t_b, load_b, q_b = _series(baseline_tl)
    t_p, load_p, q_p = _series(pareto_tl)

    storm_start_h = STORM_ONSET_MIN / 60.0 + 6.0
    ax.axvspan(storm_start_h, 22.0, color="#e74c3c", alpha=0.15, label="Thunderstorm delay window")

    ax.plot(t_b, load_b, color="#7f8c8d", linewidth=1.8, label=f"{baseline_label} system load")
    ax.plot(t_p, load_p, color="#2980b9", linewidth=2.0, label=f"{pareto_label} system load")
    ax.plot(t_b, q_b, color="#7f8c8d", linewidth=1.0, linestyle="--", alpha=0.8, label=f"{baseline_label} wait queue")
    ax.plot(t_p, q_p, color="#c0392b", linewidth=1.2, linestyle="--", alpha=0.9, label=f"{pareto_label} wait queue")

    ax.set_xlim(6, 22)
    ax.set_xlabel("Time of day (hours)")
    ax.set_ylabel("Passengers")
    ax.set_title("Thunderstorm stress: queue & system load (06:00–22:00)")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bdi_sankey(
    flow: Dict[str, int],
    out_path: Path,
) -> None:
    """Figure 3: BDI intention flow (Sankey + outcome bars)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calm = int(flow.get("flow_calm", 0))
    fast = int(flow.get("flow_to_fast", 0))
    slow = int(flow.get("flow_to_slow", 0))
    bank = int(flow.get("flow_to_powerbank", 0))
    abort = int(flow.get("flow_to_abort", 0))
    anxious = fast + slow + bank + abort
    arrivals = calm + anxious
    if arrivals <= 0:
        arrivals = 1
    if anxious <= 0:
        anxious = 1

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), dpi=300)

    # Left: arrivals → calm / anxious (matplotlib Sankey, single stage).
    ax0 = axes[0]
    ax0.set_xticks([])
    ax0.set_yticks([])
    s0 = 1.0 / float(arrivals)
    sankey = Sankey(ax=ax0, scale=1.0, gap=0.4, offset=0.2, format="%.0f", unit=None)
    sankey.add(
        flows=[arrivals * s0, -calm * s0, -anxious * s0],
        labels=["Arrivals", f"Calm IDLE\n({calm})", f"Anxious\n({anxious})"],
        orientations=[0, 1, -1],
        facecolor="#3498db",
    )
    sankey.finish()
    ax0.set_title("Stage A: arrival → anxiety screen")

    # Right: anxious → outcomes (horizontal bars; robust vs Sankey connect bugs).
    ax1 = axes[1]
    labels = ["Fast 65W", "Slow 15W", "Powerbank", "Reneging / Abort"]
    values = [fast, slow, bank, abort]
    colors = ["#3498db", "#9b59b6", "#2ecc71", "#e74c3c"]
    ypos = np.arange(len(labels))
    ax1.barh(ypos, values, color=colors, edgecolor="white")
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Passenger count")
    ax1.set_title("Stage B: BDI intention outcomes (anxious seekers)")
    for y, v in zip(ypos, values):
        ax1.text(v + max(values) * 0.01 + 0.5, y, str(v), va="center", fontsize=9)
    ax1.grid(True, axis="x", linestyle=":", alpha=0.5)

    fig.suptitle(
        "BDI passenger intention flow under thunderstorm stress",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_pareto_sizing_pipeline(
    *,
    planning_json: Path = PLANNING_JSON,
    export_csv: Path = PARETO_CSV,
    densify: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline: Pareto extract → storm stress → three paper figures.

    Returns
    -------
    dict
        Paths and selected knee / stress KPIs.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (PACK_ROOT / "results").mkdir(parents=True, exist_ok=True)

    # --- Collect designs (re-simulate planning points under rush intensity) -
    rush_lambda = 0.55
    rush_peak = 4.0
    planning = load_planning_candidates(planning_json)
    rows: List[Dict[str, Any]] = []
    seen_hw: set[Tuple[int, int, int]] = set()
    for c in planning:
        params = c.get("params") or {}
        hw = (
            int(params.get("c_fast", 0)),
            int(params.get("c_slow", 0)),
            int(params.get("c_bank", 0)),
        )
        if hw in seen_hw:
            continue
        seen_hw.add(hw)
        row = evaluate_design(
            hw[0],
            hw[1],
            hw[2],
            pricing_mode=1.0,
            seed=11,
            lambda_base=rush_lambda,
            peak_multiplier=rush_peak,
        )
        row["source"] = f"planning:{c.get('subtask_id', 'unknown')}"
        row.pop("queue_timeline", None)
        rows.append(row)

    if densify:
        print("[pareto] densifying feasible grid under rush intensity...")
        rows.extend(
            densify_design_grid(
                seed=11, lambda_base=rush_lambda, peak_multiplier=rush_peak
            )
        )

    # Deduplicate by hardware triple (keep lowest wait).
    best: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for r in rows:
        key = (int(r["c_fast"]), int(r["c_slow"]), int(r["c_bank"]))
        prev = best.get(key)
        if prev is None or float(r["mean_wait_time"]) < float(prev["mean_wait_time"]):
            best[key] = r
    uniq = list(best.values())

    # Service-quality filter: keep designs with reneging ≤ 5% for the kiosk
    # sizing front (pure CapEx–wait Pareto otherwise collapses to tiny packs).
    serviceable = [
        r for r in uniq if float(r.get("reneging_rate", 1.0)) <= 0.05 + 1e-12
    ]
    if len(serviceable) < 5:
        print(
            f"[pareto] warning: only {len(serviceable)} designs meet reneging≤5%; "
            "loosening filter to full set"
        )
        serviceable = uniq
    pareto = extract_pareto_front_2d(serviceable)
    knee = select_knee_near_budget(pareto, BUDGET_CAP_USD, max_reneging=0.05)

    # Mark Pareto / knee flags for CSV.
    pareto_keys = {
        (int(p["c_fast"]), int(p["c_slow"]), int(p["c_bank"])) for p in pareto
    }
    knee_key = (int(knee["c_fast"]), int(knee["c_slow"]), int(knee["c_bank"]))
    for r in uniq:
        key = (int(r["c_fast"]), int(r["c_slow"]), int(r["c_bank"]))
        r["is_pareto"] = key in pareto_keys
        r["is_knee_80k"] = key == knee_key

    df = pd.DataFrame(uniq).sort_values(
        ["is_pareto", "total_cost", "mean_wait_time"], ascending=[False, True, True]
    )
    df.to_csv(export_csv, index=False)
    print(f"[pareto] wrote {export_csv} ({len(df)} rows, {len(pareto)} Pareto)")

    fig1 = FIG_DIR / "fig_pareto_cost_vs_wait.png"
    plot_pareto_cost_vs_wait(uniq, pareto, knee, fig1)
    print(f"[pareto] figure1 {fig1}")

    # --- Thunderstorm stress: baseline empirical vs Pareto knee ------------
    print("[stress] running thunderstorm injection on baseline vs knee...")
    baseline_storm = evaluate_design(
        BASELINE_EMPIRICAL["c_fast"],
        BASELINE_EMPIRICAL["c_slow"],
        BASELINE_EMPIRICAL["c_bank"],
        pricing_mode=1.0,
        seed=99,
        storm_injection=True,
    )
    knee_storm = evaluate_design(
        int(knee["c_fast"]),
        int(knee["c_slow"]),
        int(knee["c_bank"]),
        pricing_mode=1.0,
        seed=99,
        storm_injection=True,
    )

    stress_df = pd.DataFrame(
        [
            {
                "label": "baseline_empirical_12_15_2",
                "c_fast": baseline_storm["c_fast"],
                "c_slow": baseline_storm["c_slow"],
                "c_bank": baseline_storm["c_bank"],
                "total_cost": baseline_storm["total_cost"],
                "mean_wait_time": baseline_storm["mean_wait_time"],
                "reneging_rate": baseline_storm["reneging_rate"],
                "max_queue_len": baseline_storm["max_queue_len"],
                "max_system_load": baseline_storm["max_system_load"],
                "n_storm_delayed": baseline_storm["n_storm_delayed"],
                "n_arrivals": baseline_storm["n_arrivals"],
            },
            {
                "label": "pareto_knee",
                "c_fast": knee_storm["c_fast"],
                "c_slow": knee_storm["c_slow"],
                "c_bank": knee_storm["c_bank"],
                "total_cost": knee_storm["total_cost"],
                "mean_wait_time": knee_storm["mean_wait_time"],
                "reneging_rate": knee_storm["reneging_rate"],
                "max_queue_len": knee_storm["max_queue_len"],
                "max_system_load": knee_storm["max_system_load"],
                "n_storm_delayed": knee_storm["n_storm_delayed"],
                "n_arrivals": knee_storm["n_arrivals"],
            },
        ]
    )
    stress_df.to_csv(STRESS_CSV, index=False)
    print(f"[stress] wrote {STRESS_CSV}")

    fig2 = FIG_DIR / "fig_queue_length_delay_stress.png"
    plot_queue_delay_stress(
        baseline_storm["queue_timeline"],
        knee_storm["queue_timeline"],
        fig2,
        baseline_label="Baseline 12/15/2",
        pareto_label=(
            f"Pareto knee {knee_storm['c_fast']}/"
            f"{knee_storm['c_slow']}/{knee_storm['c_bank']}"
        ),
    )
    print(f"[stress] figure2 {fig2}")

    # Sankey from storm knee (or normal knee re-run with flows).
    fig3 = FIG_DIR / "fig_bdi_intention_flow_sankey.png"
    plot_bdi_sankey(knee_storm, fig3)
    print(f"[flow] figure3 {fig3}")

    return {
        "pareto_csv": str(export_csv),
        "stress_csv": str(STRESS_CSV),
        "fig_pareto": str(fig1),
        "fig_stress": str(fig2),
        "fig_sankey": str(fig3),
        "knee": {
            k: knee[k]
            for k in (
                "c_fast",
                "c_slow",
                "c_bank",
                "total_cost",
                "mean_wait_time",
                "reneging_rate",
            )
            if k in knee
        },
        "n_evaluated": len(uniq),
        "n_pareto": len(pareto),
        "stress": {
            "baseline_reneging": baseline_storm["reneging_rate"],
            "knee_reneging": knee_storm["reneging_rate"],
            "baseline_max_queue": baseline_storm["max_queue_len"],
            "knee_max_queue": knee_storm["max_queue_len"],
        },
    }


def main() -> None:
    """CLI entry."""
    summary = run_pareto_sizing_pipeline()
    print(json.dumps({k: v for k, v in summary.items() if k != "knee"}, indent=2))
    print("knee:", summary["knee"])
    print("stress:", summary["stress"])


if __name__ == "__main__":
    main()

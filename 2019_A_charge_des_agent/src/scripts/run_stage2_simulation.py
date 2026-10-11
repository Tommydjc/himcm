"""
Stage-2 baseline runner: SimPy airport charging DES + BDI passengers.

Writes:
  - results/stage2_baseline_simulation.csv
  - results/stage2_baseline_metrics.csv
  - paper_figures/stage2_queue_baseline.png

Run from the pack root:
  PYTHONPATH=src python src/scripts/run_stage2_simulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PACK_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PACK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from des_env.airport_sim import run_airport_simulation


def main() -> None:
    """Execute baseline hardware config and export CSV + dynamics figure."""
    results_dir = PACK_ROOT / "results"
    figures_dir = PACK_ROOT / "paper_figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_config = {
        "c_fast": 12,
        "c_slow": 15,
        "c_bank": 2,
        "pricing_mode": 1.0,
        "seed": 42,
        "scenario": "normal",
    }

    print("=" * 70)
    print("[Stage 2] Airport terminal charging DES (SimPy + BDI passenger)")
    print("=" * 70)
    print("Baseline hardware:")
    print(f"  65W fast chargers : {baseline_config['c_fast']}")
    print(f"  15W slow lockers  : {baseline_config['c_slow']}")
    print(f"  Powerbank kiosks  : {baseline_config['c_bank']} groups")
    print("Advancing 16 h DES clock (06:00-22:00)...")

    metrics = run_airport_simulation(
        c_fast=int(baseline_config["c_fast"]),
        c_slow=int(baseline_config["c_slow"]),
        c_bank=int(baseline_config["c_bank"]),
        pricing_mode=float(baseline_config["pricing_mode"]),
        seed=int(baseline_config["seed"]),
        scenario=str(baseline_config["scenario"]),
    )
    queue_timeline = metrics.get("queue_timeline", [])

    print()
    print("=" * 70)
    print("SimPy telemetry summary")
    print("=" * 70)
    print(f"Total arrivals              : {metrics['total_passengers']}")
    print(f"Anxious passengers (E<15%)  : {metrics['anxious_passengers']}")
    print(f"Served on fast chargers     : {metrics['served_fast']}")
    print(f"Served on slow lockers      : {metrics['served_slow']}")
    print(f"Powerbank rentals           : {metrics['served_powerbank']}")
    print("-" * 70)
    print(f"Reneged passengers          : {metrics['reneged_passengers']}")
    print(f"Reneging rate               : {metrics['reneging_rate'] * 100:.2f}%")
    print(f"Mean wait time (min)        : {metrics['mean_wait_time']:.2f}")
    print(f"Max waiting queue           : {metrics['max_queue_len']}")
    print(f"Max system load (q+serve)   : {metrics['max_system_load']}")
    print(f"Utilization fast            : {metrics['utilization_fast'] * 100:.1f}%")
    print(f"Utilization slow            : {metrics['utilization_slow'] * 100:.1f}%")
    print("=" * 70)

    metrics_path = results_dir / "stage2_baseline_metrics.csv"
    scalar_keys = [
        "total_passengers",
        "anxious_passengers",
        "served_fast",
        "served_slow",
        "served_powerbank",
        "reneged_passengers",
        "reneging_rate",
        "mean_wait_time",
        "max_queue_len",
        "max_system_load",
        "utilization_fast",
        "utilization_slow",
        "n_seekers",
        "c_fast",
        "c_slow",
        "c_bank",
        "pricing_mode",
        "seed",
        "horizon_min",
        "lambda_base",
        "peak_multiplier",
        "scenario",
    ]
    pd.DataFrame([{k: metrics[k] for k in scalar_keys}]).to_csv(
        metrics_path, index=False
    )
    print(f"Metrics CSV : {metrics_path}")

    if not queue_timeline:
        print("WARNING: empty queue_timeline; skip figure export.")
        return

    df_timeline = pd.DataFrame(queue_timeline)
    csv_path = results_dir / "stage2_baseline_simulation.csv"
    df_timeline.to_csv(csv_path, index=False)
    print(f"Timeline CSV: {csv_path}")

    # Dual-axis figure: system load / wait queue vs NHPP intensity.
    # Waiting-only series was flat at 0 under over-provisioned baseline hardware;
    # system_load = waiting + in_service is the correct congestion series.
    fig_path = figures_dir / "stage2_queue_baseline.png"
    time_hours = df_timeline["time_minutes"] / 60.0 + 6.0

    fig, ax1 = plt.subplots(figsize=(10, 4.8), dpi=300)
    ax1.fill_between(
        time_hours,
        df_timeline["system_load"],
        color="#3498db",
        alpha=0.20,
        label="System load (wait + in service)",
    )
    ax1.plot(
        time_hours,
        df_timeline["system_load"],
        color="#2980b9",
        linewidth=1.8,
    )
    ax1.plot(
        time_hours,
        df_timeline["in_service"],
        color="#27ae60",
        linewidth=1.4,
        label="In service",
    )
    ax1.plot(
        time_hours,
        df_timeline["queue_length"],
        color="#e74c3c",
        linewidth=1.2,
        linestyle="--",
        label="Waiting queue only",
    )
    ax1.axhline(y=15, color="orange", linestyle=":", label="Warning threshold (15)")
    ax1.set_xlabel("Time of day (hours)")
    ax1.set_ylabel("Passengers")
    ax1.set_xlim(6, 22)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(
        time_hours,
        df_timeline["lambda_t"],
        color="#8e44ad",
        linewidth=1.2,
        alpha=0.85,
        label="Arrival intensity λ(t)",
    )
    ax2.set_ylabel("Arrival rate (passengers / min)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.set_title(
        "Stage 2 baseline: charging congestion + NHPP waves (06:00-22:00)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Figure PNG : {fig_path}")
    print(
        "Note: waiting queue can be ~0 when C_fast/C_slow are large; "
        "use system_load for congestion dynamics."
    )


if __name__ == "__main__":
    main()

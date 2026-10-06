"""Build the 8760-hour PV / load net-power table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.generation.solar_pv import PVArrayParams, simulate_pv_timeseries
from src.load_profiler.mcmc_load import LoadSynthParams, synthesize_load_kw

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
RESULTS_DIR: Path = PACK_ROOT / "results"
OUT_CSV: Path = RESULTS_DIR / "annual_pv_load_8760h.csv"


def run_annual_pv_load(
    out_csv: Path | None = None,
    pv_params: PVArrayParams | None = None,
    load_params: LoadSynthParams | None = None,
) -> pd.DataFrame:
    """Simulate PV and load, write ``[hour, month, pv, load, net]``.

    ``net_power_kW = pv_output_kW - load_demand_kW`` (positive = surplus).
    """

    if out_csv is None:
        out_csv = OUT_CSV
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if pv_params is None:
        pv_params = PVArrayParams()
    if load_params is None:
        load_params = LoadSynthParams()

    pv = simulate_pv_timeseries(params=pv_params)
    load = synthesize_load_kw(
        hour_local=pv["hour"].to_numpy(),
        t_amb_c=pv["temperature_c"].to_numpy(),
        params=load_params,
    )
    table = pd.DataFrame(
        {
            "hour": pv["hour"].to_numpy(dtype=int),
            "month": pv["month"].to_numpy(dtype=int),
            "pv_output_kW": pv["pv_output_kW"].to_numpy(dtype=float),
            "load_demand_kW": load["load_demand_kW"].to_numpy(dtype=float),
        }
    )
    table["net_power_kW"] = table["pv_output_kW"] - table["load_demand_kW"]
    table.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  rows={len(table)}")
    print(
        f"PV mean={table['pv_output_kW'].mean():.3f} kW  "
        f"max={table['pv_output_kW'].max():.3f} kW  "
        f"annual={table['pv_output_kW'].sum():.1f} kWh"
    )
    print(
        f"load mean={table['load_demand_kW'].mean():.3f} kW  "
        f"max={table['load_demand_kW'].max():.3f} kW  "
        f"annual={table['load_demand_kW'].sum():.1f} kWh"
    )
    print(
        f"rigid-target check: load HVAC mean="
        f"{load['load_hvac_kW'].mean():.3f} kW  "
        f"behavior mean={load['load_behavior_kW'].mean():.3f} kW"
    )
    return table


def main() -> None:
    """CLI."""

    run_annual_pv_load()


if __name__ == "__main__":
    main()

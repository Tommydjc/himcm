r"""Map a ``LoadSheddingAgreement`` onto physical household kW.

Neuro-symbolic step: LLM names stay symbolic; this adapter is the only
place they become watts. Cuts never go negative and never raise load.

HVAC law (Task 2 spec)
    \[
    P_{\mathrm{hvac}}'
    = P_{\mathrm{hvac}}\bigl(1-0.08\cdot\Delta T_{\mathrm{set}}\bigr),
    \]
with \(\Delta T_{\mathrm{set}}=\texttt{hvac\_temp\_offset\_c}\ge 0\)
(degrees the setpoint is *lowered*). Appliance sheds subtract
\(P_{\mathrm{rated}}\times\mathrm{duty}\) in catalog peak hours.
Residual load is clipped to the rigid floor \(0.3\,\mathrm{kW}\)
without inventing energy when the raw load is already below the floor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.agent_sim.crew_engine import CATALOG_CSV, PROTECTED, load_appliance_catalog
from src.agent_sim.models import LoadSheddingAgreement
from src.load_profiler.mcmc_load import HVAC_NAME, _hour_in_peaks

RIGID_FLOOR_KW: float = 0.30
HVAC_CUT_FRAC_PER_C: float = 0.08


class AdaptiveLoadAdapter:
    """Catalog-backed curtailment of a crewAI load-shedding contract."""

    def __init__(self, catalog_path: Path | None = None) -> None:
        """Load the appliance dictionary once.

        Args:
            catalog_path:
                Optional override of ``data/raw/appliances_catalog.csv``.
        """

        csv_path = Path(catalog_path) if catalog_path is not None else CATALOG_CSV
        frame = load_appliance_catalog(csv_path)
        self.catalog: pd.DataFrame = frame.set_index("appliance_name")
        self.catalog_path: Path = csv_path

    def appliance_cut_kw(
        self,
        name: str,
        hour_of_day: np.ndarray,
    ) -> np.ndarray:
        r"""Expected kW of ``name`` at each hour: \(P_{\mathrm{rated}}\eta_{\mathrm{duty}}\) in peaks."""

        hour = np.asarray(hour_of_day, dtype=np.int64)
        if name not in self.catalog.index:
            return np.zeros(hour.shape, dtype=np.float64)
        if name in PROTECTED:
            return np.zeros(hour.shape, dtype=np.float64)
        row = self.catalog.loc[name]
        rated = float(row["rated_power_W"]) / 1000.0
        duty = float(row["duty_cycle"])
        in_peak = _hour_in_peaks(hour, str(row["peak_hours"]))
        return np.where(in_peak, rated * duty, 0.0).astype(np.float64)

    def hvac_cut_kw(
        self,
        hour_of_day: np.ndarray,
        offset_c: float,
        shed_hvac: bool,
    ) -> np.ndarray:
        r"""HVAC reduction: full coincidence if shed, else \(8\%/^\circ\mathrm{C}\)."""

        profile = self.appliance_cut_kw(HVAC_NAME, hour_of_day)
        if HVAC_NAME in PROTECTED:
            return np.zeros(profile.shape, dtype=np.float64)
        if shed_hvac:
            return profile
        frac = min(max(float(offset_c), 0.0) * HVAC_CUT_FRAC_PER_C, 1.0)
        return profile * frac

    def apply_curtailment(
        self,
        raw_load_kW: float | np.ndarray,
        agreement: LoadSheddingAgreement,
        hour_of_day: int | np.ndarray,
    ) -> np.ndarray:
        r"""Return post-DSR load (kW), same shape as ``raw_load_kW``.

        Args:
            raw_load_kW:
                Pre-negotiation household demand (kW), scalar or ``(T,)``.
            agreement:
                Structured crewAI / mock contract.
            hour_of_day:
                Local hour in \{0,...,23\}, broadcastable to the load.

        Returns:
            Curtailed load \(\max(P_{\mathrm{raw}}-\Delta P, P_{\mathrm{floor}})\),
            with floor applied only when raw load itself is at least the floor.
        """

        raw = np.asarray(raw_load_kW, dtype=np.float64)
        hour = np.asarray(hour_of_day, dtype=np.int64)
        if hour.shape != raw.shape:
            hour = np.broadcast_to(hour, raw.shape).copy()
        cut = np.zeros(raw.shape, dtype=np.float64)
        shed_hvac = HVAC_NAME in set(agreement.shed_appliances)
        for name in agreement.shed_appliances:
            if name == HVAC_NAME:
                continue
            cut += self.appliance_cut_kw(name, hour)
        cut += self.hvac_cut_kw(hour, agreement.hvac_temp_offset_c, shed_hvac)
        cut = np.minimum(np.maximum(cut, 0.0), np.maximum(raw, 0.0))
        residual = raw - cut
        floor = np.where(raw >= RIGID_FLOOR_KW, RIGID_FLOOR_KW, raw)
        return np.maximum(residual, floor)

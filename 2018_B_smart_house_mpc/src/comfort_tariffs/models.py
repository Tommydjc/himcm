"""
Teaching-track comfort + TOU models (Phase 2).

- ISO 7730 Fanger PMV–PPD with continuous |PMV|>0.5 quadratic penalty.
- CA/TX-style three-period TOU (off / mid / on-peak at $0.12 / $0.22 / $0.48).

Legacy week track continues to use ``fanger_pmv.py`` + ``tou_tariffs.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from comfort_tariffs.fanger_pmv import fanger_pmv

PACK_ROOT = Path(__file__).resolve().parents[2]
TOU_V2_PATH = PACK_ROOT / "data" / "raw" / "tou_tariffs_v2.json"

# Teaching defaults (summer light clothing, light activity).
DEFAULT_MET = 1.2
DEFAULT_CLO = 0.5
DEFAULT_VEL = 0.1
DEFAULT_RH = 50.0
PMV_COMFORT_BAND = 0.5


def calculate_pmv_ppd(
    t_air: float,
    t_rad: Optional[float] = None,
    v_air: float = DEFAULT_VEL,
    rh: float = DEFAULT_RH,
    met: float = DEFAULT_MET,
    clo: float = DEFAULT_CLO,
) -> Tuple[float, float]:
    """
    ISO 7730 Fanger PMV and PPD.

    Parameters
    ----------
    t_air :
        Air dry-bulb temperature [°C].
    t_rad :
        Mean radiant temperature [°C]; defaults to ``t_air``.
    v_air :
        Relative air speed [m/s] (default 0.1).
    rh :
        Relative humidity [%] (default 50).
    met, clo :
        Metabolic rate [met] and clothing [clo] (summer defaults 1.2 / 0.5).

    Returns
    -------
    (pmv, ppd)
        PMV in [-3, 3]; PPD in percent.
    """
    return fanger_pmv(
        float(t_air),
        float(rh),
        T_mrt_C=float(t_air if t_rad is None else t_rad),
        vel_m_s=float(v_air),
        met=float(met),
        clo=float(clo),
    )


def discomfort_cost(
    pmv: float,
    *,
    lambda_comf: float = 1.0,
    band: float = PMV_COMFORT_BAND,
) -> float:
    """
    Continuous comfort penalty outside the |PMV| ≤ band neutral zone.

    C_discomfort = λ_comf * max(0, |PMV| - band)²
    """
    excess = max(0.0, abs(float(pmv)) - float(band))
    return float(lambda_comf) * excess * excess


def discomfort_cost_from_air(
    t_air: float,
    *,
    t_rad: Optional[float] = None,
    rh: float = DEFAULT_RH,
    lambda_comf: float = 1.0,
    met: float = DEFAULT_MET,
    clo: float = DEFAULT_CLO,
) -> Dict[str, float]:
    """Convenience: PMV/PPD + discomfort cost for one indoor state."""
    pmv, ppd = calculate_pmv_ppd(t_air, t_rad=t_rad, rh=rh, met=met, clo=clo)
    return {
        "pmv": float(pmv),
        "ppd": float(ppd),
        "discomfort_cost": discomfort_cost(pmv, lambda_comf=lambda_comf),
    }


@dataclass
class TOUTariffSchedule:
    """
    Three-period time-of-use tariff calendar.

    Default teaching rates (USD/kWh):
      off-peak  23:00–07:00 → 0.12
      mid-peak  07:00–14:00 & 19:00–23:00 → 0.22
      on-peak   14:00–19:00 → 0.48
    """

    periods: List[Dict[str, Any]]
    surge_multiplier: float = 1.0
    currency: str = "USD"
    unit: str = "USD_per_kWh"

    @classmethod
    def from_json(
        cls,
        path: Path = TOU_V2_PATH,
        *,
        surge_multiplier: Optional[float] = None,
    ) -> "TOUTariffSchedule":
        """Load teaching TOU v2 JSON."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        surge = (
            float(surge_multiplier)
            if surge_multiplier is not None
            else float(raw.get("surge_multiplier_default", 1.0))
        )
        return cls(
            periods=list(raw["periods"]),
            surge_multiplier=surge,
            currency=str(raw.get("currency", "USD")),
            unit=str(raw.get("unit", "USD_per_kWh")),
        )

    def rate_usd_per_kwh(self, hour_of_day: int) -> float:
        """Return TOU rate for civil hour in [0, 23]."""
        hod = int(hour_of_day) % 24
        for period in self.periods:
            if hod in period["hours"]:
                return float(period["rate"]) * float(self.surge_multiplier)
        raise ValueError(f"hour {hod} not covered by TOU periods")

    def period_name(self, hour_of_day: int) -> str:
        """Return period name for civil hour."""
        hod = int(hour_of_day) % 24
        for period in self.periods:
            if hod in period["hours"]:
                return str(period["name"])
        raise ValueError(f"hour {hod} not covered by TOU periods")

    def is_on_peak(self, hour_of_day: int) -> bool:
        """True if hour is in the on-peak block."""
        return self.period_name(hour_of_day) == "on_peak"

    def bill_from_power_W(
        self,
        power_W: Sequence[float],
        hour_of_day: Sequence[int],
        *,
        dt_hours: float = 1.0,
    ) -> Dict[str, float]:
        """
        Accumulate energy [kWh] and cost [USD] from site electric power.

        Also reports on-peak energy and on-peak cost share.
        """
        p = np.asarray(power_W, dtype=float)
        h = np.asarray(hour_of_day, dtype=int)
        if p.shape != h.shape:
            raise ValueError("power_W and hour_of_day shape mismatch")
        energy = p * float(dt_hours) / 1000.0
        rates = np.asarray([self.rate_usd_per_kwh(int(hh)) for hh in h], dtype=float)
        cost = energy * rates
        peak_mask = np.asarray([self.is_on_peak(int(hh)) for hh in h], dtype=bool)
        peak_energy = float(np.sum(energy[peak_mask]))
        peak_cost = float(np.sum(cost[peak_mask]))
        total_cost = float(np.sum(cost))
        return {
            "energy_kwh": float(np.sum(energy)),
            "cost_usd": total_cost,
            "peak_energy_kwh": peak_energy,
            "peak_cost_usd": peak_cost,
            "peak_cost_share": (peak_cost / total_cost) if total_cost > 1e-12 else 0.0,
        }

    def hourly_rate_table(self) -> Dict[int, float]:
        """Map hour 0..23 → rate (USD/kWh)."""
        return {h: self.rate_usd_per_kwh(h) for h in range(24)}


def load_tou_v2(
    path: Path = TOU_V2_PATH,
    *,
    surge_multiplier: Optional[float] = None,
) -> TOUTariffSchedule:
    """Alias for ``TOUTariffSchedule.from_json``."""
    return TOUTariffSchedule.from_json(path, surge_multiplier=surge_multiplier)


def main() -> None:
    """Phase-2 self-check: PMV≈0 at 24°C; peak rate = 0.48."""
    print("=== Phase 2 comfort + TOU v2 ===")
    pmv, ppd = calculate_pmv_ppd(24.0, rh=50.0, met=1.2, clo=0.5)
    cost = discomfort_cost(pmv)
    print(f"T_air=24°C → PMV={pmv:.4f}, PPD={ppd:.2f}%, C_dis={cost:.6f}")
    if abs(pmv) > 0.35:
        raise AssertionError(f"expected near-neutral PMV at 24°C, got {pmv}")
    if cost > 1e-12:
        # |PMV| may be slightly > 0; penalty must be 0 if |PMV|<=0.5
        if abs(pmv) <= PMV_COMFORT_BAND + 1e-12 and cost > 1e-12:
            raise AssertionError("discomfort cost should be 0 inside band")

    hot = discomfort_cost_from_air(30.0, rh=50.0)
    print(f"T_air=30°C → PMV={hot['pmv']:.4f}, C_dis={hot['discomfort_cost']:.4f}")
    if hot["discomfort_cost"] <= 0.0:
        raise AssertionError("expected positive discomfort at 30°C")

    tou = TOUTariffSchedule.from_json()
    r15 = tou.rate_usd_per_kwh(15)
    r3 = tou.rate_usd_per_kwh(3)
    r10 = tou.rate_usd_per_kwh(10)
    print(f"rate@03={r3:.2f}  rate@10={r10:.2f}  rate@15={r15:.2f}")
    if abs(r15 - 0.48) > 1e-12:
        raise AssertionError(f"on-peak rate expected 0.48, got {r15}")
    if abs(r3 - 0.12) > 1e-12:
        raise AssertionError(f"off-peak rate expected 0.12, got {r3}")
    if abs(r10 - 0.22) > 1e-12:
        raise AssertionError(f"mid-peak rate expected 0.22, got {r10}")
    if abs(r15 / r3 - 4.0) > 1e-12:
        raise AssertionError("peak/off ratio should be 4×")

    # Spot bill: 1 kW for one peak hour → $0.48
    bill = tou.bill_from_power_W([1000.0], [15], dt_hours=1.0)
    assert abs(bill["cost_usd"] - 0.48) < 1e-12
    print("=== Phase 2 self-check PASSED ===")


if __name__ == "__main__":
    main()

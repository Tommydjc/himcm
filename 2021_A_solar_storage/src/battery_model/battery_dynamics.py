r"""Off-grid battery SOC state machine and cycle aging (SOH).

SOC updates (hour step \(\Delta t=1\,\mathrm{h}\))
    \[
    \mathrm{SOC}_{t+1}
    =
    \begin{cases}
      \mathrm{SOC}_t
      + \dfrac{P_{\mathrm{ch}}\,\eta_{\mathrm{ch}}}{E_{\mathrm{cap}}}\,\Delta t,
      & P_{\mathrm{ch}}>0,\\[0.6em]
      \mathrm{SOC}_t
      - \dfrac{P_{\mathrm{dis}}}{\eta_{\mathrm{dis}} E_{\mathrm{cap}}}\,\Delta t,
      & P_{\mathrm{dis}}>0,
    \end{cases}
    \]
clipped to \([\mathrm{SOC}_{\min},\mathrm{SOC}_{\max}]\)
(LFP default \(0.10\)–\(0.95\)).

Cycle aging (energy-throughput / EFC)
    \[
    \Delta\mathrm{SOH}(t)
    =
    \dfrac{|P_{\mathrm{bat}}(t)|\,\Delta t}
    {2\,E_{\mathrm{cap}}\,N_{\mathrm{cycle90}}},
    \]
where \(P_{\mathrm{bat}}\) is DC-side charge or discharge power and
\(N_{\mathrm{cycle90}}\) is the nameplate cycle count to ~90\% remaining
capacity. One-year fade is \(\sum_t\Delta\mathrm{SOH}\).

Round-trip split: \(\eta_{\mathrm{ch}}=\eta_{\mathrm{dis}}=\sqrt{\eta_{\mathrm{rt}}}\).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_BATTERY_CSV: Path = (
    PACK_ROOT / "data" / "parameters" / "commercial_batteries.csv"
)

# Pack baselines named in the modeling brief (used when CSV lacks power / aliases)
PACK_BASELINES: tuple[dict[str, object], ...] = (
    {
        "model_name": "Tesla_Powerwall_Plus",
        "aliases": ("tesla powerwall+", "tesla powerwall"),
        "capacity_kWh": 13.5,
        "continuous_power_kW": 5.0,
        "roundtrip_eff": 0.90,
        "cost_usd": 10500.0,
        "cycle_life": 4000,
        "chemistry": "LFP",
        "soc_min": 0.10,
        "soc_max": 0.95,
    },
    {
        "model_name": "Enphase_IQ_10T",
        "aliases": ("enphase iq", "enphase"),
        "capacity_kWh": 10.08,
        "continuous_power_kW": 3.84,
        "roundtrip_eff": 0.89,
        "cost_usd": 8200.0,
        "cycle_life": 4000,
        "chemistry": "LFP",
        "soc_min": 0.10,
        "soc_max": 0.95,
    },
    {
        "model_name": "FREEDOH_Module",
        "aliases": ("freedoh",),
        "capacity_kWh": 5.0,
        "continuous_power_kW": 2.5,
        "roundtrip_eff": 0.85,
        "cost_usd": 2200.0,
        "cycle_life": 2500,
        "chemistry": "LFP",
        "soc_min": 0.10,
        "soc_max": 0.95,
    },
    {
        "model_name": "Lead_Carbon_Array",
        "aliases": ("lead carbon", "lead-carbon", "deka", "trojan"),
        "capacity_kWh": 9.6,
        "continuous_power_kW": 3.0,
        "roundtrip_eff": 0.78,
        "cost_usd": 3100.0,
        "cycle_life": 1200,
        "chemistry": "Lead-Carbon",
        "soc_min": 0.30,
        "soc_max": 0.90,
    },
)


@dataclass(frozen=True, slots=True)
class CommercialBattery:
    r"""Normalized commercial pack contract.

    Attributes
    ----------
    model_name :
        Canonical identifier.
    capacity_kWh :
        Usable nameplate \(E_{\mathrm{cap}}\) (kWh) at SOH=1.
    continuous_power_kW :
        Inverter-limited charge/discharge power (kW).
    roundtrip_eff :
        AC–AC or DC–DC round-trip \(\eta_{\mathrm{rt}}\in(0,1]\).
    cost_usd :
        Capex (USD).
    cycle_life :
        \(N_{\mathrm{cycle90}}\) equivalent full cycles to ~10\% fade.
    chemistry :
        Chemistry label (LFP / Lead-Carbon / …).
    soc_min, soc_max :
        Health / warranty SOC window (dimensionless).
    """

    model_name: str
    capacity_kWh: float
    continuous_power_kW: float
    roundtrip_eff: float
    cost_usd: float
    cycle_life: int
    chemistry: str = "LFP"
    soc_min: float = 0.10
    soc_max: float = 0.95

    @property
    def eta_ch(self) -> float:
        r"""Charge efficiency \(\eta_{\mathrm{ch}}=\sqrt{\eta_{\mathrm{rt}}}\)."""

        return math.sqrt(max(float(self.roundtrip_eff), 1.0e-6))

    @property
    def eta_dis(self) -> float:
        r"""Discharge efficiency \(\eta_{\mathrm{dis}}=\sqrt{\eta_{\mathrm{rt}}}\)."""

        return math.sqrt(max(float(self.roundtrip_eff), 1.0e-6))

    @property
    def usable_window_kWh(self) -> float:
        """Energy between SOC bounds at SOH=1."""

        return self.capacity_kWh * (self.soc_max - self.soc_min)

    def with_units(self, n_units: int) -> "CommercialBattery":
        """Scale capacity / power / cost by parallel unit count."""

        n = max(int(n_units), 1)
        return replace(
            self,
            capacity_kWh=self.capacity_kWh * n,
            continuous_power_kW=self.continuous_power_kW * n,
            cost_usd=self.cost_usd * n,
            model_name=f"{self.model_name}_x{n}",
        )


@dataclass(frozen=True, slots=True)
class BatterySimResult:
    """Trajectory and annual aging summary."""

    soc: np.ndarray
    soh: np.ndarray
    p_ch_kw: np.ndarray
    p_dis_kw: np.ndarray
    energy_in_kWh: float
    energy_out_kWh: float
    energy_to_store_kWh: float
    energy_from_store_kWh: float
    efc_year: float
    soh_fade_year: float
    capacity_end_kWh: float

    @property
    def roundtrip_observed(self) -> float:
        r"""Measured AC \(E_{\mathrm{out}}/E_{\mathrm{in}}\) (0 if no charge)."""

        if self.energy_in_kWh <= 0.0:
            return 0.0
        return self.energy_out_kWh / self.energy_in_kWh


def _clean_name(raw: str) -> str:
    return re.sub(r"\s+", " ", str(raw).replace(";", "").strip())


def _infer_chemistry(name: str) -> str:
    low = name.lower()
    if "lead" in low or "deka" in low or "trojan" in low or "gcc" in low:
        return "Lead-Acid"
    if "lifepo" in low or "lfp" in low or "lithium" in low or "powerwall" in low:
        return "LFP"
    return "LFP"


def _infer_power_kw(capacity_kWh: float, name: str, chemistry: str) -> float:
    """Continuous power when CSV has no power column."""

    low = name.lower()
    if "powerwall" in low:
        return 5.0
    if "electriq" in low or "powerpod" in low:
        return min(5.0, 0.5 * capacity_kWh)
    if "discover" in low:
        return min(5.0, 0.5 * capacity_kWh)
    c_rate = 0.25 if chemistry.startswith("Lead") else 0.40
    return max(0.5, c_rate * capacity_kWh)


def _infer_cycle_life(life_span_years: float, chemistry: str, name: str) -> int:
    r"""Map warranty years / chemistry to \(N_{\mathrm{cycle90}}\)."""

    low = name.lower()
    if "powerwall" in low:
        return 4000
    if chemistry.startswith("Lead"):
        return max(800, int(round(life_span_years * 200.0)))
    return max(1500, int(round(life_span_years * 400.0)))


def _soc_window(chemistry: str) -> tuple[float, float]:
    if chemistry.startswith("Lead"):
        return 0.30, 0.90
    return 0.10, 0.95


def _match_baseline(name: str) -> dict[str, object] | None:
    low = name.lower()
    for row in PACK_BASELINES:
        aliases = row["aliases"]
        assert isinstance(aliases, tuple)
        if any(a in low for a in aliases):
            return row
    return None


def parse_batteries_csv(path: Path | None = None) -> list[CommercialBattery]:
    """Parse Team-11691-style or pack-normalized battery CSV."""

    if path is None:
        path = DEFAULT_BATTERY_CSV
    path = Path(path)
    raw = pd.read_csv(path)
    # normalize headers
    rename = {c: c.strip() for c in raw.columns}
    raw = raw.rename(columns=rename)
    cols = {c.lower(): c for c in raw.columns}

    def col(*cands: str) -> str | None:
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    name_c = col("Name", "model_name")
    cap_c = col("Usabe Capacity(kWh)", "Usable Capacity(kWh)", "capacity_kWh")
    cost_c = col("Cost($)", "cost_usd")
    eff_c = col("Round-Trip Efficiency (%)", "roundtrip_eff")
    life_c = col("Life span", "cycle_life")
    power_c = col("continuous_power_kW", "Power (kW)")
    chem_c = col("chemistry")
    if name_c is None or cap_c is None:
        raise ValueError(f"battery CSV missing Name/Capacity columns: {path}")

    packs: list[CommercialBattery] = []
    for _, row in raw.iterrows():
        name = _clean_name(row[name_c])
        if not name:
            continue
        capacity = float(row[cap_c])
        baseline = _match_baseline(name)
        chemistry = (
            str(row[chem_c])
            if chem_c is not None and pd.notna(row[chem_c])
            else (
                str(baseline["chemistry"])
                if baseline is not None
                else _infer_chemistry(name)
            )
        )
        if eff_c is not None and pd.notna(row[eff_c]):
            eff_raw = float(row[eff_c])
            # Team CSV uses percent; pack-normalized uses fraction
            roundtrip = eff_raw / 100.0 if eff_raw > 1.5 else eff_raw
        elif baseline is not None:
            roundtrip = float(baseline["roundtrip_eff"])
        else:
            roundtrip = 0.90 if chemistry == "LFP" else 0.78

        if cost_c is not None and pd.notna(row[cost_c]):
            cost = float(row[cost_c])
        elif baseline is not None:
            cost = float(baseline["cost_usd"])
        else:
            cost = 500.0 * capacity

        if life_c is not None and pd.notna(row[life_c]):
            life_val = float(row[life_c])
            # Team CSV "Life span" is years (~3–9); pack CSV uses cycle counts
            if life_val < 50:
                cycles = _infer_cycle_life(life_val, chemistry, name)
            else:
                cycles = int(round(life_val))
        elif baseline is not None:
            cycles = int(baseline["cycle_life"])
        else:
            cycles = _infer_cycle_life(7.0, chemistry, name)

        if power_c is not None and pd.notna(row[power_c]):
            power = float(row[power_c])
        elif baseline is not None and "powerwall" in name.lower():
            power = float(baseline["continuous_power_kW"])
        elif baseline is not None and "freedoh" in name.lower():
            # CSV usable kWh may be module-cell scale; keep CSV capacity, baseline power scaled
            power = _infer_power_kw(capacity, name, chemistry)
        else:
            power = _infer_power_kw(capacity, name, chemistry)

        # Prefer brief nameplate capacity for Powerwall+ when CSV matches
        if baseline is not None and "powerwall" in name.lower():
            capacity = float(baseline["capacity_kWh"])
            power = float(baseline["continuous_power_kW"])
            # Keep CSV cost if present (8500) unless user brief override desired —
            # brief says $10500; use max of CSV and brief for conservatism? Use CSV cost.
            roundtrip = float(baseline["roundtrip_eff"])
            cycles = int(baseline["cycle_life"])
            chemistry = str(baseline["chemistry"])

        soc_min, soc_max = _soc_window(chemistry)
        if baseline is not None:
            soc_min = float(baseline["soc_min"])
            soc_max = float(baseline["soc_max"])

        packs.append(
            CommercialBattery(
                model_name=name.replace(" ", "_"),
                capacity_kWh=capacity,
                continuous_power_kW=power,
                roundtrip_eff=roundtrip,
                cost_usd=cost,
                cycle_life=cycles,
                chemistry=chemistry,
                soc_min=soc_min,
                soc_max=soc_max,
            )
        )
    return packs


def load_commercial_batteries(
    path: Path | None = None,
    include_pack_baselines: bool = True,
) -> list[CommercialBattery]:
    """Load CSV packs and optionally append missing brief baselines (Enphase, Lead-Carbon)."""

    packs = parse_batteries_csv(path)
    if not include_pack_baselines:
        return packs
    have = " ".join(p.model_name.lower() for p in packs)
    for row in PACK_BASELINES:
        aliases = row["aliases"]
        assert isinstance(aliases, tuple)
        if any(a.replace(" ", "_") in have or a in have for a in aliases):
            continue
        # skip if any alias already substring-matched in loaded names
        if any(a in have for a in aliases):
            continue
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


def get_battery_by_name(
    name: str,
    packs: Sequence[CommercialBattery] | None = None,
) -> CommercialBattery:
    """Case-insensitive substring / token lookup (e.g. ``FREEDOH`` → FREEDOH_72V…)."""

    if packs is None:
        packs = load_commercial_batteries()
    key = name.lower().replace(" ", "_")
    key_toks = {t for t in re.split(r"[_\W]+", key) if len(t) >= 4}
    for p in packs:
        model = p.model_name.lower()
        if key in model or model in key:
            return p
        model_toks = {t for t in re.split(r"[_\W]+", model) if len(t) >= 4}
        if key_toks and key_toks.issubset(model_toks):
            return p
        if key_toks and (key_toks & model_toks) and len(key_toks & model_toks) >= 1:
            # single distinctive token (freedoh, powerwall, enphase)
            distinctive = key_toks & model_toks
            if distinctive & {"freedoh", "powerwall", "enphase", "electriq", "discover"}:
                return p
    raise KeyError(f"battery {name!r} not found among {[p.model_name for p in packs]}")


def step_soc(
    soc: float,
    p_ch_kw: float,
    p_dis_kw: float,
    battery: CommercialBattery,
    soh: float = 1.0,
    dt_h: float = 1.0,
) -> tuple[float, float, float, float]:
    r"""Advance one SOC / SOH step with power limits and SOC clamps.

    Returns
    -------
    soc_next, soh_next, p_ch_applied, p_dis_applied
        Applied powers may be reduced by inverter / SOC headroom.
    """

    if p_ch_kw < 0.0 or p_dis_kw < 0.0:
        raise ValueError("p_ch_kw and p_dis_kw must be non-negative")
    if p_ch_kw > 0.0 and p_dis_kw > 0.0:
        raise ValueError("cannot charge and discharge in the same step")

    e_cap = battery.capacity_kWh * max(float(soh), 1.0e-6)
    p_max = battery.continuous_power_kW
    soc_min = battery.soc_min
    soc_max = battery.soc_max
    eta_c = battery.eta_ch
    eta_d = battery.eta_dis

    p_ch = min(float(p_ch_kw), p_max)
    p_dis = min(float(p_dis_kw), p_max)

    if p_ch > 0.0:
        headroom = max(soc_max - soc, 0.0) * e_cap
        # energy into store = P_ch * eta_ch * dt
        e_store = p_ch * eta_c * dt_h
        if e_store > headroom:
            p_ch = headroom / max(eta_c * dt_h, 1.0e-12)
            e_store = p_ch * eta_c * dt_h
        soc_next = soc + e_store / e_cap
        p_bat = p_ch
    elif p_dis > 0.0:
        available = max(soc - soc_min, 0.0) * e_cap
        # energy leaving store = P_dis / eta_dis * dt
        e_store = p_dis / eta_d * dt_h
        if e_store > available:
            p_dis = available * eta_d / max(dt_h, 1.0e-12)
            e_store = p_dis / eta_d * dt_h
        soc_next = soc - e_store / e_cap
        p_bat = p_dis
    else:
        soc_next = soc
        p_bat = 0.0

    soc_next = float(np.clip(soc_next, soc_min, soc_max))
    d_soh = abs(p_bat) * dt_h / (2.0 * battery.capacity_kWh * max(battery.cycle_life, 1))
    soh_next = float(max(soh - d_soh, 0.0))
    return soc_next, soh_next, float(p_ch), float(p_dis)


def simulate_battery_year(
    p_ch_kw: np.ndarray | Sequence[float],
    p_dis_kw: np.ndarray | Sequence[float],
    battery: CommercialBattery,
    soc0: float | None = None,
    soh0: float = 1.0,
    dt_h: float = 1.0,
) -> BatterySimResult:
    """Roll SOC/SOH over an hourly charge/discharge schedule.

    Args:
        p_ch_kw, p_dis_kw:
            Non-negative hourly powers (kW), same length (typically 8760).
        battery:
            Pack contract.
        soc0:
            Initial SOC; default mid-window.
        soh0:
            Initial state of health.
        dt_h:
            Step length (h).

    Returns:
        ``BatterySimResult`` with trajectories and annual fade.
    """

    ch = np.asarray(p_ch_kw, dtype=np.float64)
    dis = np.asarray(p_dis_kw, dtype=np.float64)
    if ch.shape != dis.shape:
        raise ValueError("p_ch_kw and p_dis_kw must share shape")
    if np.any(ch < -1.0e-12) or np.any(dis < -1.0e-12):
        raise ValueError("powers must be non-negative")
    if np.any((ch > 1.0e-12) & (dis > 1.0e-12)):
        raise ValueError("charge and discharge overlap at some hour")

    n = ch.size
    if soc0 is None:
        soc0 = 0.5 * (battery.soc_min + battery.soc_max)
    soc = np.zeros(n + 1, dtype=np.float64)
    soh = np.zeros(n + 1, dtype=np.float64)
    soc[0] = float(np.clip(soc0, battery.soc_min, battery.soc_max))
    soh[0] = float(soh0)
    ch_app = np.zeros(n, dtype=np.float64)
    dis_app = np.zeros(n, dtype=np.float64)

    for t in range(n):
        soc[t + 1], soh[t + 1], ch_app[t], dis_app[t] = step_soc(
            soc=soc[t],
            p_ch_kw=float(ch[t]),
            p_dis_kw=float(dis[t]),
            battery=battery,
            soh=soh[t],
            dt_h=dt_h,
        )

    energy_in = float(ch_app.sum() * dt_h)
    energy_out = float(dis_app.sum() * dt_h)
    energy_to_store = float((ch_app * battery.eta_ch).sum() * dt_h)
    energy_from_store = float((dis_app / battery.eta_dis).sum() * dt_h)
    throughput = float((ch_app + dis_app).sum() * dt_h)
    efc = throughput / (2.0 * battery.capacity_kWh)
    fade = float(soh[0] - soh[-1])
    return BatterySimResult(
        soc=soc,
        soh=soh,
        p_ch_kw=ch_app,
        p_dis_kw=dis_app,
        energy_in_kWh=energy_in,
        energy_out_kWh=energy_out,
        energy_to_store_kWh=energy_to_store,
        energy_from_store_kWh=energy_from_store,
        efc_year=efc,
        soh_fade_year=fade,
        capacity_end_kWh=battery.capacity_kWh * float(soh[-1]),
    )



def annual_fade_factor(result: BatterySimResult) -> float:
    r"""Multiplicative capacity factor after one year: \(1-\Delta\mathrm{SOH}\)."""

    return max(1.0 - result.soh_fade_year, 0.0)


def batteries_to_frame(packs: Iterable[CommercialBattery]) -> pd.DataFrame:
    """Tabular view for optimization / LCOS."""

    rows = [
        {
            "model_name": p.model_name,
            "capacity_kWh": p.capacity_kWh,
            "continuous_power_kW": p.continuous_power_kW,
            "roundtrip_eff": p.roundtrip_eff,
            "eta_ch": p.eta_ch,
            "eta_dis": p.eta_dis,
            "cost_usd": p.cost_usd,
            "cycle_life": p.cycle_life,
            "chemistry": p.chemistry,
            "soc_min": p.soc_min,
            "soc_max": p.soc_max,
            "usable_window_kWh": p.usable_window_kWh,
        }
        for p in packs
    ]
    return pd.DataFrame(rows)

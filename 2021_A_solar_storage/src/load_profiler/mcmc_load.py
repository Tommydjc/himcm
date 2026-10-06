r"""MCMC / Gauss–Markov household load synthesizer (8760 h).

Three additive layers (kW):
    1. Rigid base — refrigerator duty cycle, comms, standby lighting.
    2. Seasonal heat pump — cooling \((T_{\mathrm{amb}}-T^\star)^+\) and
       heating \((T^\star-T_{\mathrm{amb}})^+\) with \(T^\star=20^\circ\mathrm{C}\).
    3. Behavioral pulses — cooking and wet appliances driven by a
       two-state Markov chain whose switch-on probability peaks at
       07–09 h and 18–22 h local time.

Catalog watts and duty cycles are read from
``data/raw/appliances_catalog.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG: Path = PACK_ROOT / "data" / "raw" / "appliances_catalog.csv"

RIGID_NAMES: tuple[str, ...] = (
    "Refrigerator_Freezer",
    "LED_Lighting_WholeHouse",
    "WiFi_Router_Comm",
    "Well_Water_Pump",
    "Home_Office_Laptop",
)
BEHAVIORAL_NAMES: tuple[str, ...] = (
    "Microwave_Oven",
    "Induction_Cooktop",
    "Washing_Machine",
    "Electric_Clothes_Dryer",
    "Dishwasher",
    "Electric_Water_Heater",
)
HVAC_NAME: str = "Heat_Pump_HVAC"


@dataclass(frozen=True, slots=True)
class LoadSynthParams:
    r"""Load-layer knobs.

    Attributes
    ----------
    t_star_c :
        Balance temperature \(T^\star\) (\(^\circ\mathrm{C}\)).
    hvac_kw_per_deg :
        Heat-pump electrical intensity (kW / \(^\circ\mathrm{C}\)) times
        catalog rated power scale.
    rigid_target_kw :
        Target mean of the rigid layer (kW); a small phantom offset is
        added so the annual rigid mean sits near \(0.35\,\mathrm{kW}\).
    seed :
        RNG seed for the Markov / Gauss pulses.
    """

    t_star_c: float = 20.0
    hvac_kw_per_deg: float = 0.045
    rigid_target_kw: float = 0.35
    seed: int = 2021


def _hour_in_peaks(hour: np.ndarray, peak_spec: str) -> np.ndarray:
    """Return a boolean mask for catalog ``peak_hours`` strings like ``7-9_18-20``."""

    mask = np.zeros(hour.shape, dtype=bool)
    for chunk in str(peak_spec).split("_"):
        if "-" not in chunk:
            continue
        lo_s, hi_s = chunk.split("-", 1)
        lo = int(lo_s)
        hi = int(hi_s)
        mask |= (hour >= lo) & (hour <= hi)
    return mask


def _load_catalog(path: Path) -> pd.DataFrame:
    """Read the appliance dictionary and index by name."""

    catalog = pd.read_csv(path)
    return catalog.set_index("appliance_name")


def _rigid_layer(
    hour: np.ndarray,
    catalog: pd.DataFrame,
    rng: np.random.Generator,
    target_mean_kw: float,
) -> np.ndarray:
    """Always-on / cyclic appliances. Fridge is a 0/1 duty oscillator plus noise."""

    n = hour.size
    load = np.zeros(n, dtype=np.float64)
    fridge = catalog.loc["Refrigerator_Freezer"]
    duty = float(fridge["duty_cycle"])
    # 40 min on / 60 min period-style hourly occupancy via Bernoulli at duty
    fridge_on = rng.random(n) < duty
    load += fridge_on * (float(fridge["rated_power_W"]) / 1000.0)

    router = catalog.loc["WiFi_Router_Comm"]
    load += float(router["rated_power_W"]) / 1000.0

    lights = catalog.loc["LED_Lighting_WholeHouse"]
    light_w = float(lights["rated_power_W"]) / 1000.0
    evening = (hour >= 18) & (hour <= 23)
    # Standby nightlights + evening occupancy
    load += np.where(evening, light_w, 0.08 * light_w)

    pump = catalog.loc["Well_Water_Pump"]
    pump_mask = _hour_in_peaks(hour, str(pump["peak_hours"]))
    pump_on = pump_mask & (rng.random(n) < 0.22)
    load += pump_on * (float(pump["rated_power_W"]) / 1000.0)

    laptop = catalog.loc["Home_Office_Laptop"]
    office = (hour >= 9) & (hour <= 17)
    laptop_on = office & (rng.random(n) < float(laptop["duty_cycle"]))
    load += laptop_on * (float(laptop["rated_power_W"]) / 1000.0)

    offset = target_mean_kw - float(load.mean())
    if offset > 0.0:
        load += offset
    return load


def _hvac_layer(
    t_amb_c: np.ndarray,
    catalog: pd.DataFrame,
    params: LoadSynthParams,
) -> np.ndarray:
    r"""Heat-pump kW \(\propto (T-T^\star)^+\) cooling and \((T^\star-T)^+\) heating."""

    rated = float(catalog.loc[HVAC_NAME]["rated_power_W"]) / 1000.0
    cool = np.maximum(t_amb_c - params.t_star_c, 0.0)
    heat = np.maximum(params.t_star_c - t_amb_c, 0.0)
    # Duty 0.5 from catalog scales the linear coupling; clip to nameplate
    duty = float(catalog.loc[HVAC_NAME]["duty_cycle"])
    raw = params.hvac_kw_per_deg * (cool + 0.85 * heat) * (rated / 2.2) * (duty / 0.5)
    return np.minimum(raw, rated)


def _gauss_markov_appliance(
    hour: np.ndarray,
    rated_kw: float,
    peak_spec: str,
    rng: np.random.Generator,
    p_on_peak: float,
    p_on_offpeak: float,
    persist: float,
) -> np.ndarray:
    r"""Two-state Markov occupancy with Gaussian amplitude around nameplate.

    \[
    \mathbb{P}(s_t=1\mid s_{t-1}=0)=p_{\mathrm{on}}(h_t),\quad
    \mathbb{P}(s_t=1\mid s_{t-1}=1)=\rho+(1-\rho)p_{\mathrm{on}}(h_t).
    \]
    """

    n = hour.size
    in_peak = _hour_in_peaks(hour, peak_spec)
    p_on = np.where(in_peak, p_on_peak, p_on_offpeak)
    state = np.zeros(n, dtype=np.int8)
    u = rng.random(n)
    for t in range(n):
        if t == 0:
            state[t] = 1 if u[t] < p_on[t] else 0
        elif state[t - 1] == 1:
            p_stay = persist + (1.0 - persist) * p_on[t]
            state[t] = 1 if u[t] < p_stay else 0
        else:
            state[t] = 1 if u[t] < p_on[t] else 0
    amp = rng.normal(loc=1.0, scale=0.12, size=n)
    amp = np.clip(amp, 0.55, 1.35)
    return state.astype(np.float64) * rated_kw * amp


def _behavioral_layer(
    hour: np.ndarray,
    catalog: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cooking / wet-appliance Markov pulses (morning and evening peaks)."""

    n = hour.size
    load = np.zeros(n, dtype=np.float64)
    specs: dict[str, tuple[float, float, float]] = {
        "Microwave_Oven": (0.28, 0.02, 0.45),
        "Induction_Cooktop": (0.35, 0.03, 0.55),
        "Washing_Machine": (0.12, 0.015, 0.70),
        "Electric_Clothes_Dryer": (0.10, 0.01, 0.65),
        "Dishwasher": (0.18, 0.02, 0.60),
        "Electric_Water_Heater": (0.22, 0.04, 0.50),
    }
    for name, (p_peak, p_off, persist) in specs.items():
        row = catalog.loc[name]
        rated = float(row["rated_power_W"]) / 1000.0
        load += _gauss_markov_appliance(
            hour=hour,
            rated_kw=rated,
            peak_spec=str(row["peak_hours"]),
            rng=rng,
            p_on_peak=p_peak,
            p_on_offpeak=p_off,
            persist=persist,
        )
    return load


def synthesize_load_kw(
    hour_local: np.ndarray,
    t_amb_c: np.ndarray,
    catalog_path: Path | None = None,
    params: LoadSynthParams | None = None,
) -> pd.DataFrame:
    r"""Assemble rigid + HVAC + behavioral kW for each hour.

    Args:
        hour_local:
            Local hour of day, shape ``(8760,)``, values in \(\{0,\ldots,23\}\).
        t_amb_c:
            Ambient temperature (\(^\circ\mathrm{C}\)), same length.
        catalog_path:
            Appliance dictionary; default pack CSV.
        params:
            Synthesis knobs.

    Returns:
        DataFrame with layer columns and ``load_demand_kW``.
    """

    if params is None:
        params = LoadSynthParams()
    if catalog_path is None:
        catalog_path = DEFAULT_CATALOG
    hour = np.asarray(hour_local, dtype=np.int64)
    t_amb = np.asarray(t_amb_c, dtype=np.float64)
    if hour.shape != t_amb.shape:
        raise ValueError("hour_local and t_amb_c must share shape")
    catalog = _load_catalog(Path(catalog_path))
    rng = np.random.default_rng(params.seed)
    rigid = _rigid_layer(hour, catalog, rng, params.rigid_target_kw)
    hvac = _hvac_layer(t_amb, catalog, params)
    behav = _behavioral_layer(hour, catalog, rng)
    total = np.maximum(rigid + hvac + behav, 0.0)
    return pd.DataFrame(
        {
            "load_rigid_kW": rigid,
            "load_hvac_kW": hvac,
            "load_behavior_kW": behav,
            "load_demand_kW": total,
        }
    )

r"""Hay–Davies tilted irradiance and temperature-corrected PV power.

POA (plane-of-array) irradiance
    \[
    G_{\mathrm{tilt}}
    = G_b R_b
      + G_d\bigl[A_i R_b + (1-A_i)(1+\cos\beta)/2\bigr]
      + G\rho(1-\cos\beta)/2,
    \]
with anisotropy index \(A_i=G_b/G_0\) and geometric factor
\(R_b=\cos\theta/\cos\theta_z\).

DC array power (W then converted to kW)
    \[
    P_{\mathrm{PV}}(t)
    = A_{\mathrm{array}}\,\eta_{\mathrm{STC}}\,G_{\mathrm{tilt}}(t)
      \bigl[1-\gamma_{\mathrm{temp}}(T_{\mathrm{cell}}(t)-25)\bigr],
    \]
    \[
    T_{\mathrm{cell}}
    = T_{\mathrm{amb}}+\frac{\mathrm{NOCT}-20}{800}\,G_{\mathrm{tilt}}.
    \]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_IRRADIANCE: Path = PACK_ROOT / "data" / "raw" / "solar_irradiance_annual.csv"

# Phoenix / Sonoran site used for the Open-Meteo archive pull
SITE_LAT_DEG: float = 33.45
SITE_LON_DEG: float = -112.07
I_SC_W_M2: float = 1367.0
GROUND_ALBEDO: float = 0.20
EPS: float = 1.0e-9


@dataclass(frozen=True, slots=True)
class PVArrayParams:
    r"""STC array and thermal coefficients.

    Attributes
    ----------
    area_m2 :
        Collector area \(A_{\mathrm{array}}\) (m\(^2\)).
    eta_stc :
        Module efficiency at STC (dimensionless).
    gamma_temp :
        Power temperature coefficient (\(1/^\circ\mathrm{C}\)), default \(0.004\).
    noct_c :
        Nominal operating cell temperature (\(^\circ\mathrm{C}\)).
    tilt_deg :
        Surface slope \(\beta\); default \(35^\circ\) (near-latitude optimum).
    azimuth_deg :
        Surface azimuth \(\gamma\) in degrees, \(0=\) due south (NH).
    latitude_deg, longitude_deg :
        Site coordinates (deg).
    """

    area_m2: float = 30.0
    eta_stc: float = 0.20
    gamma_temp: float = 0.004
    noct_c: float = 45.0
    tilt_deg: float = 35.0
    azimuth_deg: float = 0.0
    latitude_deg: float = SITE_LAT_DEG
    longitude_deg: float = SITE_LON_DEG

    @property
    def p_stc_kw(self) -> float:
        r"""Nameplate at \(G=1000\,\mathrm{W/m}^2\), \(25^\circ\mathrm{C}\)."""

        return self.area_m2 * self.eta_stc * 1.0


def _solar_geometry(
    utc: pd.DatetimeIndex,
    latitude_deg: float,
    longitude_deg: float,
    tilt_deg: float,
    azimuth_deg: float,
) -> Mapping[str, np.ndarray]:
    r"""Vectorized solar zenith, incidence cosine, and extraterrestrial \(G_0\).

    Uses Spencer/NOAA noon-centered declination and equation of time.
    Timestamps are treated as UTC (Open-Meteo archive default).
    """

    n = utc.dayofyear.to_numpy(dtype=np.float64)
    hour_utc = (
        utc.hour.to_numpy(dtype=np.float64)
        + utc.minute.to_numpy(dtype=np.float64) / 60.0
    )
    gamma = 2.0 * np.pi * (n - 1.0) / 365.0
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )
    eot_min = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    lst = hour_utc + longitude_deg / 15.0 + eot_min / 60.0
    hour_angle = np.deg2rad(15.0 * (lst - 12.0))
    lat = np.deg2rad(latitude_deg)
    beta = np.deg2rad(tilt_deg)
    az = np.deg2rad(azimuth_deg)

    cos_zen = np.sin(decl) * np.sin(lat) + np.cos(decl) * np.cos(lat) * np.cos(
        hour_angle
    )
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    # Incidence on a south-referenced tilted plane (Duffie & Beckman)
    cos_inc = (
        np.sin(decl) * np.sin(lat) * np.cos(beta)
        - np.sin(decl) * np.cos(lat) * np.sin(beta) * np.cos(az)
        + np.cos(decl) * np.cos(lat) * np.cos(beta) * np.cos(hour_angle)
        + np.cos(decl) * np.sin(lat) * np.sin(beta) * np.cos(az) * np.cos(hour_angle)
        + np.cos(decl) * np.sin(beta) * np.sin(az) * np.sin(hour_angle)
    )
    cos_inc = np.clip(cos_inc, -1.0, 1.0)
    e0 = 1.0 + 0.033 * np.cos(2.0 * np.pi * n / 365.0)
    g0 = I_SC_W_M2 * e0 * np.maximum(cos_zen, 0.0)
    return {
        "cos_zenith": cos_zen,
        "cos_incidence": cos_inc,
        "g0_wm2": g0,
        "dayofyear": n,
    }


def hay_davies_g_tilt(
    ghi_wm2: np.ndarray,
    dhi_wm2: np.ndarray,
    dni_wm2: np.ndarray,
    cos_zenith: np.ndarray,
    cos_incidence: np.ndarray,
    g0_wm2: np.ndarray,
    tilt_deg: float,
    albedo: float = GROUND_ALBEDO,
) -> np.ndarray:
    r"""Hay–Davies POA irradiance \(G_{\mathrm{tilt}}\) (W/m\(^2\))."""

    beta = np.deg2rad(tilt_deg)
    beam_horiz = np.maximum(dni_wm2 * np.maximum(cos_zenith, 0.0), 0.0)
    # If DNI is missing/inconsistent, fall back to GHI − DHI
    residual = np.maximum(ghi_wm2 - dhi_wm2, 0.0)
    g_b = np.where(beam_horiz > 1.0, beam_horiz, residual)
    g_d = np.maximum(dhi_wm2, 0.0)
    g_g = np.maximum(ghi_wm2, 0.0)

    sun_up = (cos_zenith > 0.0) & (cos_incidence > 0.0)
    r_b = np.where(sun_up, cos_incidence / np.maximum(cos_zenith, EPS), 0.0)
    r_b = np.clip(r_b, 0.0, 6.0)
    a_i = np.clip(g_b / np.maximum(g0_wm2, EPS), 0.0, 1.0)
    sky_iso = (1.0 + np.cos(beta)) / 2.0
    ground = albedo * (1.0 - np.cos(beta)) / 2.0
    g_tilt = g_b * r_b + g_d * (a_i * r_b + (1.0 - a_i) * sky_iso) + g_g * ground
    return np.maximum(g_tilt, 0.0)


def cell_temperature_c(
    t_amb_c: np.ndarray,
    g_tilt_wm2: np.ndarray,
    noct_c: float,
) -> np.ndarray:
    r"""\(T_{\mathrm{cell}}=T_{\mathrm{amb}}+(\mathrm{NOCT}-20)G_{\mathrm{tilt}}/800\)."""

    return t_amb_c + (noct_c - 20.0) / 800.0 * g_tilt_wm2


def pv_power_kw(
    g_tilt_wm2: np.ndarray,
    t_cell_c: np.ndarray,
    params: PVArrayParams,
) -> np.ndarray:
    r"""Temperature-corrected DC power in kW (clipped at 0)."""

    thermal = 1.0 - params.gamma_temp * (t_cell_c - 25.0)
    thermal = np.maximum(thermal, 0.05)
    watts = params.area_m2 * params.eta_stc * g_tilt_wm2 * thermal
    return np.maximum(watts / 1000.0, 0.0)


def simulate_pv_timeseries(
    irradiance: pd.DataFrame | Path | None = None,
    params: PVArrayParams | None = None,
) -> pd.DataFrame:
    r"""Compute 8760-hour tilted irradiance and PV kW from the archive CSV.

    Args:
        irradiance:
            Table with ``time``, ``temperature_c``, ``dni_W_m2``,
            ``dhi_W_m2``, ``ghi_W_m2``, or a path. Default pack CSV.
        params:
            Array contract; default \(30\,\mathrm{m}^2\), \(\eta=0.20\),
            \(\beta=35^\circ\).

    Returns:
        Frame with UTC/local clocks, \(G_{\mathrm{tilt}}\), \(T_{\mathrm{cell}}\),
        and ``pv_output_kW``.
    """

    if params is None:
        params = PVArrayParams()
    if irradiance is None:
        irradiance = DEFAULT_IRRADIANCE
    if isinstance(irradiance, Path) or isinstance(irradiance, str):
        frame = pd.read_csv(irradiance)
    else:
        frame = irradiance.copy()
    if "time" not in frame.columns:
        raise ValueError("irradiance table must contain a 'time' column")
    utc = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True))
    if len(utc) != 8760:
        raise ValueError(f"expected 8760 hourly rows, got {len(utc)}")

    geo = _solar_geometry(
        utc,
        latitude_deg=params.latitude_deg,
        longitude_deg=params.longitude_deg,
        tilt_deg=params.tilt_deg,
        azimuth_deg=params.azimuth_deg,
    )
    g_tilt = hay_davies_g_tilt(
        ghi_wm2=frame["ghi_W_m2"].to_numpy(dtype=np.float64),
        dhi_wm2=frame["dhi_W_m2"].to_numpy(dtype=np.float64),
        dni_wm2=frame["dni_W_m2"].to_numpy(dtype=np.float64),
        cos_zenith=geo["cos_zenith"],
        cos_incidence=geo["cos_incidence"],
        g0_wm2=geo["g0_wm2"],
        tilt_deg=params.tilt_deg,
    )
    t_amb = frame["temperature_c"].to_numpy(dtype=np.float64)
    t_cell = cell_temperature_c(t_amb, g_tilt, params.noct_c)
    p_kw = pv_power_kw(g_tilt, t_cell, params)
    local = utc.tz_convert("America/Phoenix")
    out = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(utc.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")),
            "time_local": pd.to_datetime(local.strftime("%Y-%m-%d %H:%M:%S")),
            "hour": local.hour.astype(int),
            "month": local.month.astype(int),
            "temperature_c": t_amb,
            "ghi_W_m2": frame["ghi_W_m2"].to_numpy(dtype=np.float64),
            "g_tilt_W_m2": g_tilt,
            "t_cell_c": t_cell,
            "pv_output_kW": p_kw,
        }
    )
    return out

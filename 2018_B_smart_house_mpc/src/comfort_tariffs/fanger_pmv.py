"""
ISO 7730 Fanger PMV–PPD continuous comfort model (ASHRAE 55 aligned).

Implements the standard iterative clothing surface temperature solve.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _clo_to_m2k_per_w(clo: float) -> float:
    return 0.155 * clo


def fanger_pmv(
    T_air_C: float,
    RH_pct: float,
    *,
    T_mrt_C: float | None = None,
    vel_m_s: float = 0.1,
    met: float = 1.1,
    clo: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute PMV and PPD.

    Parameters
    ----------
    T_air_C :
        Air temperature [°C].
    RH_pct :
        Relative humidity [%].
    T_mrt_C :
        Mean radiant temperature [°C]; defaults to air temperature.
    vel_m_s :
        Relative air velocity [m/s].
    met :
        Metabolic rate [met].
    clo :
        Clothing insulation [clo].

    Returns
    -------
    (pmv, ppd)
        Predicted Mean Vote and Predicted Percentage of Dissatisfied [%].
    """
    if T_mrt_C is None:
        T_mrt_C = T_air_C
    pa = RH_pct * 10.0 * np.exp(16.6536 - 4030.183 / (T_air_C + 235.0))
    icl = _clo_to_m2k_per_w(clo)
    m = met * 58.15
    w = 0.0
    mw = m - w
    if icl <= 0.078:
        fcl = 1.0 + 1.29 * icl
    else:
        fcl = 1.05 + 0.645 * icl

    hcf = 12.1 * np.sqrt(vel_m_s)
    taa = T_air_C + 273.0
    tra = T_mrt_C + 273.0
    tcla = taa + (35.5 - T_air_C) / (3.5 * icl + 0.1)

    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * ((tra / 100.0) ** 4)
    xn = tcla / 100.0
    xf = tcla / 50.0
    eps = 1e-6
    n_iter = 0
    while abs(xn - xf) > eps and n_iter < 150:
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = max(hcf, hcn)
        xn = (p5 + p4 * hc - p2 * xf**4) / (100.0 + p3 * hc)
        n_iter += 1
    tcl = 100.0 * xn - 273.0

    hl1 = 3.05 * 0.001 * (5733.0 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0
    hl3 = 1.7e-5 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - T_air_C)
    hl5 = 3.96 * fcl * ((xn**4) - (tra / 100.0) ** 4)
    hl6 = fcl * hc * (tcl - T_air_C)
    ts = 0.303 * np.exp(-0.036 * m) + 0.028
    pmv = float(ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6))
    ppd = float(100.0 - 95.0 * np.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2))
    return pmv, ppd


def discomfort_degree_hours(
    T_series_C: np.ndarray,
    T_set_C: float,
    occupied: np.ndarray,
) -> float:
    """
    Occupied |T - T_set| integrated over hours (degree-hours).

    Parameters
    ----------
    T_series_C, occupied :
        Same length arrays; occupied in {0,1} or [0,1].
    """
    T_series_C = np.asarray(T_series_C, dtype=float)
    occupied = np.asarray(occupied, dtype=float)
    return float(np.sum(occupied * np.abs(T_series_C - T_set_C)))

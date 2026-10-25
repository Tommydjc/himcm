"""
Single-zone 2R2C thermal + moisture + pollutant ODEs.

State x = [T_in, T_env, w_in, C_pol] (°C, °C, g/kg, µg/m³).
Control u = [P_HVAC, m_hum, Q_vent, u_pur] (W, g/s, m³/s, -).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

PACK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPECS = PACK_ROOT / "data" / "parameters" / "building_specs.json"


def load_building_specs(path: Path = DEFAULT_SPECS) -> Dict[str, Any]:
    """Load building RC / HVAC / IAQ parameters from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def saturation_vapor_pressure_kpa(T_C: float) -> float:
    """Antoine / Magnus form: P_sat [kPa] from dry-bulb °C."""
    return float(0.61078 * np.exp(17.27 * T_C / (T_C + 237.3)))


def relative_humidity_pct(T_C: float, w_g_per_kg: float, P_atm_kpa: float = 101.325) -> float:
    """
    Convert absolute humidity to RH%.

    RH = 100 * P_atm * w / ((621.98 + w) * P_sat(T)).
    """
    P_sat = saturation_vapor_pressure_kpa(T_C)
    denom = (621.98 + w_g_per_kg) * P_sat
    if denom <= 1e-12:
        return 0.0
    rh = 100.0 * P_atm_kpa * w_g_per_kg / denom
    return float(np.clip(rh, 0.0, 100.0))


def bio_generation_ug_per_s(
    T_in: float,
    RH: float,
    *,
    G0: float,
    RH_crit: float,
    T_opt: float,
    sigma_T: float,
) -> float:
    """
    Indoor dust-mite / mold spore generation G_bio(T, RH).

    G_bio = G0 * σ(RH - RH_crit) * exp(-(T - T_opt)² / (2 σ_T²)).
    """
    # Smooth sigmoid activation above RH_crit.
    sigma = 1.0 / (1.0 + np.exp(-0.35 * (RH - RH_crit)))
    thermal = np.exp(-((T_in - T_opt) ** 2) / (2.0 * sigma_T**2))
    return float(G0 * sigma * thermal)


def cop_from_outdoor(T_out: float, *, mode: str, specs: Mapping[str, Any]) -> float:
    """
    Outdoor-temperature dependent COP (Carnot-inspired derating).

    Heating COP falls in cold weather; cooling COP falls in hot weather.
    """
    if mode == "heat":
        base = float(specs["COP_heat_ref"])
        return float(max(1.5, base * (1.0 - 0.015 * max(0.0, 5.0 - T_out))))
    base = float(specs["COP_cool_ref"])
    return float(max(1.5, base * (1.0 - 0.02 * max(0.0, T_out - 28.0))))


@dataclass
class Disturbance:
    """Exogenous inputs at one time step."""

    T_out: float
    w_out: float
    C_out: float
    I_solar: float
    n_occ: float


@dataclass
class ControlInput:
    """HVAC / humidification / ventilation / purifier command."""

    P_HVAC: float
    m_hum: float
    Q_vent: float
    u_pur: float

    def clip(self, specs: Mapping[str, Any]) -> "ControlInput":
        """Project onto actuator bounds."""
        return ControlInput(
            P_HVAC=float(
                np.clip(
                    self.P_HVAC,
                    -float(specs["P_cool_max_W"]),
                    float(specs["P_heat_max_W"]),
                )
            ),
            m_hum=float(
                np.clip(
                    self.m_hum,
                    -float(specs["m_dehum_max_g_per_s"]),
                    float(specs["m_hum_max_g_per_s"]),
                )
            ),
            Q_vent=float(
                np.clip(
                    self.Q_vent,
                    float(specs["Q_vent_min_m3_per_s"]),
                    float(specs["Q_vent_max_m3_per_s"]),
                )
            ),
            u_pur=float(np.clip(self.u_pur, 0.0, 1.0)),
        )


def electric_power_W(
    u: ControlInput,
    T_out: float,
    specs: Mapping[str, Any],
) -> float:
    """Map thermal/actuator commands to site electric power [W]."""
    if u.P_HVAC >= 0.0:
        cop = cop_from_outdoor(T_out, mode="heat", specs=specs)
        p_hvac = abs(u.P_HVAC) / max(cop, 1e-6)
    else:
        cop = cop_from_outdoor(T_out, mode="cool", specs=specs)
        p_hvac = abs(u.P_HVAC) / max(cop, 1e-6)
    p_hum = abs(u.m_hum) * float(specs["P_hum_W_per_gps"])
    p_fan = float(specs["P_fan_coeff_W_s2_per_m6"]) * (u.Q_vent**2)
    p_pur = float(specs["P_pur_max_W"]) * u.u_pur
    return float(p_hvac + p_hum + p_fan + p_pur)


def ode_rhs(
    x: NDArray[np.floating],
    u: ControlInput,
    d: Disturbance,
    specs: Mapping[str, Any],
) -> NDArray[np.float64]:
    """
    Continuous-time RHS f(x, u, d) for the 4-state 3-field model.

    Returns
    -------
    dx/dt : shape (4,)
        [dT_in/dt, dT_env/dt, dw_in/dt, dC_pol/dt].
    """
    T_in, T_env, w_in, C_pol = map(float, x)
    u = u.clip(specs)
    C_air = float(specs["C_air_J_per_K"])
    C_env = float(specs["C_env_J_per_K"])
    R_in = float(specs["R_in_K_per_W"])
    R_win = float(specs["R_win_K_per_W"])
    R_wall = float(specs["R_wall_K_per_W"])
    rho = float(specs["rho_air_kg_per_m3"])
    c_p = float(specs["c_p_J_per_kgK"])
    V = float(specs["volume_m3"])

    vent_heat = rho * c_p * u.Q_vent * (d.T_out - T_in)
    occ_heat = float(specs["q_occ_W"]) * d.n_occ
    solar_in = float(specs["alpha_in"]) * float(specs["window_area_m2"]) * d.I_solar
    dT_in = (
        (T_env - T_in) / R_in
        + (d.T_out - T_in) / R_win
        + vent_heat
        + u.P_HVAC
        + occ_heat
        + solar_in
    ) / C_air

    # Opaque-wall solar uses an effective projected area (not full facade).
    A_wall_sol = float(specs.get("wall_solar_area_m2", 0.15 * float(specs["wall_area_m2"])))
    solar_env = float(specs["alpha_env"]) * A_wall_sol * d.I_solar
    dT_env = ((T_in - T_env) / R_in + (d.T_out - T_env) / R_wall + solar_env) / C_env

    occ_moist = float(specs["g_occ_g_per_s"]) * d.n_occ
    dw = (
        rho * u.Q_vent * (d.w_out - w_in) + u.m_hum + occ_moist
    ) / (rho * V)

    RH = relative_humidity_pct(T_in, w_in)
    G_bio = bio_generation_ug_per_s(
        T_in,
        RH,
        G0=float(specs["G0_ug_per_s"]),
        RH_crit=float(specs["RH_crit_pct"]),
        T_opt=float(specs["T_opt_C"]),
        sigma_T=float(specs["sigma_T_C"]),
    )
    Q_inf = float(specs["Q_inf_m3_per_s"])
    eta = float(specs["eta_filt"])
    p_pen = float(specs["p_pen"])
    beta = float(specs["beta_dep_1_per_s"])
    CADR = float(specs["CADR_max_m3_per_s"]) * u.u_pur
    dC = (
        u.Q_vent * ((1.0 - eta) * d.C_out - C_pol)
        + Q_inf * (p_pen * d.C_out - C_pol)
        - (beta * V + CADR) * C_pol
        + G_bio
    ) / V

    return np.asarray([dT_in, dT_env, dw, dC], dtype=np.float64)


def rk4_step(
    x: NDArray[np.floating],
    u: ControlInput,
    d: Disturbance,
    specs: Mapping[str, Any],
    dt_s: float,
) -> NDArray[np.float64]:
    """One RK4 step of length ``dt_s`` seconds."""
    k1 = ode_rhs(x, u, d, specs)
    k2 = ode_rhs(x + 0.5 * dt_s * k1, u, d, specs)
    k3 = ode_rhs(x + 0.5 * dt_s * k2, u, d, specs)
    k4 = ode_rhs(x + dt_s * k3, u, d, specs)
    x_next = x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Physical clamps.
    x_next[2] = max(0.0, float(x_next[2]))
    x_next[3] = max(0.0, float(x_next[3]))
    return np.asarray(x_next, dtype=np.float64)


def simulate_horizon(
    x0: Sequence[float],
    controls: Sequence[ControlInput],
    disturbances: Sequence[Disturbance],
    specs: Mapping[str, Any],
    *,
    dt_s: float = 3600.0,
    substeps: int = 12,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Simulate N control intervals.

    Parameters
    ----------
    x0 :
        Initial state length 4.
    controls, disturbances :
        Length-N sequences held piecewise constant on each hour.
    dt_s :
        Interval length in seconds (default 1 h).
    substeps :
        RK4 sub-steps per interval.

    Returns
    -------
    X : (N+1, 4)
    P_elec : (N,)
    RH : (N+1,)
    """
    if len(controls) != len(disturbances):
        raise ValueError("controls and disturbances length mismatch")
    n = len(controls)
    X = np.zeros((n + 1, 4), dtype=np.float64)
    X[0] = np.asarray(x0, dtype=np.float64)
    P_elec = np.zeros(n, dtype=np.float64)
    h = dt_s / float(substeps)
    for k in range(n):
        x = X[k].copy()
        u = controls[k].clip(specs)
        d = disturbances[k]
        P_elec[k] = electric_power_W(u, d.T_out, specs)
        for _ in range(substeps):
            x = rk4_step(x, u, d, specs, h)
        X[k + 1] = x
    RH = np.asarray(
        [relative_humidity_pct(float(row[0]), float(row[2])) for row in X],
        dtype=np.float64,
    )
    return X, P_elec, RH


def free_float_check(
    specs: Optional[Mapping[str, Any]] = None,
    *,
    hours: int = 24,
    T_out: float = 32.0,
    RH_out: float = 70.0,
) -> Dict[str, float]:
    """
    Uncontrolled hot/humid free-float diagnostic.

    Returns peak RH and peak C_pol for self-consistency checks.
    """
    specs = dict(specs or load_building_specs())
    Psat = saturation_vapor_pressure_kpa(T_out)
    w_out = 621.98 * (RH_out / 100.0) * Psat / (101.325 - (RH_out / 100.0) * Psat)
    x = np.asarray([26.0, 27.0, 12.0, 20.0], dtype=np.float64)
    u = ControlInput(0.0, 0.0, float(specs["Q_vent_min_m3_per_s"]), 0.0)
    # Mild solar (not perpetual noon) for a stable free-float diagnostic.
    d = Disturbance(T_out, w_out, 40.0, 180.0, 0.0)
    peak_rh = relative_humidity_pct(float(x[0]), float(x[2]))
    peak_c = float(x[3])
    for _ in range(hours):
        for __ in range(12):
            x = rk4_step(x, u, d, specs, 300.0)
        rh = relative_humidity_pct(float(x[0]), float(x[2]))
        peak_rh = max(peak_rh, rh)
        peak_c = max(peak_c, float(x[3]))
    return {
        "final_T_in": float(x[0]),
        "peak_RH_pct": float(peak_rh),
        "peak_C_pol": float(peak_c),
        "final_C_pol": float(x[3]),
    }

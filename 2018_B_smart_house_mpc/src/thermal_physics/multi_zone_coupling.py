"""
Multi-zone thermal graph with interior-wall coupling and VAV dampers.

Task 2: rooms as nodes; edges carry conductance G_ij = 1/R_ij and door airflow.
Damper shares u_i satisfy sum_i u_i = 1 and allocate central HVAC power.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from thermal_physics.single_zone_rc import (
    ControlInput,
    Disturbance,
    load_building_specs,
    relative_humidity_pct,
    rk4_step,
)


def zone_adjacency_matrix(specs: Mapping[str, Any]) -> NDArray[np.float64]:
    """Return symmetric adjacency (0/1) from building_specs.multi_zone."""
    mz = specs["multi_zone"]
    A = np.asarray(mz["adjacency"], dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square")
    return 0.5 * (A + A.T)


def graph_laplacian_conductance(specs: Mapping[str, Any]) -> NDArray[np.float64]:
    """
    Thermal graph Laplacian L with L_ii = sum_j G_ij, L_ij = -G_ij.

    G_ij = 1/R_ij for adjacent pairs; 0 otherwise.
    """
    A = zone_adjacency_matrix(specs)
    G = float(1.0 / specs["multi_zone"]["R_ij_K_per_W"])
    W = A * G
    deg = np.sum(W, axis=1)
    L = np.diag(deg) - W
    return L.astype(np.float64)


def allocate_dampers(
    occupancy_weights: Sequence[float],
    *,
    empty_floor: float = 0.05,
) -> NDArray[np.float64]:
    """
    Map zone occupancy weights to damper fractions summing to 1.

    Empty zones keep a small floor share as thermal buffers.
    """
    w = np.asarray(occupancy_weights, dtype=np.float64)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("occupancy_weights must be 1-D nonempty")
    w = np.maximum(w, 0.0)
    if float(np.sum(w)) < 1e-12:
        u = np.full(w.size, 1.0 / w.size)
    else:
        u = w / np.sum(w)
    u = (1.0 - empty_floor) * u + empty_floor / w.size
    u = u / np.sum(u)
    return u.astype(np.float64)


def simulate_multi_zone(
    hours: int,
    weather_rows: Sequence[Mapping[str, float]],
    zone_occ: NDArray[np.floating],
    *,
    P_hvac_total_schedule: Sequence[float],
    specs: Mapping[str, Any] | None = None,
    x0_T: float = 24.0,
) -> Dict[str, Any]:
    """
    Simplified multi-zone air-node model with Laplacian coupling.

    Parameters
    ----------
    hours :
        Horizon length.
    weather_rows :
        Sequence of dicts with T_out_C, w_out_g_per_kg, C_out_ug_per_m3, I_solar.
    zone_occ :
        Shape (hours, Z) occupancy intensity in [0, 1].
    P_hvac_total_schedule :
        Length ``hours`` central plant sensible power [W] (neg=cool).

    Returns
    -------
    dict
        T_zone (H+1,Z), damper_U (H,Z), interzone_heat_W (H,Z), RH_mean (H+1,).
    """
    specs = dict(specs or load_building_specs())
    mz = specs["multi_zone"]
    names: List[str] = list(mz["zone_names"])
    z = len(names)
    scales = np.asarray(mz["C_air_scale"], dtype=np.float64)
    if zone_occ.shape != (hours, z):
        raise ValueError(f"zone_occ shape must be ({hours}, {z})")
    L = graph_laplacian_conductance(specs)
    rho = float(specs["rho_air_kg_per_m3"])
    c_p = float(specs["c_p_J_per_kgK"])
    Q_door = float(mz["Q_door_m3_per_s"])
    # Effective Laplacian with airflow edges: G_air = rho*cp*Q_door on adjacency.
    A = zone_adjacency_matrix(specs)
    L_air = np.diag(np.sum(A, axis=1) * rho * c_p * Q_door) - A * (rho * c_p * Q_door)
    L_tot = L + L_air

    C_air = float(specs["C_air_J_per_K"]) * scales
    R_win = float(specs["R_win_K_per_W"])
    T = np.zeros((hours + 1, z), dtype=np.float64)
    T[0] = x0_T
    U = np.zeros((hours, z), dtype=np.float64)
    Q_inter = np.zeros((hours, z), dtype=np.float64)
    RH_mean = np.zeros(hours + 1, dtype=np.float64)
    # Track a shared humidity/pollutant proxy via single-zone RC for IAQ KPI.
    w = 10.0
    C_pol = 25.0
    T_env = x0_T + 0.5
    RH_mean[0] = relative_humidity_pct(x0_T, w)

    for k in range(hours):
        row = weather_rows[k]
        T_out = float(row["T_out_C"])
        u = allocate_dampers(zone_occ[k])
        U[k] = u
        P_tot = float(P_hvac_total_schedule[k])
        # Continuous air-node ODE integrated with Euler on 12 substeps.
        T_k = T[k].copy()
        dt = 3600.0 / 12.0
        for _ in range(12):
            coup = -L_tot @ T_k
            Q_inter[k] = coup
            dT = np.zeros(z)
            for i in range(z):
                win = (T_out - T_k[i]) / R_win
                occ = float(specs["q_occ_W"]) * float(zone_occ[k, i])
                solar = (
                    float(specs["alpha_in"])
                    * float(specs["window_area_m2"])
                    / z
                    * float(row["I_solar_W_per_m2"])
                )
                dT[i] = (win + coup[i] + u[i] * P_tot + occ + solar) / C_air[i]
            T_k = T_k + dt * dT
        T[k + 1] = T_k
        # Shared IAQ column using mean zone temp as air node.
        d_iaq = Disturbance(
            T_out=T_out,
            w_out=float(row["w_out_g_per_kg"]),
            C_out=float(row["C_out_ug_per_m3"]),
            I_solar=float(row["I_solar_W_per_m2"]),
            n_occ=float(np.sum(zone_occ[k])),
        )
        u_iaq = ControlInput(
            P_HVAC=P_tot,
            m_hum=-0.2 if relative_humidity_pct(float(np.mean(T_k)), w) > 55 else 0.0,
            Q_vent=float(specs["Q_vent_min_m3_per_s"]) + 0.04 * float(np.mean(u)),
            u_pur=0.6,
        )
        x = np.asarray([float(np.mean(T_k)), T_env, w, C_pol], dtype=np.float64)
        for __ in range(12):
            x = rk4_step(x, u_iaq, d_iaq, specs, 300.0)
        T_env, w, C_pol = float(x[1]), float(x[2]), float(x[3])
        RH_mean[k + 1] = relative_humidity_pct(float(np.mean(T[k + 1])), w)

    return {
        "zone_names": names,
        "T_zone": T,
        "damper_U": U,
        "interzone_heat_W": Q_inter,
        "RH_mean": RH_mean,
        "C_pol_final": C_pol,
        "laplacian": L_tot,
    }

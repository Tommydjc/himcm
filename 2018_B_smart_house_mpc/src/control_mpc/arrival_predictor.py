"""
Irregular-arrival predictor: geofence ETA fused with weekday/weekend priors.

p_m(t+τ | F_t) = λ(τ) Φ((τ - τ̂_arr)/σ) + (1-λ(τ)) π_m(t+τ | day_type)
Lead time τ_lead from linearized 2R2C recovery to T_set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.stats import norm

PACK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = PACK_ROOT / "data" / "parameters" / "occupant_profiles.json"


def load_occupant_profiles(path: Path = DEFAULT_PROFILES) -> Dict[str, Any]:
    """Load multi-inhabitant preference / schedule priors."""
    return json.loads(path.read_text(encoding="utf-8"))


def prior_occupancy(
    inhabitant: Mapping[str, Any],
    hour_of_day: int,
    *,
    is_weekend: bool,
) -> float:
    """Calendar prior π_m in {0,1} softened to (0.05, 0.95)."""
    key = "home_base_hours_weekend" if is_weekend else "home_base_hours_weekday"
    home = int(hour_of_day) % 24 in set(inhabitant[key])
    return 0.92 if home else 0.08


def eta_hours(distance_km: float, speed_km_h: float, *, v_eps: float = 3.0) -> float:
    """Estimated time of arrival [h] from distance / radial speed."""
    return float(distance_km / max(speed_km_h, v_eps))


def arrival_probability(
    tau_h: float,
    *,
    distance_km: float,
    speed_km_h: float,
    prior: float,
    sigma_traffic_h: float = 0.35,
    gamma: float = 0.45,
) -> float:
    """
    Bayesian geofence–prior blend for one future horizon hour τ.

    λ(τ) = exp(-γ τ) elevates real-time ETA for short horizons.
    """
    tau_hat = eta_hours(distance_km, speed_km_h)
    lam = float(np.exp(-gamma * max(tau_h, 0.0)))
    # Prob already home / arriving by τ: CDF of arrival time.
    geo = float(norm.cdf((tau_h - tau_hat) / max(sigma_traffic_h, 1e-3)))
    # If currently very close and inbound, boost.
    if distance_km < 1.0 and speed_km_h > 5.0:
        geo = max(geo, 0.85)
    p = lam * geo + (1.0 - lam) * prior
    return float(np.clip(p, 0.0, 1.0))


def thermal_lead_time_h(
    T_in: float,
    T_set: float,
    T_out: float,
    *,
    P_hvac_W: float,
    C_air: float,
    R_win: float,
) -> float:
    """
    Approximate pre-conditioning lead time τ_lead [h].

    Linearized: C_air dT/dt ≈ (T_out - T)/R_win + P.
    Time to close (T_set - T_in) under constant cooling/heating.
    """
    gap = T_set - T_in
    if abs(gap) < 0.15:
        return 0.0
    # Steady drive toward set under constant P.
    # Effective rate ≈ (P + (T_out - T_in)/R) / C_air  [K/s]
    drive = (P_hvac_W + (T_out - T_in) / R_win) / max(C_air, 1.0)
    if gap < 0.0:
        # Need cooling: drive should be negative.
        P_use = -abs(P_hvac_W)
        drive = (P_use + (T_out - T_in) / R_win) / max(C_air, 1.0)
        if drive >= -1e-6:
            return 3.0  # cannot cool — cap
        tau_s = gap / drive  # gap<0, drive<0 → positive
    else:
        if drive <= 1e-6:
            return 3.0
        tau_s = gap / drive
    return float(np.clip(tau_s / 3600.0, 0.0, 6.0))


def build_occupancy_scenario(
    n_hours: int,
    profiles: Mapping[str, Any] | None = None,
    *,
    seed: int = 2018,
    irregular_events: Sequence[Dict[str, float]] | None = None,
) -> Dict[str, Any]:
    """
    Synthesize multi-inhabitant occupancy + geofence telemetry.

    irregular_events :
        Optional list of {hour_index, inhabitant_idx, distance_km, speed_km_h}
        sudden returns that break the calendar prior.
    """
    profiles = profiles or load_occupant_profiles()
    people = list(profiles["inhabitants"])
    rng = np.random.default_rng(seed)
    M = len(people)
    occ = np.zeros((n_hours, M), dtype=np.float64)
    dist = np.zeros((n_hours, M), dtype=np.float64)
    speed = np.zeros((n_hours, M), dtype=np.float64)
    p_arr = np.zeros((n_hours, M), dtype=np.float64)

    for k in range(n_hours):
        hod = k % 24
        is_weekend = (k // 24) % 7 >= 5
        for m, person in enumerate(people):
            prior = prior_occupancy(person, hod, is_weekend=is_weekend)
            # Nominal distance: far when prior low.
            if prior > 0.5:
                d_km = float(rng.uniform(0.0, 0.4))
                v = float(rng.uniform(0.0, 2.0))
                occ[k, m] = 1.0 if rng.random() < prior else 0.0
            else:
                d_km = float(rng.uniform(8.0, 25.0))
                v = float(rng.uniform(15.0, 45.0))
                occ[k, m] = 0.0
            dist[k, m] = d_km
            speed[k, m] = v
            p_arr[k, m] = arrival_probability(
                0.0 if occ[k, m] > 0.5 else eta_hours(d_km, v),
                distance_km=d_km,
                speed_km_h=v,
                prior=prior,
            )
            if occ[k, m] > 0.5:
                p_arr[k, m] = max(p_arr[k, m], 0.95)

    # Inject irregular returns (geofence beats calendar).
    events = list(irregular_events or [])
    if not events:
        events = [
            {"hour_index": 14, "inhabitant_idx": 1, "distance_km": 4.0, "speed_km_h": 35.0},
            {"hour_index": 38, "inhabitant_idx": 0, "distance_km": 6.0, "speed_km_h": 40.0},
            {"hour_index": 62, "inhabitant_idx": 1, "distance_km": 3.5, "speed_km_h": 30.0},
        ]
    for ev in events:
        k = int(ev["hour_index"])
        m = int(ev["inhabitant_idx"])
        if k >= n_hours or m >= M:
            continue
        # Mark inbound for k and next 2 hours occupied.
        for j in range(k, min(n_hours, k + 3)):
            dist[j, m] = float(ev["distance_km"]) * max(0.2, 1.0 - 0.4 * (j - k))
            speed[j, m] = float(ev["speed_km_h"])
            prior = prior_occupancy(people[m], j % 24, is_weekend=(j // 24) % 7 >= 5)
            p_arr[j, m] = arrival_probability(
                float(j - k),
                distance_km=dist[j, m],
                speed_km_h=speed[j, m],
                prior=prior,
            )
            if j >= k + 1:
                occ[j, m] = 1.0

    return {
        "occupancy": occ,
        "distance_km": dist,
        "speed_km_h": speed,
        "p_arrival": p_arr,
        "names": [p["id"] for p in people],
        "profiles": people,
        "rho_fair": float(profiles.get("nash_fairness_rho", 0.35)),
    }


def nash_social_discomfort(
    T_in: float,
    RH: float,
    C_pol: float,
    hour_of_day: int,
    p_occ: Sequence[float],
    people: Sequence[Mapping[str, Any]],
    *,
    rho_fair: float,
) -> float:
    """
    Expected Nash social discomfort L_comfort.

    L = sum_m p_m D_m + ρ max_m (p_m D_m)
    """
    Ds: List[float] = []
    weighted = 0.0
    for m, person in enumerate(people):
        night = hour_of_day < 6 or hour_of_day >= 22
        T_pref = float(person["T_pref_night_C"] if night else person["T_pref_day_C"])
        RH_pref = float(person["RH_pref_pct"])
        C_th = float(person["C_thresh_ug_per_m3"])
        Dm = (
            float(person["omega_T"]) * (T_in - T_pref) ** 2
            + float(person["omega_H"]) * (RH - RH_pref) ** 2
            + float(person["omega_P"]) * max(0.0, C_pol - C_th) ** 2
        )
        pm = float(p_occ[m]) if m < len(p_occ) else 0.0
        weighted += pm * Dm
        Ds.append(pm * Dm)
    return float(weighted + rho_fair * (max(Ds) if Ds else 0.0))

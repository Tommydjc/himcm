"""
Multi-inhabitant preference bargaining via local Ollama (qwen2.5:7b).

Falls back to asymmetric Nash product maximization if Ollama is offline.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from control_mpc.arrival_predictor import load_occupant_profiles, nash_social_discomfort


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"


def _ollama_available(timeout_s: float = 1.5) -> bool:
    """Return True if local Ollama responds to /api/tags."""
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status) == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _ollama_propose_setpoints(
    people: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> Optional[Dict[str, float]]:
    """Ask Ollama for a compromise (T, RH) pair; None on failure."""
    prompt = (
        "You are a household climate mediator. "
        "Return ONLY JSON with keys T_set_C and RH_set_pct. "
        f"Residents: {json.dumps([{k: p[k] for k in ('id','T_pref_day_C','RH_pref_pct','allergy_weight') for p in people}])}. "
        f"Context: {json.dumps(context)}. "
        "Bias toward lower RH and cleaner air when any allergy_weight > 1.5."
    )
    body = json.dumps(
        {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}
    ).encode("utf-8")
    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = payload.get("response", "{}")
        data = json.loads(text) if isinstance(text, str) else text
        return {
            "T_set_C": float(data["T_set_C"]),
            "RH_set_pct": float(data["RH_set_pct"]),
            "source": "ollama",
        }
    except Exception:
        return None


def asymmetric_nash_bargain(
    people: Sequence[Mapping[str, Any]],
    *,
    T_grid: Optional[np.ndarray] = None,
    RH_grid: Optional[np.ndarray] = None,
    C_pol: float = 20.0,
    hour_of_day: int = 19,
) -> Dict[str, Any]:
    """
    Maximize product_m (U_max,m - D_m)^{w_m} over a (T, RH) grid.

    Weights w_m = allergy_weight (asymmetric bargaining powers).
    """
    T_grid = T_grid if T_grid is not None else np.linspace(20.0, 26.0, 25)
    RH_grid = RH_grid if RH_grid is not None else np.linspace(35.0, 55.0, 21)
    best: Tuple[float, float, float] = (23.0, 45.0, -1e99)
    front: List[Dict[str, float]] = []
    for T in T_grid:
        for RH in RH_grid:
            utilities = []
            weights = []
            for p in people:
                night = hour_of_day < 6 or hour_of_day >= 22
                T_pref = float(p["T_pref_night_C"] if night else p["T_pref_day_C"])
                Dm = (
                    float(p["omega_T"]) * (T - T_pref) ** 2
                    + float(p["omega_H"]) * (RH - float(p["RH_pref_pct"])) ** 2
                    + float(p["omega_P"]) * max(0.0, C_pol - float(p["C_thresh_ug_per_m3"])) ** 2
                )
                # Map discomfort to utility with positive surplus.
                U = max(1e-6, 80.0 - Dm)
                utilities.append(U)
                weights.append(float(p.get("allergy_weight", 1.0)))
            # Nash product with asymmetric exponents.
            log_prod = 0.0
            for U, w in zip(utilities, weights):
                log_prod += w * np.log(U)
            score = float(log_prod)
            front.append({"T_set_C": float(T), "RH_set_pct": float(RH), "nash_log": score})
            if score > best[2]:
                best = (float(T), float(RH), score)
    # Approximate Pareto: keep non-dominated on mean discomfort vs max allergy discomfort.
    return {
        "T_set_C": best[0],
        "RH_set_pct": best[1],
        "nash_log": best[2],
        "source": "asymmetric_nash",
        "front_size": len(front),
    }


def bargain_setpoints(
    *,
    C_pol: float = 20.0,
    hour_of_day: int = 19,
    use_ollama: bool = True,
) -> Dict[str, Any]:
    """
    High-level API: try Ollama, else asymmetric Nash grid search.
    """
    profiles = load_occupant_profiles()
    people = list(profiles["inhabitants"])
    context = {
        "hour_of_day": hour_of_day,
        "C_pol_ug_m3": C_pol,
        "goal": "Pareto compromise temperature and humidity",
    }
    if use_ollama and _ollama_available():
        prop = _ollama_propose_setpoints(people, context)
        if prop is not None:
            # Evaluate social discomfort at proposal for audit.
            p_occ = [1.0] * len(people)
            L = nash_social_discomfort(
                prop["T_set_C"],
                prop["RH_set_pct"],
                C_pol,
                hour_of_day,
                p_occ,
                people,
                rho_fair=float(profiles.get("nash_fairness_rho", 0.35)),
            )
            prop["nash_social_discomfort"] = L
            return prop
    result = asymmetric_nash_bargain(
        people, C_pol=C_pol, hour_of_day=hour_of_day
    )
    p_occ = [1.0] * len(people)
    result["nash_social_discomfort"] = nash_social_discomfort(
        result["T_set_C"],
        result["RH_set_pct"],
        C_pol,
        hour_of_day,
        p_occ,
        people,
        rho_fair=float(profiles.get("nash_fairness_rho", 0.35)),
    )
    return result

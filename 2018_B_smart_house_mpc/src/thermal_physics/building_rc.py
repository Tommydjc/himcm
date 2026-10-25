"""
Teaching-track second-order RC (2R2C) thermal network for a single dwelling.

Phase 1 of the AeroTherm dual-track plan:
  - Authoritative heat dynamics for Bang-Bang vs prediction-matrix MPC.
  - RK4 integration at Δt = 60 s.
  - Outdoor-temperature dependent COP for site electric power.

State: (T_in, T_wall) in °C.
Control heat rate q_hvac > 0 heating, < 0 cooling (W, sensible).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

PACK_ROOT = Path(__file__).resolve().parents[2]
TEACHING_SPECS = PACK_ROOT / "data" / "parameters" / "building_specs_teaching.json"


def load_teaching_specs(path: Path = TEACHING_SPECS) -> Dict[str, Any]:
    """Load Phase-1 teaching envelope JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class BuildingThermalEnvelope:
    """
    Lumped-parameter envelope for a standard single-family teaching house.

    Derived geometry
    ----------------
    volume_m3 = floor_area_m2 * ceiling_height_m
    wall_area_m2 ≈ 4 * sqrt(floor_area_m2) * ceiling_height_m  (square plan)
    window_area_m2 = window_wall_ratio * wall_area_m2
    """

    floor_area_m2: float = 180.0
    ceiling_height_m: float = 2.8
    window_wall_ratio: float = 0.20
    C_air_J_per_C: float = 6.2e5
    C_wall_J_per_C: float = 1.8e7
    R_in_C_per_W: float = 0.08
    R_out_C_per_W: float = 0.12
    R_win_C_per_W: float = 0.25
    alpha_solar_win: float = 0.08
    alpha_solar_wall: float = 0.04
    COP_ref: float = 3.8
    COP_slope_per_C: float = 0.04
    COP_T_ref_C: float = 25.0
    COP_min: float = 1.5
    Q_cool_max_W: float = 5000.0
    Q_heat_max_W: float = 5000.0
    dt_s: float = 60.0

    @classmethod
    def from_json(cls, path: Path = TEACHING_SPECS) -> "BuildingThermalEnvelope":
        """Construct from teaching specs file."""
        raw = load_teaching_specs(path)
        keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: float(raw[k]) for k in keys if k in raw})

    @property
    def volume_m3(self) -> float:
        return float(self.floor_area_m2 * self.ceiling_height_m)

    @property
    def wall_area_m2(self) -> float:
        side = float(np.sqrt(self.floor_area_m2))
        return float(4.0 * side * self.ceiling_height_m)

    @property
    def window_area_m2(self) -> float:
        return float(self.window_wall_ratio * self.wall_area_m2)

    def cop(self, T_out_C: float) -> float:
        """
        Cooling/heating COP derating with outdoor temperature.

        COP(T_out) = COP_ref - slope * (T_out - T_ref), floored at COP_min.
        """
        raw = self.COP_ref - self.COP_slope_per_C * (float(T_out_C) - self.COP_T_ref_C)
        return float(max(self.COP_min, raw))

    def electric_power_W(self, q_hvac_W: float, T_out_C: float) -> float:
        """Map sensible HVAC heat rate to site electric demand [W]."""
        return float(abs(q_hvac_W) / max(self.cop(T_out_C), 1e-9))

    def clip_q_hvac(self, q_hvac_W: float) -> float:
        """Project HVAC heat rate onto actuator bounds."""
        return float(np.clip(q_hvac_W, -self.Q_cool_max_W, self.Q_heat_max_W))


class ThermalDynamics2R2C:
    """
    Continuous-time 2R2C dynamics.

    C_air * dT_in/dt =
        (T_wall - T_in)/R_in + (T_out - T_in)/R_win
        + q_hvac + internal_gains + alpha_win * A_win * solar

    C_wall * dT_wall/dt =
        (T_in - T_wall)/R_in + (T_out - T_wall)/R_out
        + alpha_wall * A_wall_eff * solar
    """

    def __init__(self, envelope: Optional[BuildingThermalEnvelope] = None) -> None:
        self.env = envelope or BuildingThermalEnvelope.from_json()

    def compute_derivatives(
        self,
        T_in: float,
        T_wall: float,
        T_out: float,
        solar_rad: float,
        q_hvac: float,
        internal_gains: float,
    ) -> Tuple[float, float]:
        """
        Return (dT_in/dt, dT_wall/dt) in °C/s.

        Parameters
        ----------
        T_in, T_wall, T_out :
            Temperatures [°C].
        solar_rad :
            Global horizontal / facade irradiance proxy [W/m²].
        q_hvac :
            Sensible HVAC heat rate [W] (+ heat, − cool), pre-clip recommended.
        internal_gains :
            Occupant/appliance sensible gains [W].
        """
        e = self.env
        q = e.clip_q_hvac(q_hvac)
        I = max(0.0, float(solar_rad))
        # Effective opaque absorbing area: shaded facade fraction.
        A_wall_sol = 0.15 * e.wall_area_m2

        dT_in = (
            (T_wall - T_in) / e.R_in_C_per_W
            + (T_out - T_in) / e.R_win_C_per_W
            + q
            + float(internal_gains)
            + e.alpha_solar_win * e.window_area_m2 * I
        ) / e.C_air_J_per_C

        dT_wall = (
            (T_in - T_wall) / e.R_in_C_per_W
            + (T_out - T_wall) / e.R_out_C_per_W
            + e.alpha_solar_wall * A_wall_sol * I
        ) / e.C_wall_J_per_C

        return float(dT_in), float(dT_wall)

    def continuous_matrices(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Linear LTI form useful for Phase-4 prediction matrices:

            x_dot = A_c x + B_c u + E_c d

        with x=[T_in, T_wall], u=q_hvac, d=[T_out, solar, internal_gains]
        (solar/internal enter as affine inputs with fixed area coefficients).
        """
        e = self.env
        gin = 1.0 / e.R_in_C_per_W
        gwin = 1.0 / e.R_win_C_per_W
        gout = 1.0 / e.R_out_C_per_W
        Ca, Cw = e.C_air_J_per_C, e.C_wall_J_per_C
        A_wall_sol = 0.15 * e.wall_area_m2

        A_c = np.array(
            [
                [-(gin + gwin) / Ca, gin / Ca],
                [gin / Cw, -(gin + gout) / Cw],
            ],
            dtype=np.float64,
        )
        B_c = np.array([[1.0 / Ca], [0.0]], dtype=np.float64)
        E_c = np.array(
            [
                [gwin / Ca, e.alpha_solar_win * e.window_area_m2 / Ca, 1.0 / Ca],
                [gout / Cw, e.alpha_solar_wall * A_wall_sol / Cw, 0.0],
            ],
            dtype=np.float64,
        )
        return A_c, B_c, E_c


def rk4_step(
    model: ThermalDynamics2R2C,
    T_in: float,
    T_wall: float,
    *,
    T_out: float,
    solar_rad: float,
    q_hvac: float,
    internal_gains: float,
    dt_s: float,
) -> Tuple[float, float]:
    """One RK4 step of length ``dt_s`` seconds."""

    def f(ti: float, tw: float) -> Tuple[float, float]:
        return model.compute_derivatives(
            ti, tw, T_out, solar_rad, q_hvac, internal_gains
        )

    k1_i, k1_w = f(T_in, T_wall)
    k2_i, k2_w = f(T_in + 0.5 * dt_s * k1_i, T_wall + 0.5 * dt_s * k1_w)
    k3_i, k3_w = f(T_in + 0.5 * dt_s * k2_i, T_wall + 0.5 * dt_s * k2_w)
    k4_i, k4_w = f(T_in + dt_s * k3_i, T_wall + dt_s * k3_w)
    T_in_n = T_in + (dt_s / 6.0) * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i)
    T_wall_n = T_wall + (dt_s / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)
    return float(T_in_n), float(T_wall_n)


def simulate_rk4(
    model: ThermalDynamics2R2C,
    T_in0: float,
    T_wall0: float,
    *,
    T_out: Sequence[float] | float,
    solar_rad: Sequence[float] | float = 0.0,
    q_hvac: Sequence[float] | float = 0.0,
    internal_gains: Sequence[float] | float = 0.0,
    n_steps: Optional[int] = None,
    dt_s: Optional[float] = None,
) -> Dict[str, NDArray[np.float64]]:
    """
    Forward RK4 simulation.

    Scalar exogenous inputs are held constant; sequences must share length.
    """
    dt = float(dt_s if dt_s is not None else model.env.dt_s)
    if n_steps is None:
        if isinstance(T_out, (int, float)):
            raise ValueError("n_steps required when T_out is scalar")
        n_steps = len(T_out)  # type: ignore[arg-type]

    def _as_arr(v: Sequence[float] | float) -> NDArray[np.float64]:
        if isinstance(v, (int, float)):
            return np.full(n_steps, float(v), dtype=np.float64)
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (n_steps,):
            raise ValueError(f"input length {arr.shape} != n_steps={n_steps}")
        return arr

    T_out_a = _as_arr(T_out)
    solar_a = _as_arr(solar_rad)
    q_a = _as_arr(q_hvac)
    g_a = _as_arr(internal_gains)

    T_in = np.zeros(n_steps + 1, dtype=np.float64)
    T_wall = np.zeros(n_steps + 1, dtype=np.float64)
    P_elec = np.zeros(n_steps, dtype=np.float64)
    T_in[0] = float(T_in0)
    T_wall[0] = float(T_wall0)

    for k in range(n_steps):
        q = model.env.clip_q_hvac(float(q_a[k]))
        P_elec[k] = model.env.electric_power_W(q, float(T_out_a[k]))
        T_in[k + 1], T_wall[k + 1] = rk4_step(
            model,
            float(T_in[k]),
            float(T_wall[k]),
            T_out=float(T_out_a[k]),
            solar_rad=float(solar_a[k]),
            q_hvac=q,
            internal_gains=float(g_a[k]),
            dt_s=dt,
        )

    return {
        "T_in": T_in,
        "T_wall": T_wall,
        "P_elec_W": P_elec,
        "t_s": np.arange(n_steps + 1, dtype=np.float64) * dt,
    }


def thermal_energy_J(
    env: BuildingThermalEnvelope,
    T_in: float,
    T_wall: float,
    *,
    T_ref: float = 0.0,
) -> float:
    """Lumped thermal energy relative to T_ref [J]."""
    return float(
        env.C_air_J_per_C * (T_in - T_ref) + env.C_wall_J_per_C * (T_wall - T_ref)
    )


def assert_steady_state_is_outdoor(T_out: float = 35.0, tol: float = 1e-9) -> None:
    """
    With q=0, I=0, gains=0, equilibrium of A_c x + E_c[:,0]*T_out = 0 is x=[T_out,T_out].
    """
    model = ThermalDynamics2R2C(BuildingThermalEnvelope.from_json())
    A_c, _B_c, E_c = model.continuous_matrices()
    # A_c x_ss + E_c @ [T_out, 0, 0] = 0  =>  x_ss = -A_c^{-1} E_c[:,0] T_out
    rhs = E_c[:, 0] * T_out
    x_ss = np.linalg.solve(A_c, -rhs)
    if float(np.max(np.abs(x_ss - T_out))) > tol:
        raise AssertionError(f"steady state {x_ss} != T_out={T_out}")
    eig = np.linalg.eigvals(A_c)
    if not bool(np.all(np.real(eig) < 0.0)):
        raise AssertionError(f"A_c not Hurwitz: {eig}")


def assert_free_float_to_outdoor(
    *,
    T_out: float = 35.0,
    T_in0: float = 22.0,
    T_wall0: float = 24.0,
    hours: float = 168.0,
    min_progress: float = 0.45,
    tol_final_long: float = 1.0,
) -> Dict[str, float]:
    """
    HVAC off, no solar/gains: T_in rises smoothly toward T_out.

    Strong wall capacitance makes full settling multi-week; we therefore:
      1) prove analytical steady state is T_out (Hurwitz A_c);
      2) require monotone approach and sufficient progress over ``hours``;
      3) check lumped energy balance ΔE ≈ ∫Q_net dt.
    """
    assert_steady_state_is_outdoor(T_out=T_out)

    model = ThermalDynamics2R2C(BuildingThermalEnvelope.from_json())
    dt = model.env.dt_s
    n = int(hours * 3600.0 / dt)
    out = simulate_rk4(
        model,
        T_in0,
        T_wall0,
        T_out=T_out,
        solar_rad=0.0,
        q_hvac=0.0,
        internal_gains=0.0,
        n_steps=n,
        dt_s=dt,
    )
    T_in = out["T_in"]
    T_wall = out["T_wall"]

    dT = np.diff(T_in)
    if T_in0 < T_out:
        if not bool(np.all(dT >= -1e-9)):
            raise AssertionError("T_in is not non-decreasing toward hotter outdoor")
    else:
        if not bool(np.all(dT <= 1e-9)):
            raise AssertionError("T_in is not non-increasing toward cooler outdoor")

    gap0 = abs(T_out - T_in0)
    progress = 1.0 - abs(float(T_in[-1]) - T_out) / max(gap0, 1e-9)
    if progress < min_progress:
        raise AssertionError(
            f"insufficient free-float progress {progress:.3f} < {min_progress} "
            f"(T_in final={T_in[-1]:.3f}, T_out={T_out})"
        )

    final_err = abs(float(T_in[-1]) - T_out)
    # Optional tighter check only when simulating very long horizons.
    if hours >= 24.0 * 21.0 and final_err > tol_final_long:
        raise AssertionError(
            f"T_in final {T_in[-1]:.3f} not within {tol_final_long} of T_out={T_out}"
        )

    e = model.env
    E0 = thermal_energy_J(e, float(T_in[0]), float(T_wall[0]), T_ref=T_out)
    E1 = thermal_energy_J(e, float(T_in[-1]), float(T_wall[-1]), T_ref=T_out)
    Q_net = (T_out - T_in[:-1]) / e.R_win_C_per_W + (T_out - T_wall[:-1]) / e.R_out_C_per_W
    Q_int = float(np.sum(Q_net) * dt)
    resid = abs((E1 - E0) - Q_int)
    scale = max(1.0, abs(Q_int), abs(E1 - E0))
    if resid / scale > 2e-3:
        raise AssertionError(
            f"energy residual too large: |ΔE - ∫Q|={resid:.3f} J, scale={scale:.3f}"
        )

    return {
        "final_T_in": float(T_in[-1]),
        "final_T_wall": float(T_wall[-1]),
        "final_err": float(final_err),
        "progress": float(progress),
        "energy_residual_J": float(resid),
        "energy_residual_rel": float(resid / scale),
        "n_steps": float(n),
        "hours": float(hours),
    }


def main() -> None:
    """Phase-1 self-check entrypoint."""
    env = BuildingThermalEnvelope.from_json()
    print("=== BuildingThermalEnvelope (teaching) ===")
    print(f"floor={env.floor_area_m2} m²  H={env.ceiling_height_m} m  WWR={env.window_wall_ratio}")
    print(f"V={env.volume_m3:.1f} m³  A_wall={env.wall_area_m2:.1f}  A_win={env.window_area_m2:.1f}")
    print(f"C_air={env.C_air_J_per_C:.3e}  C_wall={env.C_wall_J_per_C:.3e}")
    print(f"R_in={env.R_in_C_per_W}  R_out={env.R_out_C_per_W}  R_win={env.R_win_C_per_W}")
    print(f"COP(35°C)={env.cop(35.0):.3f}  COP(25°C)={env.cop(25.0):.3f}")

    model = ThermalDynamics2R2C(env)
    A_c, B_c, E_c = model.continuous_matrices()
    print("A_c =\n", A_c)
    print("B_c =\n", B_c)
    print("E_c =\n", E_c)

    stats = assert_free_float_to_outdoor()
    print("=== free-float self-check PASSED ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

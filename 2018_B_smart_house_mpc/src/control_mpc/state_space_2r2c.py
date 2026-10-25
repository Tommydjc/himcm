"""
Continuous → discrete LTI state-space for the teaching 2R2C envelope.

Follows MPC_02 (ZOH / Euler discretization):
  x_dot = A_c x + B_c u + E_c d
  x_{k+1} = A_d x_k + B_d u_k + E_d d_k

Default control step T_s = 15 min for Phase-5 rolling MPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from thermal_physics.building_rc import BuildingThermalEnvelope, ThermalDynamics2R2C

DiscretizationMethod = Literal["zoh", "euler"]

# Teaching MPC control interval (seconds).
DEFAULT_TS_S = 15.0 * 60.0


@dataclass(frozen=True)
class ContinuousSS:
    """Continuous-time state-space triple (A_c, B_c, E_c)."""

    A_c: NDArray[np.float64]
    B_c: NDArray[np.float64]
    E_c: NDArray[np.float64]

    @property
    def n(self) -> int:
        return int(self.A_c.shape[0])

    @property
    def m(self) -> int:
        return int(self.B_c.shape[1])

    @property
    def n_d(self) -> int:
        return int(self.E_c.shape[1])


@dataclass(frozen=True)
class DiscreteSS:
    """Discrete-time state-space with sampling period Ts."""

    A_d: NDArray[np.float64]
    B_d: NDArray[np.float64]
    E_d: NDArray[np.float64]
    Ts: float
    method: str

    @property
    def n(self) -> int:
        return int(self.A_d.shape[0])

    @property
    def m(self) -> int:
        return int(self.B_d.shape[1])


def build_continuous_ss(
    envelope: Optional[BuildingThermalEnvelope] = None,
) -> ContinuousSS:
    """Build A_c, B_c, E_c from the teaching envelope."""
    model = ThermalDynamics2R2C(envelope or BuildingThermalEnvelope.from_json())
    A_c, B_c, E_c = model.continuous_matrices()
    return ContinuousSS(A_c=A_c, B_c=B_c, E_c=E_c)


def discretize_zoh(css: ContinuousSS, Ts: float) -> DiscreteSS:
    """
    Exact ZOH discretization via Van Loan augmented exponential.

    Van Loan augmented exponential for ZOH hold of inputs.

    For x_dot = A x + G w with w constant on [0, Ts):
      M = [[A, G], [0, 0]] * Ts
      exp(M) = [[A_d, G_d], [0, I]]
    Here G = [B_c | E_c], w = [u; d].
    """
    n, m, nd = css.n, css.m, css.n_d
    G = np.hstack([css.B_c, css.E_c])  # (n, m+nd)
    nw = m + nd
    M = np.zeros((n + nw, n + nw), dtype=np.float64)
    M[:n, :n] = css.A_c * Ts
    M[:n, n:] = G * Ts
    expM = expm(M)
    A_d = expM[:n, :n]
    G_d = expM[:n, n:]
    B_d = G_d[:, :m]
    E_d = G_d[:, m:]
    return DiscreteSS(A_d=A_d, B_d=B_d, E_d=E_d, Ts=float(Ts), method="zoh")


def discretize_euler(css: ContinuousSS, Ts: float) -> DiscreteSS:
    """Forward-Euler approximation: A_d ≈ I + A_c Ts, B_d ≈ B_c Ts."""
    n = css.n
    A_d = np.eye(n) + css.A_c * Ts
    B_d = css.B_c * Ts
    E_d = css.E_c * Ts
    return DiscreteSS(A_d=A_d, B_d=B_d, E_d=E_d, Ts=float(Ts), method="euler")


def discretize(
    css: ContinuousSS,
    Ts: float = DEFAULT_TS_S,
    *,
    method: DiscretizationMethod = "zoh",
) -> DiscreteSS:
    """Dispatch ZOH or Euler discretization."""
    if method == "zoh":
        return discretize_zoh(css, Ts)
    if method == "euler":
        return discretize_euler(css, Ts)
    raise ValueError(f"unknown method {method}")


def step_discrete(
    dss: DiscreteSS,
    x: NDArray[np.floating],
    u: float | NDArray[np.floating],
    d: NDArray[np.floating],
) -> NDArray[np.float64]:
    """One discrete step: x+ = A x + B u + E d."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    u_vec = np.atleast_1d(np.asarray(u, dtype=np.float64)).reshape(-1)
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    return dss.A_d @ x + (dss.B_d @ u_vec).reshape(-1) + (dss.E_d @ d).reshape(-1)


def is_schur_stable(A_d: NDArray[np.floating], tol: float = 1e-9) -> bool:
    """Discrete stability: all eigenvalues inside the unit circle."""
    eigs = np.linalg.eigvals(np.asarray(A_d, dtype=np.float64))
    return bool(np.all(np.abs(eigs) < 1.0 - tol + 1e-12))


def main() -> None:
    """Print continuous/discrete matrices and stability check."""
    css = build_continuous_ss()
    print("=== Continuous 2R2C ===")
    print("A_c =\n", css.A_c)
    print("B_c =\n", css.B_c)
    print("E_c =\n", css.E_c)
    dss = discretize(css, DEFAULT_TS_S, method="zoh")
    print(f"\n=== ZOH discrete Ts={dss.Ts:.0f}s ===")
    print("A_d =\n", dss.A_d)
    print("B_d =\n", dss.B_d)
    print("E_d =\n", dss.E_d)
    print("Schur stable:", is_schur_stable(dss.A_d))
    dss_e = discretize(css, DEFAULT_TS_S, method="euler")
    print("||A_zoh - A_euler|| =", float(np.linalg.norm(dss.A_d - dss_e.A_d)))


if __name__ == "__main__":
    main()

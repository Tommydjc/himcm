"""
MPC prediction matrices 𝒜, ℬ (and disturbance ℰ) from discrete LTI.

X = 𝒜 x0 + ℬ U + ℰ D
  X stacked [x_1; ...; x_N]
  U stacked [u_0; ...; u_{N-1}]
  D stacked [d_0; ...; d_{N-1}]

Reference: InterviewGuide MPC_02 — state space & prediction equations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from control_mpc.state_space_2r2c import (
    DEFAULT_TS_S,
    DiscreteSS,
    build_continuous_ss,
    discretize,
    is_schur_stable,
    step_discrete,
)


def build_prediction_matrices(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    N: int,
    E: Optional[NDArray[np.floating]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]:
    """
    Build calA (N n × n), calB (N n × N m), optional calE (N n × N n_d).

    calB / calE are block lower-triangular (causality).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = A.shape[0]
    m = B.shape[1]
    if A.shape != (n, n) or B.shape[0] != n:
        raise ValueError("A/B dimension mismatch")
    if N < 1:
        raise ValueError("N must be >= 1")

    calA = np.zeros((N * n, n), dtype=np.float64)
    calB = np.zeros((N * n, N * m), dtype=np.float64)
    Apow = A.copy()
    for k in range(N):
        calA[k * n : (k + 1) * n, :] = Apow
        Apow = A @ Apow

    for j in range(N):
        AkB = B.copy()
        for k in range(j, N):
            calB[k * n : (k + 1) * n, j * m : (j + 1) * m] = AkB
            AkB = A @ AkB

    calE: Optional[NDArray[np.float64]] = None
    if E is not None:
        E = np.asarray(E, dtype=np.float64)
        nd = E.shape[1]
        if E.shape[0] != n:
            raise ValueError("E row dim must match A")
        calE = np.zeros((N * n, N * nd), dtype=np.float64)
        for j in range(N):
            AkE = E.copy()
            for k in range(j, N):
                calE[k * n : (k + 1) * n, j * nd : (j + 1) * nd] = AkE
                AkE = A @ AkE

    return calA, calB, calE


def build_from_discrete(
    dss: DiscreteSS,
    N: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convenience wrapper using a DiscreteSS object."""
    calA, calB, calE = build_prediction_matrices(dss.A_d, dss.B_d, N, E=dss.E_d)
    assert calE is not None
    return calA, calB, calE


def check_causality(
    calB: NDArray[np.floating],
    n: int,
    m: int,
    *,
    tol: float = 1e-10,
) -> bool:
    """Upper block-triangle of calB must be ~0."""
    N = calB.shape[0] // n
    for k in range(N):
        for j in range(k + 1, N):
            block = calB[k * n : (k + 1) * n, j * m : (j + 1) * m]
            if float(np.linalg.norm(block)) > tol:
                return False
    return True


def simulate_open_loop(
    dss: DiscreteSS,
    x0: NDArray[np.floating],
    U: NDArray[np.floating],
    D: NDArray[np.floating],
    N: int,
) -> NDArray[np.float64]:
    """
    Step-by-step discrete simulation; return stacked X = [x_1; ...; x_N].
    """
    n, m = dss.n, dss.m
    nd = dss.E_d.shape[1]
    U = np.asarray(U, dtype=np.float64).reshape(N, m)
    D = np.asarray(D, dtype=np.float64).reshape(N, nd)
    x = np.asarray(x0, dtype=np.float64).reshape(n)
    Xs = []
    for k in range(N):
        x = step_discrete(dss, x, U[k], D[k])
        Xs.append(x.copy())
    return np.concatenate(Xs, axis=0)


def predict_matrix(
    calA: NDArray[np.floating],
    calB: NDArray[np.floating],
    calE: NDArray[np.floating],
    x0: NDArray[np.floating],
    U: NDArray[np.floating],
    D: NDArray[np.floating],
) -> NDArray[np.float64]:
    """X = calA x0 + calB U + calE D (U,D flattened)."""
    return (
        calA @ np.asarray(x0, dtype=np.float64).reshape(-1)
        + calB @ np.asarray(U, dtype=np.float64).reshape(-1)
        + calE @ np.asarray(D, dtype=np.float64).reshape(-1)
    )


def verify_prediction(
    dss: DiscreteSS,
    N: int = 24,
    *,
    seed: int = 2018,
    tol: float = 1e-9,
) -> Dict[str, float]:
    """
    Matrix prediction vs step simulation; expect machine-precision agreement.
    """
    rng = np.random.default_rng(seed)
    n, m = dss.n, dss.m
    nd = dss.E_d.shape[1]
    x0 = rng.normal(size=n) * 2.0 + np.array([24.0, 25.0])[:n]
    U = rng.uniform(-3000.0, 0.0, size=(N, m))  # cooling-ish
    D = np.column_stack(
        [
            rng.uniform(28.0, 36.0, size=N),  # T_out
            rng.uniform(0.0, 600.0, size=N),  # solar
            rng.uniform(0.0, 400.0, size=N),  # gains
        ]
    )

    calA, calB, calE = build_from_discrete(dss, N)
    if not check_causality(calB, n, m):
        raise AssertionError("calB violates causality (upper blocks nonzero)")
    if calE is not None and not check_causality(calE, n, nd):
        raise AssertionError("calE violates causality")

    X_mat = predict_matrix(calA, calB, calE, x0, U, D)
    X_sim = simulate_open_loop(dss, x0, U, D, N)
    err = float(np.max(np.abs(X_mat - X_sim)))
    if err > tol:
        raise AssertionError(f"prediction mismatch max|err|={err} > tol={tol}")
    return {
        "N": float(N),
        "max_abs_err": err,
        "rms_err": float(np.sqrt(np.mean((X_mat - X_sim) ** 2))),
        "schur_stable": 1.0 if is_schur_stable(dss.A_d) else 0.0,
        "Ts": float(dss.Ts),
    }


def main() -> None:
    """Phase-4 self-check entrypoint."""
    css = build_continuous_ss()
    dss = discretize(css, DEFAULT_TS_S, method="zoh")
    print(f"=== Prediction matrices N=24, Ts={dss.Ts:.0f}s (ZOH) ===")
    stats = verify_prediction(dss, N=24, tol=1e-9)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    # Also verify Euler path (looser structural check only on causality+consistency).
    dss_e = discretize(css, DEFAULT_TS_S, method="euler")
    stats_e = verify_prediction(dss_e, N=12, tol=1e-9)
    print("Euler verify max_abs_err:", stats_e["max_abs_err"])
    print("=== Phase 4 self-check PASSED ===")


if __name__ == "__main__":
    main()

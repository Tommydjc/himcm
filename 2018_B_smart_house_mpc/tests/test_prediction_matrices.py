"""Phase-4 state-space + prediction-matrix tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / "src"))

from control_mpc.prediction_matrices import (
    build_from_discrete,
    build_prediction_matrices,
    check_causality,
    verify_prediction,
)
from control_mpc.state_space_2r2c import (
    DEFAULT_TS_S,
    build_continuous_ss,
    discretize,
    is_schur_stable,
)


def test_continuous_dims() -> None:
    css = build_continuous_ss()
    assert css.A_c.shape == (2, 2)
    assert css.B_c.shape == (2, 1)
    assert css.E_c.shape == (2, 3)


def test_zoh_schur_stable() -> None:
    dss = discretize(build_continuous_ss(), DEFAULT_TS_S, method="zoh")
    assert is_schur_stable(dss.A_d)


def test_prediction_matches_simulation_zoh() -> None:
    dss = discretize(build_continuous_ss(), DEFAULT_TS_S, method="zoh")
    stats = verify_prediction(dss, N=24, tol=1e-9)
    assert stats["max_abs_err"] < 1e-9


def test_prediction_matches_simulation_euler() -> None:
    dss = discretize(build_continuous_ss(), DEFAULT_TS_S, method="euler")
    stats = verify_prediction(dss, N=16, tol=1e-9)
    assert stats["max_abs_err"] < 1e-9


def test_calB_causality() -> None:
    dss = discretize(build_continuous_ss(), DEFAULT_TS_S, method="zoh")
    _A, calB, calE = build_from_discrete(dss, N=8)
    assert check_causality(calB, dss.n, dss.m)
    assert calE is not None
    assert check_causality(calE, dss.n, dss.E_d.shape[1])


def test_spring_mass_like_structure() -> None:
    """Tiny hand check: N=2, shapes and first block of calB == B."""
    A = np.array([[0.9, 0.1], [0.0, 0.8]])
    B = np.array([[0.0], [0.1]])
    calA, calB, _ = build_prediction_matrices(A, B, N=2)
    assert calA.shape == (4, 2)
    assert calB.shape == (4, 2)
    np.testing.assert_allclose(calB[0:2, 0:1], B)
    np.testing.assert_allclose(calB[2:4, 0:1], A @ B)
    np.testing.assert_allclose(calB[2:4, 1:2], B)
    np.testing.assert_allclose(calB[0:2, 1:2], 0.0)

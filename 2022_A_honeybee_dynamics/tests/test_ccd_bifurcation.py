"""CCD 分岔与胁迫护栏。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.stress_test.ccd_bifurcation import (
    STRESS_CCD,
    STRESS_FIELD,
    STRESS_NONE,
    detect_tipping_point,
    run_bifurcation_sweep,
    simulate_year,
    summarize_trajectory,
)


class CCDBifurcationTests(unittest.TestCase):
    """胁迫生效、拐点检测、摘要字段。"""

    def test_stress_reduces_overwinter(self) -> None:
        healthy = summarize_trajectory(simulate_year(0.14, STRESS_NONE), 0.14)
        stressed = summarize_trajectory(simulate_year(0.14, STRESS_CCD), 0.14)
        self.assertLess(stressed["n_overwinter"], healthy["n_overwinter"])

    def test_short_sweep_and_tipping(self) -> None:
        frame = run_bifurcation_sweep(mu_min=0.08, mu_max=0.40, step=0.04, stress=STRESS_FIELD)
        self.assertGreaterEqual(len(frame), 4)
        self.assertIn("conversion_age_proxy_days", frame.columns)
        tip_soft = detect_tipping_point(frame, n_collapse=5000.0)
        tip_hard = detect_tipping_point(frame, n_collapse=3000.0)
        self.assertTrue(np.isfinite(tip_soft["mu_star_threshold"]))
        self.assertTrue(np.isfinite(tip_hard["mu_star_threshold"]))
        self.assertGreaterEqual(tip_hard["mu_star_threshold"], tip_soft["mu_star_threshold"])


if __name__ == "__main__":
    unittest.main()

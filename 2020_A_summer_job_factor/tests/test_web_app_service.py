"""Navigator mapping, wage parse, and Softmax ranking (no Streamlit)."""

from __future__ import annotations

import unittest

import numpy as np

from src.web_app.service import (
    factor_to_radar,
    load_navigator_bundle,
    parse_wage_range,
    profile_to_x,
    rank_jobs,
    slider_to_factor,
)


class MappingTests(unittest.TestCase):
    def test_slider_endpoints(self) -> None:
        self.assertAlmostEqual(slider_to_factor(1.0, -2.0, 4.0), -2.0)
        self.assertAlmostEqual(slider_to_factor(10.0, -2.0, 4.0), 4.0)
        mid = slider_to_factor(5.5, -2.0, 4.0)
        self.assertAlmostEqual(mid, 1.0, places=6)

    def test_radar_roundtrip_mid(self) -> None:
        f = slider_to_factor(7.0, -1.0, 2.0)
        back = factor_to_radar(f, -1.0, 2.0)
        self.assertAlmostEqual(back, 7.0, places=6)

    def test_wage_parse_lifeguard_span(self) -> None:
        text = "Typical wage: $15.50 to $18.00 per hour before overtime."
        self.assertIn("15.50", parse_wage_range(text))
        self.assertIn("18.00", parse_wage_range(text))


class RankingTests(unittest.TestCase):
    def test_eight_jobs_sum_to_one(self) -> None:
        bundle = load_navigator_bundle()
        ranked = rank_jobs(bundle, 9.0, 5.0, 8.0)
        self.assertEqual(len(ranked), 8)
        total = sum(c.proba for c in ranked)
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertGreaterEqual(ranked[0].proba, ranked[1].proba)
        self.assertGreaterEqual(ranked[1].proba, ranked[-1].proba)
        x = profile_to_x(9.0, 5.0, 8.0, bundle.col_min, bundle.col_max)
        self.assertEqual(x.shape, (1, 3))
        self.assertTrue(np.all(x >= bundle.col_min - 1.0e-12))
        self.assertTrue(np.all(x <= bundle.col_max + 1.0e-12))
        self.assertIn("$", ranked[0].wage_range)


if __name__ == "__main__":
    unittest.main()

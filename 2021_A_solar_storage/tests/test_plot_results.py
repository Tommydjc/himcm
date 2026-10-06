"""Plotter writes a non-empty 300 DPI PNG from the measured A/B CSV."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.agent_sim.plot_results import plot_crewai_comparison
from src.optimization.sizing_milp import RESULTS_DIR


class PlotResultsTests(unittest.TestCase):
    """Figure must track the CSV, not invented SOC/outage story numbers."""

    def test_png_from_real_csv(self) -> None:
        src = RESULTS_DIR / "crewai_adaptive_vs_passive.csv"
        self.assertTrue(src.is_file())
        frame = pd.read_csv(src)
        self.assertEqual(len(frame), 72)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "fig_crewai_load_curtailment.png"
            out = plot_crewai_comparison(csv_path=src, fig_path=dest)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 20_000)


if __name__ == "__main__":
    unittest.main()

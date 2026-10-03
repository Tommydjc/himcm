"""Autograd 弹性流水线护栏。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.sensitivity.autograd_sens import (
    LABEL_POP,
    compute_elasticities,
    plot_elasticity_bars,
    run_elasticity_pipeline,
)


class AutogradSensTests(unittest.TestCase):
    """弹性表形状、符号与出图。"""

    def test_elasticity_table(self) -> None:
        frame = compute_elasticities()
        self.assertIn("elasticity", frame.columns)
        self.assertEqual(len(frame), 10)  # 5 params × 2 objectives
        pop = frame.loc[frame["objective"] == LABEL_POP]
        self.assertEqual(len(pop), 5)
        self.assertTrue((pop["abs_elasticity"] >= 0).all())
        # 外勤死亡对越冬规模应为负向脆弱点
        mort = pop.loc[pop["parameter"] == "forager_mortality"].iloc[0]
        self.assertLess(float(mort["elasticity"]), 0.0)

    def test_pipeline_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / "autograd_elasticity_ranking.csv"
            fig = root / "fig_autograd_elasticity_bars.png"
            frame = run_elasticity_pipeline(csv_path=csv, fig_path=fig)
            self.assertTrue(csv.is_file())
            self.assertTrue(fig.is_file())
            loaded = pd.read_csv(csv)
            self.assertEqual(len(loaded), len(frame))


if __name__ == "__main__":
    unittest.main()

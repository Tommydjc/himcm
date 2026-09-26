"""loader：xlsx 出场次数与研究因子合并。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indicators.loader import (
    BRISBANE_CANDIDATE_FACTORS,
    TARGET_CODES,
    build_brisbane_candidates,
    build_sde_matrix,
)
from src.indicators.schema import (
    HOST_POPULARITY_COLUMN,
    MATRIX_EXPORT_COLUMNS,
    REQUIRED_INDICATOR_COLUMNS,
)


class TestSdeLoader(unittest.TestCase):
    """九个指定 Code 均能从官方表对上，校验列无空、无负。"""

    def test_build_matrix_has_nine_target_rows(self) -> None:
        frame = build_sde_matrix()
        self.assertEqual(list(frame["Code"]), list(TARGET_CODES))
        for col in REQUIRED_INDICATOR_COLUMNS:
            self.assertIn(col, frame.columns)
        self.assertFalse(frame.loc[:, list(REQUIRED_INDICATOR_COLUMNS)].isna().any().any())
        self.assertGreater(int(frame.loc[frame["Code"] == "ATH", "Appearances"].iloc[0]), 20)
        self.assertGreater(float(frame.loc[frame["Code"] == "SWM", "Events_Count"].iloc[0]), 0.0)
        self.assertEqual(
            str(frame.loc[frame["Code"] == "CKT", "Cohort"].iloc[0]),
            "candidate_2032",
        )
        self.assertEqual(list(frame.columns), list(MATRIX_EXPORT_COLUMNS))

    def test_brisbane_candidates_match_locked_scenario(self) -> None:
        matrix = build_sde_matrix()
        candidates = build_brisbane_candidates()
        self.assertEqual(list(candidates["Code"]), ["CKT", "SQU", "AFB"])
        self.assertEqual(list(candidates.columns), list(matrix.columns))
        self.assertEqual(list(candidates.columns), list(MATRIX_EXPORT_COLUMNS))
        self.assertFalse(candidates[HOST_POPULARITY_COLUMN].isna().any())
        for code, factors in BRISBANE_CANDIDATE_FACTORS.items():
            row = candidates.loc[candidates["Code"] == code].iloc[0]
            for column, expected in factors.items():
                self.assertAlmostEqual(float(row[column]), float(expected), places=6)


if __name__ == "__main__":
    unittest.main()

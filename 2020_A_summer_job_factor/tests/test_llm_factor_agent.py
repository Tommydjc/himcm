"""Unit checks for LLMFactor naming, JSON parse, and mock job scores."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.llm_factor.llm_factor_agent import (
    CONTEST_NARRATIVE,
    clip_unit_interval_score,
    keyword_item_scores,
    list_job_files,
    load_rotated_loadings,
    mock_job_scores,
    mock_names,
    parse_job_id,
    parse_names_json,
    parse_scores_json,
    run_llm_factor_pipeline,
    top_abs_loadings,
)


class LoadingRankTests(unittest.TestCase):
    def test_pack_top4_factor1_matches_csv_order(self) -> None:
        frame = load_rotated_loadings()
        top = top_abs_loadings(frame, 1, k=4)
        self.assertEqual(len(top), 4)
        self.assertEqual(top[0].item, "safety_level")
        self.assertGreater(top[0].abs_loading, top[3].abs_loading)
        items = [t.item for t in top]
        self.assertIn("social_networking", items)
        self.assertIn("skill_acquisition", items)
        self.assertIn("tip_potential", items)

    def test_contest_narrative_not_equal_to_extracted_f1_leader(self) -> None:
        frame = load_rotated_loadings()
        leader = top_abs_loadings(frame, 1, k=1)[0].item
        self.assertNotEqual(leader, "hourly_wage_need")
        self.assertEqual(CONTEST_NARRATIVE[0][2], "Immediate Financial Yield")


class JsonAndClipTests(unittest.TestCase):
    def test_clip(self) -> None:
        self.assertEqual(clip_unit_interval_score(4.2), 3.0)
        self.assertEqual(clip_unit_interval_score(-9.0), -3.0)
        self.assertEqual(clip_unit_interval_score(0.5), 0.5)

    def test_parse_names(self) -> None:
        names = parse_names_json(
            {
                "factors": [
                    {"factor_id": 2, "name_zh": "乙", "name_en": "Beta", "theory": "t2"},
                    {"factor_id": 1, "name_zh": "甲", "name_en": "Alpha", "theory": "t1"},
                    {"factor_id": 3, "name_zh": "丙", "name_en": "Gamma", "theory": "t3"},
                ]
            }
        )
        self.assertEqual(names[0].name_en, "Alpha")
        self.assertEqual(names[2].factor_id, 3)

    def test_parse_scores_clips(self) -> None:
        vec, why = parse_scores_json(
            {"scores": {"F1": 8, "F2": -4, "F3": 0.25}, "rationale": "ok"}
        )
        np.testing.assert_allclose(vec, [3.0, -3.0, 0.25])
        self.assertEqual(why, "ok")


class MockScorerTests(unittest.TestCase):
    def test_keyword_tutor_is_indoor_low_sun(self) -> None:
        text = Path("data/job_descriptions/2_private_tutor.txt").read_text()
        names = ("outdoor_sun_exposure", "hourly_wage_need", "autonomous_control")
        s = keyword_item_scores(text, names)
        self.assertLess(s[0], 0.0)
        self.assertGreater(s[1], 0.0)

    def test_mock_projection_shape(self) -> None:
        text = Path("data/job_descriptions/0_lifeguard.txt").read_text()
        items = ("physical_strength", "outdoor_sun_exposure", "autonomous_control")
        lam = np.array([[0.8, 0.1, 0.0], [0.7, 0.0, 0.2], [0.0, 0.9, 0.1]], dtype=np.float64)
        scores, rationale = mock_job_scores(text, items, lam)
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.all(scores >= -3.0) and np.all(scores <= 3.0))
        self.assertIn("Mock", rationale)

    def test_mock_names_reject_designed_triad(self) -> None:
        frame = load_rotated_loadings()
        from src.llm_factor.llm_factor_agent import all_top_items

        names = mock_names(all_top_items(frame))
        joined = " ".join(n.name_en for n in names)
        self.assertNotIn("Immediate Financial Yield", joined)
        self.assertNotIn("Human Capital & Career Value", joined)


class PipelineIoTests(unittest.TestCase):
    def test_eight_job_files(self) -> None:
        files = list_job_files()
        self.assertEqual(len(files), 8)
        ids = [parse_job_id(p) for p in files]
        self.assertEqual(ids, list(range(8)))

    def test_mock_pipeline_writes_md_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md = root / "factor_interpretation.md"
            csv_path = root / "job_factor_matrix.csv"
            result = run_llm_factor_pipeline(
                interpretation_path=md,
                matrix_path=csv_path,
                backend="mock",
            )
            self.assertEqual(result.backend, "mock")
            self.assertTrue(md.is_file())
            self.assertTrue(csv_path.is_file())
            text = md.read_text(encoding="utf-8")
            self.assertIn("safety_level", text)
            self.assertIn("Immediate Financial Yield", text)
            self.assertIn("not what Kaiser-normalized Varimax recovered", text)
            frame = pd.read_csv(csv_path)
            self.assertEqual(len(frame), 8)
            for col in ("Factor_1", "Factor_2", "Factor_3"):
                self.assertTrue(frame[col].between(-3.0, 3.0).all())
            self.assertEqual(set(frame["job_id"].tolist()), set(range(8)))


if __name__ == "__main__":
    unittest.main()

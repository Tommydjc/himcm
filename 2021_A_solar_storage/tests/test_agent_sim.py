"""crewAI contract tests that do not require a live Ollama model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agent_sim.crew_engine import (
    PROTECTED,
    check_ollama_status,
    heuristic_agreement,
    load_appliance_catalog,
    run_family_negotiation,
    sanitize_agreement,
)
from src.agent_sim.models import LoadSheddingAgreement


class AgentSimTests(unittest.TestCase):
    """Pydantic contract, catalog names, mock fallback."""

    def test_agreement_fields(self) -> None:
        item = LoadSheddingAgreement(
            consensus_reached=True,
            shed_appliances=["Electric_Clothes_Dryer", "Dishwasher"],
            hvac_temp_offset_c=2.5,
            negotiated_reduction_kW=2.1,
            rationale="Keep router; defer dryer.",
        )
        self.assertTrue(item.consensus_reached)
        self.assertAlmostEqual(item.hvac_temp_offset_c, 2.5)

    def test_negative_hvac_clipped(self) -> None:
        item = LoadSheddingAgreement(
            consensus_reached=False,
            shed_appliances=[],
            hvac_temp_offset_c=-3.0,
            negotiated_reduction_kW=-1.0,
            rationale="x",
        )
        self.assertGreaterEqual(item.hvac_temp_offset_c, 0.0)
        self.assertGreaterEqual(item.negotiated_reduction_kW, 0.0)

    def test_sanitize_drops_protected(self) -> None:
        catalog = load_appliance_catalog()
        dirty = LoadSheddingAgreement(
            consensus_reached=True,
            shed_appliances=["Electric_Clothes_Dryer", "WiFi_Router_Comm", "NotAThing"],
            hvac_temp_offset_c=2.5,
            negotiated_reduction_kW=0.0,
            rationale="test",
        )
        clean = sanitize_agreement(dirty, catalog)
        self.assertIn("Electric_Clothes_Dryer", clean.shed_appliances)
        self.assertNotIn("WiFi_Router_Comm", clean.shed_appliances)
        self.assertNotIn("NotAThing", clean.shed_appliances)
        self.assertGreater(clean.negotiated_reduction_kW, 0.0)

    def test_heuristic_uses_catalog_only(self) -> None:
        catalog = load_appliance_catalog()
        names = set(catalog["appliance_name"].astype(str))
        agr = heuristic_agreement(catalog, soc=0.18, target_cut_frac=0.45)
        for item in agr.shed_appliances:
            self.assertIn(item, names)
            self.assertNotIn(item, PROTECTED)
        self.assertTrue(agr.consensus_reached)

    def test_check_ollama_does_not_raise(self) -> None:
        ok, msg = check_ollama_status(timeout_s=1.0)
        self.assertIsInstance(ok, bool)
        self.assertTrue(len(msg) > 0)

    def test_run_force_mock_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crew_load_shedding_agreement.json"
            agr = run_family_negotiation(force_mock=True, out_json=path)
            self.assertTrue(path.is_file())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("shed_appliances", payload)
            self.assertEqual(payload["hvac_temp_offset_c"], agr.hvac_temp_offset_c)


if __name__ == "__main__":
    unittest.main()

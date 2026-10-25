"""Tests for Ollama/Mock household MAS comfort negotiation."""

from __future__ import annotations

from agent_inhabitant.cognitive_thermostat import (
    ComfortConsensus,
    consensus_to_comfort,
    negotiate_via_mock,
    run_household_negotiation,
)


def test_mock_consensus_respects_elder_floor() -> None:
    """Mock bargain never cools below 21.5°C."""
    c = negotiate_via_mock({"extreme_peak_day": True})
    data = c.model_dump()
    assert data["agreed_temp_min"] >= 21.5
    assert data["agreed_temp_max"] <= 25.0
    assert data["comfort_weight_scale"] >= 1.0


def test_consensus_maps_to_mpc_config() -> None:
    """ComfortConsensus → MPCComfortConfig preserves band and λ scale."""
    c = ComfortConsensus(
        agreed_temp_min=21.5,
        agreed_temp_max=25.0,
        comfort_weight_scale=1.25,
        consensus_notes="unit-test",
    )
    cfg = consensus_to_comfort(c)
    assert cfg.T_min_C == 21.5
    assert cfg.T_max_C == 25.0
    assert abs(cfg.lambda_comf - 0.05 * 1.25) < 1e-12
    assert cfg.T_precool_C >= 21.5


def test_negotiation_always_returns_contract() -> None:
    """Offline path still yields a valid consensus dict."""
    out = run_household_negotiation(extreme_peak_day=True)
    assert "consensus" in out
    assert out["consensus"]["agreed_temp_min"] >= 21.5
    assert out["source"] in {"langchain_ollama", "ollama_http", "mock_heuristic"}

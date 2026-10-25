"""Inhabitant agents: legacy Nash MAS + cognitive thermostat (Prompt 6)."""

from agent_inhabitant.cognitive_thermostat import (
    ComfortConsensus,
    consensus_to_comfort,
    run_household_negotiation,
)

__all__ = [
    "ComfortConsensus",
    "consensus_to_comfort",
    "run_household_negotiation",
]

"""Phase 6: JSONL run_logger unit tests."""

from __future__ import annotations

import json
from pathlib import Path

from eval_harness.run_logger import RunLogger, new_session


def test_emit_appends_jsonl(tmp_path: Path) -> None:
    """session_start + mpc_step + export produce parseable lines."""
    log = RunLogger(session_id="testphase6", log_dir=tmp_path, track="teaching")
    log.session_start(component="pytest")
    log.mpc_step(
        ctrl_step=1,
        hour_of_day=14,
        u0=0.2,
        cost=3.14,
        nit=4,
        success=True,
        rate_usd_per_kwh=0.48,
        T_in_C=23.5,
    )
    log.export(paths={"out": "dummy.csv"})
    events = log.read_events()
    assert len(events) == 3
    assert events[0]["event"] == "session_start"
    assert events[1]["event"] == "mpc_step"
    assert events[1]["u0"] == 0.2
    assert events[2]["event"] == "export"
    # File is valid JSONL.
    raw = log.path.read_text(encoding="utf-8").strip().splitlines()
    assert all(json.loads(line)["session_id"] == "testphase6" for line in raw)


def test_compare_batch_slims_metrics(tmp_path: Path) -> None:
    """compare_batch keeps scalar KPIs only."""
    log = new_session(track="teaching")
    # Redirect to tmp by replacing path via fresh logger.
    log = RunLogger(session_id="cmp6", log_dir=tmp_path, track="teaching")
    log.session_start()
    log.compare_batch(
        {
            "cost_savings_pct": 4.2,
            "peak_energy_savings_pct": 15.8,
            "short_cycle_reduction": 39,
            "bang": {"controller": "bang_bang", "cost_usd": 1.682, "extra": "drop"},
            "mpc": {"controller": "aerotherm_mpc", "cost_usd": 1.611},
        }
    )
    ev = log.read_events()[-1]
    assert ev["event"] == "compare_batch"
    assert ev["cost_savings_pct"] == 4.2
    assert "extra" not in ev["bang"]
    assert ev["bang"]["cost_usd"] == 1.682

"""
JSONL run logger for teaching-track / Streamlit observability (Phase 6).

Events are append-only lines under ``logs/run_<session_id>.jsonl``.
Never invent KPIs — only log what callers pass.

Run self-check:
  PYTHONPATH=src python -m eval_harness.run_logger
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

PACK_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PACK_ROOT / "logs"


def _utc_now_iso() -> str:
    """UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunLogger:
    """
    Append-only JSONL session logger.

    Parameters
    ----------
    session_id :
        Optional fixed id; otherwise a short uuid4 hex.
    log_dir :
        Directory for ``run_<id>.jsonl`` files.
    track :
        ``teaching`` or ``legacy`` label written into every event.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        *,
        log_dir: Path = LOG_DIR,
        track: str = "teaching",
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.track = track
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"run_{self.session_id}.jsonl"
        self._n_events = 0

    def emit(self, event: str, **payload: Any) -> Dict[str, Any]:
        """
        Append one JSON object and return it.

        Parameters
        ----------
        event :
            Event name, e.g. ``session_start``, ``mpc_step``, ``export``.
        payload :
            Arbitrary JSON-serializable fields (floats, ints, str, dict, list).
        """
        record: Dict[str, Any] = {
            "ts": _utc_now_iso(),
            "session_id": self.session_id,
            "track": self.track,
            "event": event,
            "seq": self._n_events,
        }
        record.update(payload)
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self._n_events += 1
        return record

    def session_start(self, **meta: Any) -> Dict[str, Any]:
        """Mark the beginning of a run session."""
        return self.emit("session_start", **meta)

    def mpc_step(
        self,
        *,
        ctrl_step: int,
        hour_of_day: int,
        u0: float,
        cost: float,
        nit: int,
        success: bool,
        rate_usd_per_kwh: Optional[float] = None,
        T_in_C: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Log one rolling-horizon MPC solve."""
        return self.emit(
            "mpc_step",
            ctrl_step=int(ctrl_step),
            hour_of_day=int(hour_of_day),
            u0=float(u0),
            cost=float(cost),
            nit=int(nit),
            success=bool(success),
            rate_usd_per_kwh=rate_usd_per_kwh,
            T_in_C=T_in_C,
        )

    def bang_day(self, metrics: Mapping[str, Any]) -> Dict[str, Any]:
        """Log Bang-Bang day summary metrics."""
        return self.emit("bang_day", metrics=dict(metrics))

    def compare_batch(self, comparison: Mapping[str, Any]) -> Dict[str, Any]:
        """Log Bang vs MPC comparison payload (no nested trajectories)."""
        slim = {
            k: comparison[k]
            for k in (
                "cost_savings_pct",
                "energy_savings_pct",
                "peak_energy_savings_pct",
                "peak_cost_savings_pct",
                "short_cycle_reduction",
            )
            if k in comparison
        }
        if "bang" in comparison and isinstance(comparison["bang"], Mapping):
            slim["bang"] = _metrics_slim(comparison["bang"])
        if "mpc" in comparison and isinstance(comparison["mpc"], Mapping):
            slim["mpc"] = _metrics_slim(comparison["mpc"])
        return self.emit("compare_batch", **slim)

    def export(self, *, paths: Mapping[str, str], **extra: Any) -> Dict[str, Any]:
        """Log artifact write paths."""
        return self.emit("export", paths=dict(paths), **extra)

    def read_events(self) -> list[Dict[str, Any]]:
        """Load all events from this session file."""
        if not self.path.is_file():
            return []
        out: list[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out


def _metrics_slim(m: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep scalar KPI keys only."""
    keep = (
        "controller",
        "energy_kwh",
        "cost_usd",
        "peak_energy_kwh",
        "peak_cost_usd",
        "peak_cost_share",
        "short_cycle_starts",
        "mean_T_in_C",
        "mean_u_peak",
        "mean_u_prepeak",
    )
    return {k: m[k] for k in keep if k in m}


def _json_default(obj: Any) -> Any:
    """Fallback serializer for NumPy scalars."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def new_session(*, track: str = "teaching", **meta: Any) -> RunLogger:
    """Create a logger and emit ``session_start``."""
    logger = RunLogger(track=track)
    logger.session_start(**meta)
    return logger


def main() -> None:
    """Tiny self-check writing a throwaway JSONL file."""
    log = new_session(component="run_logger_selfcheck")
    log.emit("run_once_teaching", note="dry-run")
    log.mpc_step(
        ctrl_step=0,
        hour_of_day=10,
        u0=0.4,
        cost=1.23,
        nit=5,
        success=True,
        rate_usd_per_kwh=0.22,
        T_in_C=23.0,
    )
    log.export(paths={"selfcheck": str(log.path)})
    events = log.read_events()
    assert len(events) >= 3
    assert events[0]["event"] == "session_start"
    print(f"[run_logger] wrote {log.path} ({len(events)} events)")


if __name__ == "__main__":
    main()

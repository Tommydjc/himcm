#!/usr/bin/env python3
"""干旱与割草对照（转发全量流水线中的两段）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scripts.run_all import run_drought, run_mowing_sweep


def main() -> None:
    drought = run_drought()
    pareto = run_mowing_sweep()
    print(f"drought rows={len(drought)} pareto rows={len(pareto)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Office 规则规划器 vs GAT-PPO：仅从真实 rollout 写论文对比图。

本脚本不再使用手填分数。评估与绘图见 ``src/utils/plot_rl_figures.py``。

运行::

    PYTHONPATH=. python scripts/compare_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.plot_rl_figures import main as plot_rl_main


if __name__ == "__main__":
    plot_rl_main()

"""Office 场景：2 名消防员评分规划器最小回归。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment.layouts import get_office_layout, searchable_nodes
from src.traditional_planner.scoring_planner import ScoringPlanner


class TestOfficeScoringPlanner(unittest.TestCase):
    """验证 Figure 1 办公室在无火条件下可被两人清空且无死锁。"""

    t_sweep_floor: float = 20.0

    def test_two_firefighters_clear_office_without_deadlock(self) -> None:
        graph = get_office_layout()
        rooms = searchable_nodes(graph)
        self.assertEqual(len(rooms), 6)

        planner = ScoringPlanner(
            graph,
            hazard=None,
            start_nodes=("E_W", "E_E"),
            n_agents=2,
        )
        result = planner.run(max_steps=2000)

        self.assertFalse(result.deadlock, msg=result.message)
        self.assertTrue(result.success, msg=result.message)
        self.assertEqual(set(result.room_clear_times.keys()), set(rooms))
        self.assertGreater(result.t_clear, 0.0)
        self.assertGreater(result.n_steps, 0)
        self.assertEqual(len(result.room_clear_times), 6)
        for room, stamp in result.room_clear_times.items():
            self.assertGreaterEqual(stamp, 0.0)
            self.assertLessEqual(stamp, result.t_clear + 1e-9)

        ordered = sorted(result.room_clear_times.values())
        for earlier, later in zip(ordered, ordered[1:]):
            self.assertLessEqual(earlier, later)
        self.assertGreaterEqual(result.t_clear, self.t_sweep_floor)

    def test_tabu_and_momentum_prefer_column_sibling(self) -> None:
        """挂牌后禁忌最近房间，动量使同列兄弟房优先于沿走廊的下一列。"""
        graph = get_office_layout()
        planner = ScoringPlanner(graph, start_nodes=("E_W", "E_E"), n_agents=2)
        result = planner.run(max_steps=2000)
        self.assertTrue(result.success)
        f1 = planner.trajectories()["F1"]
        self.assertLess(f1.index("N1"), f1.index("S1"))
        self.assertLess(f1.index("S1"), f1.index("N2"))
        self.assertEqual(list(planner._agents[0].tabu_rooms), ["N1", "S1", "N2"])
        self.assertEqual(planner.tabu_tenure, 3)


if __name__ == "__main__":
    unittest.main()

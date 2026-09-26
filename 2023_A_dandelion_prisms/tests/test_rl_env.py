"""RL 环境：8 维观测、早春重罚、缓冲带只切外圈。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.env_wrapper import DandelionEcoEnv, SPRING_PENALTY


class TestDandelionEcoEnv(unittest.TestCase):
    """Gymnasium 契约。"""

    def test_obs_and_episode_length(self) -> None:
        env = DandelionEcoEnv(weather_seed=0)
        obs, info = env.reset(options={"weather_seed": 0})
        self.assertEqual(obs.shape, (8,))
        steps = 0
        terminated = False
        while not terminated:
            obs, _r, terminated, _t, info = env.step(0)
            steps += 1
        self.assertEqual(steps, 365)
        self.assertTrue(terminated)
        self.assertEqual(int(info["day"]), 365)

    def test_spring_core_mow_penalty(self) -> None:
        env = DandelionEcoEnv(weather_seed=1)
        env.reset(options={"weather_seed": 1})
        while env.day_index < 70:
            env.step(0)
        _obs, reward, _d, _t, info = env.step(2)
        self.assertIn(int(info["month"]), (3, 4))
        self.assertLess(reward, -SPRING_PENALTY + 5.0)

    def test_buffer_keeps_center_adult(self) -> None:
        env = DandelionEcoEnv(weather_seed=2)
        env.reset(options={"weather_seed": 2})
        env.state.adult[:, :] = 0.0
        env.state.adult[5, 5] = 4.0
        env.state.adult[0, 0] = 4.0
        env.step(1)
        self.assertGreater(float(env.state.adult[5, 5]), 3.0)
        self.assertLess(float(env.state.adult[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()

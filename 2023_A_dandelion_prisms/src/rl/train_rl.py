#!/usr/bin/env python3
"""PPO 训练 + 评估出图。

运行::

    PYTHONPATH=. /opt/anaconda3/envs/pytorch_env/bin/python -m src.rl.train_rl
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from src.rl.compare_policies import evaluate_and_plot
from src.rl.env_wrapper import DandelionEcoEnv
from src.rl.ppo_agent import PPOAgent

CKPT: Path = PACK_ROOT / "src" / "rl" / "checkpoints" / "ppo_dandelion_policy.pt"
N_EPISODES: int = 220
LOG_EVERY: int = 20
WEATHER_SEED: int = 2023


def train(n_episodes: int = N_EPISODES) -> Path:
    """训 ``n_episodes`` 年，保存最高回报权重。"""
    env = DandelionEcoEnv(weather_seed=WEATHER_SEED)
    agent = PPOAgent()
    print(f"device={agent.device} episodes={n_episodes}", flush=True)
    best = -1.0e18
    window: list[float] = []
    t0 = time.perf_counter()
    for ep in range(1, n_episodes + 1):
        obs, _info = env.reset(options={"weather_seed": WEATHER_SEED})
        obs_buf: list[np.ndarray] = []
        act_buf: list[int] = []
        logp_buf: list[float] = []
        rew_buf: list[float] = []
        done_buf: list[float] = []
        val_buf: list[float] = []
        ep_ret = 0.0
        terminated = False
        last_info: dict[str, object] = {}
        while not terminated:
            action, logp, value = agent.act(obs)
            nxt, reward, terminated, _trunc, info = env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(reward)
            done_buf.append(1.0 if terminated else 0.0)
            val_buf.append(value)
            ep_ret += reward
            last_info = info
            obs = nxt
        agent.update(
            np.stack(obs_buf).astype(np.float32),
            np.asarray(act_buf, dtype=np.int64),
            np.asarray(logp_buf, dtype=np.float32),
            np.asarray(rew_buf, dtype=np.float32),
            np.asarray(done_buf, dtype=np.float32),
            np.asarray(val_buf, dtype=np.float32),
        )
        window.append(ep_ret)
        if ep_ret > best:
            best = ep_ret
            agent.save(CKPT)
        if ep % LOG_EVERY == 0:
            mean_r = float(np.mean(window[-LOG_EVERY:]))
            print(
                f"ep={ep:03d} mean_R={mean_r:8.2f} best={best:8.2f} "
                f"mows={last_info.get('n_mows')} sprays={last_info.get('n_sprays')} "
                f"bee={float(last_info.get('bee_sum', 0.0)):6.2f}",
                flush=True,
            )
    elapsed = time.perf_counter() - t0
    print(f"train_s={elapsed:.2f} wrote {CKPT.relative_to(PACK_ROOT)}")
    return CKPT


def main() -> None:
    train()
    evaluate_and_plot(CKPT, weather_seed=WEATHER_SEED)


if __name__ == "__main__":
    main()

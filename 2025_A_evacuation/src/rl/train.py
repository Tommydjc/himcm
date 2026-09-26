#!/usr/bin/env python3
"""GAT-PPO 课程学习入口：Office、2 名消防员、200 个 Episode 预训练。

运行::

    PYTHONPATH=. python src/rl/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment.layouts import get_office_layout
from src.rl.env_wrapper import GraphEvacuationEnv
from src.rl.gat_net import GATActorCritic
from src.rl.ppo_agent import PPOAgent, select_device

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "gat_ppo_office.pt"
N_PRETRAIN_EPISODES: int = 200
LOG_EVERY: int = 10
EVAL_EPISODES: int = 10


def evaluate_policy(
    env: GraphEvacuationEnv,
    agent: PPOAgent,
    n_episodes: int = EVAL_EPISODES,
) -> dict[str, float]:
    """贪心策略评估：平均回报、步数、存活率、全清率。"""
    returns: list[float] = []
    steps: list[float] = []
    survivals: list[float] = []
    clears: list[float] = []
    for _ in range(n_episodes):
        obs, _info = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        n_steps = 0
        last_info = _info
        while not (done or truncated):
            actions, _logp, _v = agent.act(obs, deterministic=True)
            obs, reward, done, truncated, last_info = env.step(actions)
            ep_ret += float(reward)
            n_steps += 1
        returns.append(ep_ret)
        steps.append(float(n_steps))
        survivals.append(1.0 if last_info.get("survived", False) else 0.0)
        clears.append(1.0 if last_info.get("all_clear", False) else 0.0)
    return {
        "reward": float(np.mean(returns)),
        "steps": float(np.mean(steps)),
        "survival": float(np.mean(survivals)),
        "clear_rate": float(np.mean(clears)),
    }


def train_curriculum(
    n_episodes: int = N_PRETRAIN_EPISODES,
    *,
    seed: int = 42,
) -> Path:
    """Office 课程第 1 级：无火预训练，保存 ``gat_ppo_office.pt``。"""
    device = select_device()
    print(f"[GAT-PPO] device={device}  episodes={n_episodes}  scene=office  agents=2")
    graph = get_office_layout()
    env = GraphEvacuationEnv(
        graph,
        n_agents=2,
        max_steps=80,
        enable_hazard=False,
        seed=seed,
    )
    net = GATActorCritic(
        in_dim=6,
        hidden_dim=64,
        heads=4,
        max_actions=env.max_actions,
    )
    agent = PPOAgent(net, device=device)
    recent_r: list[float] = []
    recent_steps: list[int] = []
    recent_surv: list[float] = []

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        ep_ret = 0.0
        done = False
        truncated = False
        last_info = info
        while not (done or truncated):
            actions, logps, value = agent.act(obs, deterministic=False)
            next_obs, reward, done, truncated, last_info = env.step(actions)
            agent.remember(obs, actions, logps, value, float(reward), bool(done or truncated))
            obs = next_obs
            ep_ret += float(reward)
        with torch.no_grad():
            if done or truncated:
                last_v = 0.0
            else:
                _a, _lp, last_v = agent.act(obs, deterministic=True)
        agent.update(last_value=last_v)
        recent_r.append(ep_ret)
        recent_steps.append(int(env.time_step))
        recent_surv.append(1.0 if last_info.get("survived", False) else 0.0)
        if ep % LOG_EVERY == 0:
            avg_r = float(np.mean(recent_r[-LOG_EVERY:]))
            avg_s = float(np.mean(recent_steps[-LOG_EVERY:]))
            avg_surv = float(np.mean(recent_surv[-LOG_EVERY:]))
            print(
                f"[Episode {ep:03d}/{n_episodes}] "
                f"Reward={avg_r:8.2f}  Steps={avg_s:5.1f}  Survival Rate={avg_surv:5.2f}"
            )

    print("[GAT-PPO] greedy evaluation ...")
    metrics = evaluate_policy(env, agent, n_episodes=EVAL_EPISODES)
    print(
        f"[Eval] Reward={metrics['reward']:.2f}  Steps={metrics['steps']:.1f}  "
        f"Survival Rate={metrics['survival']:.2f}  Clear Rate={metrics['clear_rate']:.2f}"
    )
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": agent.net.state_dict(),
            "max_actions": env.max_actions,
            "n_agents": env.n_agents,
            "eval": metrics,
        },
        CHECKPOINT_PATH,
    )
    print(f"[GAT-PPO] saved {CHECKPOINT_PATH}")
    return CHECKPOINT_PATH


def main() -> None:
    """命令行入口。"""
    train_curriculum()


if __name__ == "__main__":
    main()

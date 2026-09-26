#!/usr/bin/env python3
"""GAT-PPO Office 推理演示：加载 gat_ppo_office.pt 并逐步打印消防员位置。

运行::

    PYTHONPATH=. python scripts/eval_rl_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.environment.layouts import get_office_layout
from src.rl.env_wrapper import GraphEvacuationEnv
from src.rl.gat_net import GATActorCritic
from src.rl.ppo_agent import PPOAgent, select_device

CHECKPOINT_PATH = ROOT / "src" / "rl" / "checkpoints" / "gat_ppo_office.pt"
MAX_DEMO_STEPS: int = 60


def run() -> None:
    """载入权重，在 Figure 1 办公室上跑一步推理循环。"""
    print("\n" + "=" * 65)
    print("[Step 1/3] GAT-PPO inference demo (Office, 2 firefighters)")
    print("=" * 65)

    if not CHECKPOINT_PATH.is_file():
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        print("Train first: PYTHONPATH=. python src/rl/train.py")
        sys.exit(1)

    print("[Step 2/3] Loading layout, env, and checkpoint")
    device = select_device()
    graph = get_office_layout()
    env = GraphEvacuationEnv(graph, n_agents=2, max_steps=MAX_DEMO_STEPS, enable_hazard=False)
    payload = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    max_actions = int(payload.get("max_actions", env.max_actions))
    net = GATActorCritic(in_dim=6, hidden_dim=64, heads=4, max_actions=max_actions)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    net.load_state_dict(state)
    agent = PPOAgent(net, device=device)
    net.eval()
    print(f"  device={device}  checkpoint={CHECKPOINT_PATH.name}")

    obs, info = env.reset(seed=0)
    trajectories: dict[int, list[str]] = {0: [env.agent_nodes[0]], 1: [env.agent_nodes[1]]}

    print("-" * 65)
    done = False
    truncated = False
    step = 0
    while step < MAX_DEMO_STEPS and not (done or truncated):
        step += 1
        actions, _logp, _value = agent.act(obs, deterministic=False)
        obs, _reward, done, truncated, info = env.step(actions)
        for agent_id in range(2):
            trajectories[agent_id].append(str(env.agent_nodes[agent_id]))
        if step <= 5 or step % 5 == 0:
            print(
                f"Step {step:02d} | F1: {trajectories[0][-1]:<6} | "
                f"F2: {trajectories[1][-1]:<6} | cleared={info.get('n_cleared', 0)}"
            )
        if info.get("all_clear", False):
            print("-" * 65)
            print(f"All rooms tagged at step {step}.")
            break

    print("=" * 65)
    print("[Step 3/3] Trajectory summary")
    print(f"  F1: {' -> '.join(trajectories[0])}")
    print(f"  F2: {' -> '.join(trajectories[1])}")
    print(
        f"  steps={step}  all_clear={bool(info.get('all_clear', False))}  "
        f"survived={bool(info.get('survived', True))}  n_cleared={info.get('n_cleared', 0)}"
    )
    print("=" * 65)


if __name__ == "__main__":
    run()

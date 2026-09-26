"""轻量 PPO：MLP[64,64]、clip ε=0.2、GAE λ=0.95，优先 MPS。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.distributions import Categorical

from src.rl.env_wrapper import OBS_DIM, N_ACTIONS


def select_device() -> torch.device:
    """默认 CPU。``RL_DEVICE=mps`` 且后端可用时走 Apple Silicon。

    部分 PyTorch/MPS 组合在首次 ``.to(mps)`` 会挂起，故不默认启用。
    """
    wanted = os.environ.get("RL_DEVICE", "cpu").strip().lower()
    if wanted == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ActorCritic(nn.Module):
    """共享隐层的离散 Actor-Critic。"""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (logits, value)。``obs`` 形状 ``(B, obs_dim)``。"""
        feat = self.backbone(obs)
        return self.actor(feat), self.critic(feat).squeeze(-1)


@dataclass
class PPOConfig:
    """PPO 超参。"""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3.0e-4
    epochs: int = 4
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad: float = 0.5


class PPOAgent:
    """单环境 rollout 上的 clipped PPO。"""

    def __init__(self, config: PPOConfig | None = None, device: torch.device | None = None) -> None:
        self.config = config if config is not None else PPOConfig()
        self.device = device if device is not None else select_device()
        self.net = ActorCritic().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)

    def act(self, obs: NDArray[np.float32]) -> tuple[int, float, float]:
        """采样动作，返回 (action, log_prob, value)。"""
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    def greedy(self, obs: NDArray[np.float32]) -> int:
        """评估用贪心动作。"""
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _value = self.net(tensor)
        return int(torch.argmax(logits, dim=-1).item())

    def update(
        self,
        obs: NDArray[np.float32],
        actions: NDArray[np.int64],
        old_logp: NDArray[np.float32],
        rewards: NDArray[np.float32],
        dones: NDArray[np.float32],
        values: NDArray[np.float32],
    ) -> dict[str, float]:
        """GAE + clipped surrogate。数组长度均为 T。"""
        adv, ret = _gae(
            rewards,
            values,
            dones,
            self.config.gamma,
            self.config.gae_lambda,
        )
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1.0e-8)
        last_loss = 0.0
        for _ in range(self.config.epochs):
            logits, values_p = self.net(obs_t)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act_t)
            ratio = torch.exp(logp - old_t)
            clipped = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
            policy_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()
            value_loss = torch.mean((values_p - ret_t) ** 2)
            entropy = dist.entropy().mean()
            loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.config.max_grad)
            self.opt.step()
            last_loss = float(loss.item())
        return {"loss": last_loss}

    def save(self, path: Path) -> None:
        """写 state_dict。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.net.state_dict(), "config": self.config.__dict__}, path)

    def load(self, path: Path) -> None:
        """读 state_dict。"""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(payload["net"])


def _gae(
    rewards: NDArray[np.float32],
    values: NDArray[np.float32],
    dones: NDArray[np.float32],
    gamma: float,
    lam: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """广义优势估计。末端 value 视为 0。"""
    t_len = rewards.shape[0]
    adv = np.zeros(t_len, dtype=np.float32)
    next_v = 0.0
    next_adv = 0.0
    for i in range(t_len - 1, -1, -1):
        mask = 1.0 - float(dones[i])
        delta = float(rewards[i]) + gamma * next_v * mask - float(values[i])
        next_adv = delta + gamma * lam * mask * next_adv
        adv[i] = next_adv
        next_v = float(values[i])
    ret = adv + values
    return adv, ret.astype(np.float32)

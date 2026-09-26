"""参数共享多智能体 PPO：GAE、裁剪目标、熵正则。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.optim import Adam

from src.rl.env_wrapper import ObsDict
from src.rl.gat_net import GATActorCritic


def select_device() -> torch.device:
    """优先 Apple Silicon MPS，否则 CPU。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class RolloutBuffer:
    """单回合轨迹缓存（按时间步存储图，智能体维在动作上展开）。"""

    xs: list[NDArray[np.float32]] = field(default_factory=list)
    edges: list[NDArray[np.int64]] = field(default_factory=list)
    agent_indices: list[NDArray[np.int64]] = field(default_factory=list)
    masks: list[NDArray[np.bool_]] = field(default_factory=list)
    actions: list[NDArray[np.int64]] = field(default_factory=list)
    logps: list[NDArray[np.float32]] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)

    def clear(self) -> None:
        """清空。"""
        self.xs.clear()
        self.edges.clear()
        self.agent_indices.clear()
        self.masks.clear()
        self.actions.clear()
        self.logps.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOAgent:
    """GAE-λ PPO，全体消防员共享 ``GATActorCritic`` 参数。

    Parameters
    ----------
    net :
        图注意力 Actor-Critic。
    gamma, gae_lambda :
        折扣 gamma=0.99 与 GAE lambda=0.95。
    clip_eps :
        裁剪比 epsilon=0.2。
    entropy_coef :
        熵奖励 beta=0.01。
    """

    def __init__(
        self,
        net: GATActorCritic,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        lr: float = 3.0e-4,
        update_epochs: int = 4,
        device: torch.device | None = None,
    ) -> None:
        self.device: torch.device = device if device is not None else select_device()
        self.net: GATActorCritic = net.to(self.device)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.update_epochs = int(update_epochs)
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def _tensors(self, obs: ObsDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(obs["x"], dtype=torch.float32, device=self.device)
        edge = torch.as_tensor(obs["edge_index"], dtype=torch.long, device=self.device)
        agents = torch.as_tensor(obs["agent_indices"], dtype=torch.long, device=self.device)
        mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool, device=self.device)
        return x, edge, agents, mask

    @torch.no_grad()
    def act(
        self,
        obs: ObsDict,
        *,
        deterministic: bool = False,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32], float]:
        """采样或贪心动作。返回 ``(actions [A], logp [A], V(s))``。"""
        self.net.eval()
        x, edge, agents, mask = self._tensors(obs)
        dist, value = self.net(x, edge, agents, mask)
        if deterministic:
            actions = torch.argmax(dist.probs, dim=-1)
        else:
            actions = dist.sample()
        logp = dist.log_prob(actions)
        v = float(value.reshape(-1)[0].cpu().item())
        return (
            actions.detach().cpu().numpy().astype(np.int64),
            logp.detach().cpu().numpy().astype(np.float32),
            v,
        )

    def remember(
        self,
        obs: ObsDict,
        actions: NDArray[np.int64],
        logps: NDArray[np.float32],
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        """写入一步（全体智能体同步动作，共享 V(s) 与团队奖励）。"""
        self.buffer.xs.append(np.asarray(obs["x"], dtype=np.float32))
        self.buffer.edges.append(np.asarray(obs["edge_index"], dtype=np.int64))
        self.buffer.agent_indices.append(np.asarray(obs["agent_indices"], dtype=np.int64))
        self.buffer.masks.append(np.asarray(obs["action_mask"], dtype=np.bool_))
        self.buffer.actions.append(np.asarray(actions, dtype=np.int64))
        self.buffer.logps.append(np.asarray(logps, dtype=np.float32))
        self.buffer.values.append(float(value))
        self.buffer.rewards.append(float(reward))
        self.buffer.dones.append(1.0 if done else 0.0)

    def _gae(self, last_value: float) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE-λ 优势与回报，形状 ``[T]``（逐步团队奖励）。"""
        t_len = len(self.buffer.rewards)
        adv = np.zeros(t_len, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(t_len)):
            next_v = last_value if t == t_len - 1 else self.buffer.values[t + 1]
            next_nonterminal = 1.0 - self.buffer.dones[t]
            delta = (
                self.buffer.rewards[t]
                + self.gamma * next_v * next_nonterminal
                - self.buffer.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + np.asarray(self.buffer.values, dtype=np.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1.0e-8)
        return adv_t, ret_t

    def update(self, last_value: float = 0.0) -> dict[str, float]:
        """对缓存轨迹做 ``update_epochs`` 轮 PPO 更新。"""
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        self.net.train()
        advantages, returns = self._gae(last_value)
        t_len = len(self.buffer.rewards)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0
        for _ in range(self.update_epochs):
            policy_loss_acc = torch.zeros((), device=self.device)
            value_loss_acc = torch.zeros((), device=self.device)
            entropy_acc = torch.zeros((), device=self.device)
            for t in range(t_len):
                x = torch.as_tensor(self.buffer.xs[t], dtype=torch.float32, device=self.device)
                edge = torch.as_tensor(self.buffer.edges[t], dtype=torch.long, device=self.device)
                agents = torch.as_tensor(
                    self.buffer.agent_indices[t], dtype=torch.long, device=self.device
                )
                mask = torch.as_tensor(self.buffer.masks[t], dtype=torch.bool, device=self.device)
                actions = torch.as_tensor(
                    self.buffer.actions[t], dtype=torch.long, device=self.device
                )
                old_logp = torch.as_tensor(
                    self.buffer.logps[t], dtype=torch.float32, device=self.device
                )
                dist, value = self.net(x, edge, agents, mask)
                logp = dist.log_prob(actions)
                ratio = torch.exp(logp - old_logp)
                adv = advantages[t].expand_as(ratio)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss_acc = policy_loss_acc + (-torch.min(surr1, surr2).mean())
                v_pred = value.reshape(-1)[0]
                value_loss_acc = value_loss_acc + 0.5 * (v_pred - returns[t]).pow(2)
                entropy_acc = entropy_acc + dist.entropy().mean()
            denom = float(t_len)
            loss = (
                policy_loss_acc / denom
                + self.value_coef * value_loss_acc / denom
                - self.entropy_coef * entropy_acc / denom
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()
            n_updates += 1
            metrics["policy_loss"] += float((policy_loss_acc / denom).detach().cpu())
            metrics["value_loss"] += float((value_loss_acc / denom).detach().cpu())
            metrics["entropy"] += float((entropy_acc / denom).detach().cpu())
        self.buffer.clear()
        if n_updates > 0:
            metrics = {k: v / n_updates for k, v in metrics.items()}
        return metrics

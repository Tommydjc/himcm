"""图注意力 Actor-Critic：两层 GATConv + 残差 LayerNorm。

Actor 对每个智能体当前节点做 Masked Softmax；
Critic 经全局注意力池化输出 V(s)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch_geometric.nn import GATConv
from torch_geometric.nn.aggr import AttentionalAggregation


class GATActorCritic(nn.Module):
    """参数共享的图注意力 Actor-Critic。

    Parameters
    ----------
        in_dim :
        节点特征维，默认 6。
    hidden_dim :
        隐层宽度，默认 64。
    heads :
        GAT 头数，默认 4；``concat=True`` 时每头通道为 ``hidden_dim // heads``。
    max_actions :
        动作槽位数 = 1（留守）+ 最大度。
    """

    def __init__(
        self,
        in_dim: int = 6,
        hidden_dim: int = 64,
        heads: int = 4,
        max_actions: int = 8,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        per_head = hidden_dim // heads
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_actions = int(max_actions)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, per_head, heads=heads, concat=True, dropout=0.1)
        self.gat2 = GATConv(hidden_dim, per_head, heads=heads, concat=True, dropout=0.1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, max_actions)
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pool = AttentionalAggregation(gate_nn)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """节点嵌入 ``[N, hidden_dim]``，含残差与 LayerNorm。"""
        h0 = self.in_proj(x)
        h1 = self.gat1(h0, edge_index)
        h1 = F.elu(self.ln1(h1 + h0))
        h2 = self.gat2(h1, edge_index)
        h2 = F.elu(self.ln2(h2 + h1))
        return h2

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        agent_indices: torch.Tensor,
        action_mask: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[Categorical, torch.Tensor]:
        """前向：Masked Categorical 与全局状态价值 V(s)。

        Parameters
        ----------
        x :
            ``[N, D_in]`` 节点特征。
        edge_index :
            ``[2, E]`` COO 边。
        agent_indices :
            ``[A]`` 智能体所在节点下标。
        action_mask :
            ``[A, max_actions]`` 布尔合法动作。
        batch :
            可选 ``[N]`` PyG batch 向量；单图时全 0。
        """
        h = self.encode(x, edge_index)
        n_nodes = x.size(0)
        if batch is None:
            batch = x.new_zeros(n_nodes, dtype=torch.long)
        graph_emb = self.pool(h, batch)
        value = self.critic_head(graph_emb).squeeze(-1)
        logits = self.actor_head(h[agent_indices])
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~action_mask.bool(), neg_inf)
        dist = Categorical(logits=logits)
        return dist, value

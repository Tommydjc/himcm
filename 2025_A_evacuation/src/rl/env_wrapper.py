"""多智能体图状态环境：建筑扫荡（Gymnasium API）。

本模块只依赖 ``layouts`` 与 ``hazards``，不导入 GAT/PPO。
节点特征维度 Din=6：
着火、烟雾、归一化脆弱人数、已清空、出口指示、本节点消防员人数。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from src.environment.hazards import HazardModel
from src.environment.layouts import exit_nodes, get_office_layout, searchable_nodes

NodeId = str
ObsDict = dict[str, NDArray[np.float32] | NDArray[np.int64] | NDArray[np.bool_]]

R_CLEAR: float = 10.0
R_STEP: float = -0.5
R_HAZARD: float = -15.0
R_REDUNDANT: float = -2.0
R_ALL_CLEAR: float = 50.0
FEATURE_DIM: int = 6


class GraphEvacuationEnv(gym.Env):
    """图上的同步多智能体扫荡环境。

    Parameters
    ----------
    graph :
        ``get_*_layout`` 图。``None`` 时使用 Figure 1 办公室。
    n_agents :
        消防员人数，标量。
    max_steps :
        回合最大步数。
    enable_hazard :
        是否启用 ``HazardModel`` 随机起火。
    hazard_p0 :
        基础点燃率 p0。
    seed :
        环境 RNG 种子。
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        n_agents: int = 2,
        max_steps: int = 100,
        enable_hazard: bool = False,
        hazard_p0: float = 0.05,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        self.graph: nx.Graph = graph if graph is not None else get_office_layout()
        self.n_agents: int = int(n_agents)
        self.max_steps: int = int(max_steps)
        self.enable_hazard: bool = bool(enable_hazard)
        self.hazard_p0: float = float(hazard_p0)
        self.node_ids: tuple[NodeId, ...] = tuple(str(n) for n in self.graph.nodes)
        self.n_nodes: int = len(self.node_ids)
        self.node_index: dict[NodeId, int] = {n: i for i, n in enumerate(self.node_ids)}
        self.searchable: tuple[NodeId, ...] = tuple(searchable_nodes(self.graph))
        self.exit_set: set[NodeId] = set(str(n) for n in exit_nodes(self.graph))
        max_deg = max(int(self.graph.degree(n)) for n in self.graph.nodes)
        self.max_actions: int = max_deg + 1
        self._max_vuln: float = max(
            (
                float(self.graph.nodes[n].get("vulnerable_weight", 0.0))
                for n in self.node_ids
            ),
            default=1.0,
        )
        if self._max_vuln <= 0.0:
            self._max_vuln = 1.0
        self._edge_index: NDArray[np.int64] = self._build_edge_index()
        self._neighbors: tuple[tuple[NodeId, ...], ...] = tuple(
            tuple(sorted(str(v) for v in self.graph.neighbors(u))) for u in self.node_ids
        )
        self.action_space = spaces.MultiDiscrete(
            [self.max_actions] * self.n_agents, dtype=np.int64
        )
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=0.0, high=np.inf, shape=(self.n_nodes, FEATURE_DIM), dtype=np.float32
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=self.n_nodes,
                    shape=self._edge_index.shape,
                    dtype=np.int64,
                ),
                "agent_indices": spaces.Box(
                    low=0, high=self.n_nodes, shape=(self.n_agents,), dtype=np.int64
                ),
                "action_mask": spaces.MultiBinary((self.n_agents, self.max_actions)),
            }
        )
        self.hazard: HazardModel | None = None
        self.agent_nodes: list[NodeId] = []
        self.cleared: set[NodeId] = set()
        self.time_step: int = 0
        self.survived: bool = True
        self._rng = np.random.default_rng(seed)
        self._init_starts()

    def _init_starts(self) -> None:
        exits = [str(n) for n in exit_nodes(self.graph)]
        extras = [n for n in self.node_ids if n not in exits]
        pool = exits + extras
        if len(pool) < self.n_agents:
            raise ValueError("not enough start nodes")
        self._start_nodes: tuple[NodeId, ...] = tuple(pool[: self.n_agents])

    def _build_edge_index(self) -> NDArray[np.int64]:
        src: list[int] = []
        dst: list[int] = []
        for u, v in self.graph.edges:
            iu, iv = self.node_index[str(u)], self.node_index[str(v)]
            src.extend([iu, iv])
            dst.extend([iv, iu])
        if not src:
            return np.zeros((2, 0), dtype=np.int64)
        return np.asarray([src, dst], dtype=np.int64)

    def _vuln(self, node: NodeId) -> float:
        weight = float(self.graph.nodes[node].get("vulnerable_weight", 0.0))
        if weight <= 0.0 and self.graph.nodes[node].get("search_required"):
            weight = 1.0
        return weight

    def _hazard_level(self, node: NodeId) -> float:
        if self.hazard is None:
            return 0.0
        fire = 1.0 if self.hazard.is_on_fire(node) else 0.0
        return float(self.hazard.smoke_density(node) + fire)

    def _action_mask(self) -> NDArray[np.bool_]:
        mask = np.zeros((self.n_agents, self.max_actions), dtype=np.bool_)
        for i, node in enumerate(self.agent_nodes):
            mask[i, 0] = True
            idx = self.node_index[node]
            deg = len(self._neighbors[idx])
            if deg > 0:
                mask[i, 1 : deg + 1] = True
        return mask

    def _features(self) -> NDArray[np.float32]:
        counts = np.zeros(self.n_nodes, dtype=np.float32)
        for node in self.agent_nodes:
            counts[self.node_index[node]] += 1.0
        x = np.zeros((self.n_nodes, FEATURE_DIM), dtype=np.float32)
        for i, node in enumerate(self.node_ids):
            on_fire = 0.0
            smoke = 0.0
            if self.hazard is not None:
                on_fire = 1.0 if self.hazard.is_on_fire(node) else 0.0
                smoke = float(self.hazard.smoke_density(node))
            x[i, 0] = on_fire
            x[i, 1] = smoke
            x[i, 2] = self._vuln(node) / self._max_vuln
            x[i, 3] = 1.0 if node in self.cleared else 0.0
            x[i, 4] = 1.0 if node in self.exit_set else 0.0
            x[i, 5] = counts[i]
        return x

    def _observation(self) -> ObsDict:
        agent_idx = np.asarray(
            [self.node_index[n] for n in self.agent_nodes], dtype=np.int64
        )
        return {
            "x": self._features(),
            "edge_index": self._edge_index.copy(),
            "agent_indices": agent_idx,
            "action_mask": self._action_mask(),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsDict, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agent_nodes = list(self._start_nodes)
        self.cleared = set()
        self.time_step = 0
        self.survived = True
        if self.enable_hazard:
            self.hazard = HazardModel(self.graph, p0=self.hazard_p0, seed=int(self._rng.integers(0, 10**9)))
            self.hazard.reset()
            rooms = [n for n in self.searchable]
            if rooms:
                self.hazard.ignite(str(rooms[int(self._rng.integers(0, len(rooms)))]))
        else:
            self.hazard = None
        return self._observation(), {"n_cleared": 0, "survived": True}

    def _decode_move(self, agent_i: int, action: int) -> NodeId:
        node = self.agent_nodes[agent_i]
        if action <= 0:
            return node
        nbrs = self._neighbors[self.node_index[node]]
        j = int(action) - 1
        if j < 0 or j >= len(nbrs):
            return node
        return nbrs[j]

    def step(
        self, action: NDArray[np.int64] | list[int]
    ) -> tuple[ObsDict, float, bool, bool, dict[str, Any]]:
        acts = np.asarray(action, dtype=np.int64).reshape(self.n_agents)
        reward = R_STEP
        newly_cleared: list[NodeId] = []

        for i in range(self.n_agents):
            dest = self._decode_move(i, int(acts[i]))
            stay = dest == self.agent_nodes[i]
            searchable = dest in self.searchable
            already = dest in self.cleared
            on_fire = self.hazard is not None and self.hazard.is_on_fire(dest)
            if searchable and already and (not stay) and (not on_fire):
                reward += R_REDUNDANT
            self.agent_nodes[i] = dest
            if stay and searchable and dest not in self.cleared and not on_fire:
                self.cleared.add(dest)
                newly_cleared.append(dest)
                reward += R_CLEAR * self._vuln(dest)
            reward += R_HAZARD * self._hazard_level(dest)
            if on_fire:
                self.survived = False

        if self.hazard is not None:
            self.hazard.step(dt=1.0)

        self.time_step += 1
        all_clear = all(r in self.cleared for r in self.searchable)
        if all_clear:
            reward += R_ALL_CLEAR
        terminated = all_clear or (not self.survived)
        truncated = self.time_step >= self.max_steps and not terminated
        info: dict[str, Any] = {
            "n_cleared": len(self.cleared),
            "all_clear": all_clear,
            "survived": self.survived,
            "t_clear": self.time_step if all_clear else None,
            "newly_cleared": newly_cleared,
        }
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

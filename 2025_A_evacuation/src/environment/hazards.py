"""动态火情与烟雾扩散（与具体建筑布局解耦）。

只依赖 ``networkx.Graph`` 的邻接关系与边长 ``length``（m），不导入 ``layouts``。

明火按邻居伯努利传播

.. math::

    P_{\\mathrm{spread}} = 1 - (1 - p_0)^{k}

其中 ``k`` 为着火邻居数，``p_0`` 为单邻居基础点燃率。

浓烟前锋速度为明火的 ``smoke_speed_ratio`` 倍（默认 1.5），
进入含烟节点后消防员速度乘子与视野同步衰减。
"""

from __future__ import annotations

from typing import Mapping

import networkx as nx
import numpy as np

NodeId = str

SMOKE_FIRE_SPEED_RATIO: float = 1.5


def _edge_length(graph: nx.Graph, u: NodeId, v: NodeId) -> float:
    """读取边长（m）；缺省时回退到 ``pos`` 欧氏距离或 1。"""
    data = graph.edges[u, v]
    if "length" in data:
        return float(data["length"])
    pos_u = graph.nodes[u].get("pos")
    pos_v = graph.nodes[v].get("pos")
    if pos_u is not None and pos_v is not None:
        x1, y1 = float(pos_u[0]), float(pos_u[1])
        x2, y2 = float(pos_v[0]), float(pos_v[1])
        return float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
    return 1.0


class HazardModel:
    """图上的离散时间火/烟耦合过程。

    Parameters
    ----------
    graph :
        任意无向建筑图。边权 ``length`` 单位 m。
    p0 :
        单着火邻居的基础点燃概率 ``p_0 \\in (0, 1]``，标量。
    v_fire :
        明火沿边的前锋标称速度（m/s），标量。浓烟速度为
        ``v_smoke = smoke_speed_ratio * v_fire``。
    smoke_speed_ratio :
        烟/火速度比，默认 ``SMOKE_FIRE_SPEED_RATIO = 1.5``。
    speed_decay_beta :
        速度衰减系数 ``\\beta``。满烟时乘子为 ``max(v_min, 1 - \\beta)``。
        默认 0.4，对应 Requirement 2 走廊浓烟降速 40%。
    visibility_decay_gamma :
        视野衰减系数 ``\\gamma``。满烟时视野为 ``max(vis_min, 1 - \\gamma)``。
    v_min, vis_min :
        速度乘子与视野的下限，避免数值归零。
    seed :
        ``numpy`` 随机种子；``None`` 表示不可复现。
    """

    def __init__(
        self,
        graph: nx.Graph,
        *,
        p0: float = 0.10,
        v_fire: float = 0.50,
        smoke_speed_ratio: float = SMOKE_FIRE_SPEED_RATIO,
        speed_decay_beta: float = 0.40,
        visibility_decay_gamma: float = 0.80,
        v_min: float = 0.15,
        vis_min: float = 0.10,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < p0 <= 1.0:
            raise ValueError(f"p0 must lie in (0, 1], got {p0}")
        if v_fire <= 0.0:
            raise ValueError("v_fire must be positive")
        if smoke_speed_ratio <= 0.0:
            raise ValueError("smoke_speed_ratio must be positive")

        self.graph: nx.Graph = graph
        self.p0: float = float(p0)
        self.v_fire: float = float(v_fire)
        self.smoke_speed_ratio: float = float(smoke_speed_ratio)
        self.speed_decay_beta: float = float(speed_decay_beta)
        self.visibility_decay_gamma: float = float(visibility_decay_gamma)
        self.v_min: float = float(v_min)
        self.vis_min: float = float(vis_min)
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._nodes: tuple[NodeId, ...] = tuple(str(n) for n in graph.nodes)
        self._fire: dict[NodeId, bool] = {n: False for n in self._nodes}
        self._smoke: dict[NodeId, float] = {n: 0.0 for n in self._nodes}
        self.time: float = 0.0
        self._weighted: nx.Graph = self._length_weighted_copy(graph)

    @staticmethod
    def _length_weighted_copy(graph: nx.Graph) -> nx.Graph:
        """拷贝边权为 Dijkstra 所需的 ``weight=length`` 图。"""
        weighted = nx.Graph()
        weighted.add_nodes_from(graph.nodes)
        for u, v in graph.edges:
            length = _edge_length(graph, str(u), str(v))
            weighted.add_edge(u, v, length=length)
        return weighted

    def reset(self) -> None:
        """清空火/烟并归零时钟（保留 ``p0`` 等超参）。"""
        for node in self._nodes:
            self._fire[node] = False
            self._smoke[node] = 0.0
        self.time = 0.0

    def ignite(self, node: NodeId, *, smoke_density: float = 1.0) -> None:
        """在 ``node`` 点燃明火，并写入初始烟浓度。"""
        if node not in self._fire:
            raise KeyError(f"unknown node {node!r}")
        self._fire[node] = True
        self._smoke[node] = float(min(1.0, max(self._smoke[node], smoke_density)))

    def is_on_fire(self, node: NodeId) -> bool:
        """节点是否已着火。"""
        return self._fire[node]

    def smoke_density(self, node: NodeId) -> float:
        """烟浓度 ``\\sigma \\in [0, 1]``。"""
        return float(self._smoke[node])

    def burning_neighbor_count(self, node: NodeId) -> int:
        """着火邻居数 ``k``（图邻接，无向）。"""
        k = 0
        for nbr in self.graph.neighbors(node):
            if self._fire[str(nbr)]:
                k += 1
        return k

    def spread_probability(self, node: NodeId) -> float:
        """未着火节点的点燃概率 :math:`P_{spread}=1-(1-p_0)^k`。

        已着火或 ``k=0`` 时返回 0。
        """
        if self._fire[node]:
            return 0.0
        k = self.burning_neighbor_count(node)
        if k <= 0:
            return 0.0
        return float(1.0 - (1.0 - self.p0) ** k)

    def firefighter_speed_multiplier(self, node: NodeId) -> float:
        """经过该节点的速度乘子，取值 ``[v_min, 1]``；明火节点为 0（禁行）。

        .. math::

            m_v(u) = \\max(v_{\\min},\\, 1 - \\beta \\sigma(u))
        """
        if self._fire[node]:
            return 0.0
        sigma = self._smoke[node]
        return float(max(self.v_min, 1.0 - self.speed_decay_beta * sigma))

    def visibility(self, node: NodeId) -> float:
        """视野系数 ``[vis_min, 1]``；明火处视为 0。

        .. math::

            m_{\\mathrm{vis}}(u) = \\max(\\mathrm{vis}_{\\min},\\, 1 - \\gamma \\sigma(u))
        """
        if self._fire[node]:
            return 0.0
        sigma = self._smoke[node]
        return float(max(self.vis_min, 1.0 - self.visibility_decay_gamma * sigma))

    def is_impassable(self, node: NodeId) -> bool:
        """明火节点对消防员不可通行。"""
        return self._fire[node]

    def v_smoke(self) -> float:
        """浓烟前锋速度 ``v_smoke = smoke_speed_ratio * v_fire``（m/s）。"""
        return self.smoke_speed_ratio * self.v_fire

    def step(self, dt: float = 1.0) -> None:
        """推进 ``dt`` 秒：先更新烟前锋，再按 :math:`P_{spread}` 点燃。

        烟从所有着火或已有烟的节点出发，沿 ``length`` 做多源最短路，
        截断距离为 ``v_smoke * dt``；到达点烟浓度取

        ``max(旧值, 1 - dist / (v_smoke * dt))``，着火点强制为 1。
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self._advance_smoke(dt)
        self._advance_fire()
        self.time += float(dt)

    def _advance_smoke(self, dt: float) -> None:
        sources = [n for n in self._nodes if self._fire[n] or self._smoke[n] > 0.0]
        if not sources:
            return
        budget = self.v_smoke() * dt
        if budget <= 0.0:
            return
        reached: dict[NodeId, float] = nx.multi_source_dijkstra_path_length(
            self._weighted,
            sources,
            cutoff=budget,
            weight="length",
        )
        for node, dist in reached.items():
            node_id = str(node)
            if self._fire[node_id]:
                self._smoke[node_id] = 1.0
                continue
            incoming = 1.0 - float(dist) / budget
            incoming = max(0.0, min(1.0, incoming))
            self._smoke[node_id] = max(self._smoke[node_id], incoming)

    def _advance_fire(self) -> None:
        ignited: list[NodeId] = []
        for node in self._nodes:
            p_spread = self.spread_probability(node)
            if p_spread <= 0.0:
                continue
            if float(self._rng.random()) < p_spread:
                ignited.append(node)
        for node in ignited:
            self._fire[node] = True
            self._smoke[node] = 1.0

    def snapshot(self) -> dict[str, Mapping[NodeId, float] | Mapping[NodeId, bool] | float]:
        """导出当前火/烟状态（供实验层写 CSV，本模块不写盘）。"""
        return {
            "time": self.time,
            "fire": dict(self._fire),
            "smoke": dict(self._smoke),
        }

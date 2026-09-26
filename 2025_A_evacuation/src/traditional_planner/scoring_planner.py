"""基于房间评分的多智能体搜救规划器（传统规划器层）。

每个仿真步长，空闲消防员对未清空房间 ``r`` 计算

.. math::

    \\mathrm{Score}(r) =
    - w_{\\mathrm{dist}} \\, d(a,r)
    - w_{\\mathrm{hazard}} \\, h(r)
    + w_{\\mathrm{priority}} \\, v(r)
    - w_{\\mathrm{redundancy}} \\, \\mathbf{1}_{\\mathrm{targeted}}
    - w_{\\mathrm{tabu}} \\, \\mathbf{1}[r \\in T_a]
    + w_{\\mathrm{mom}} \\, M(a,r)

其中 ``T_a`` 为长度 ``tabu_tenure=3`` 的房间禁忌表（不含走廊割点）；
若剩余未清房间全在 ``T_a`` 中则启动渴望准则、忽略禁忌。
``M(a,r)`` 取值 ``[0, 1]``，奖励同一走廊枢纽的兄弟房间及与行进方向一致的目标。
选路时禁止无替代方案以外的边反向，以消除环上震荡。

随后做确定性状态转移：``move -> sweep -> tag -> move``。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx

from src.environment.hazards import HazardModel
from src.environment.layouts import exit_nodes, searchable_nodes

NodeId = str
AgentPhase = Literal["idle", "move", "sweep", "tag"]
TickStatus = Literal["running", "cleared", "deadlock"]
_CIRCULATION_TYPES: frozenset[str] = frozenset({"hallway", "aisle", "stair", "exit"})


@dataclass(frozen=True)
class PlannerConfig:
    """评分规划器超参，符号与 Score(r) 公式一致。

    Parameters
    ----------
    n_agents :
        消防员人数。
    start_nodes :
        出生节点；``None`` 时按图中 exit 顺序截取。
    w_dist, w_hazard, w_priority, w_redundancy :
        评分权重（无量纲）。
    w_tabu, w_mom :
        禁忌罚与动量奖；``tabu_tenure`` 为短期记忆长度（默认 3）。
    v_hall, v_room :
        走廊 / 室内速度（m/s）。
    t_sweep, t_tag :
        扫荡确认 / 挂牌耗时（s）。
    dt :
        离散步长（s）。演示脚本取 15。
    """

    n_agents: int = 2
    start_nodes: tuple[NodeId, ...] | None = None
    w_dist: float = 1.0
    w_hazard: float = 5.0
    w_priority: float = 2.0
    w_redundancy: float = 80.0
    w_tabu: float = 50.0
    w_mom: float = 15.0
    tabu_tenure: int = 3
    v_hall: float = 1.5
    v_room: float = 0.8
    t_sweep: float = 20.0
    t_tag: float = 5.0
    dt: float = 1.0


@dataclass
class ClearanceResult:
    """一次联合搜救的清空记录。

    Attributes
    ----------
    t_clear :
        最后一间房完成挂牌的仿真时刻（s）。失败时为最后一步时刻。
    n_steps :
        离散步数（Steps）。墙钟近似 ``n_steps * dt``。
    dt :
        步长（s）。
    room_clear_times :
        房间 id -> 挂牌完成时刻（s）。
    success :
        是否在 ``max_steps`` 内清空全部 ``search_required`` 节点。
    deadlock :
        是否因不可达 / 无进展而提前终止。
    """

    t_clear: float
    n_steps: int
    dt: float
    room_clear_times: dict[NodeId, float]
    success: bool
    deadlock: bool = False
    message: str = ""


@dataclass
class _Agent:
    agent_id: str
    node: NodeId
    phase: AgentPhase = "idle"
    target: NodeId | None = None
    path: list[NodeId] = field(default_factory=list)
    edge_remaining: float = 0.0
    phase_remaining: float = 0.0
    visited: list[NodeId] = field(default_factory=list)
    tabu_rooms: deque[NodeId] = field(default_factory=deque)
    last_edge: tuple[NodeId, NodeId] | None = None


class ScoringPlanner:
    """贪婪评分 + 有限状态机的多智能体基线。

    Parameters
    ----------
    graph :
        ``get_*_layout`` 返回的无向图；边权 ``length`` 单位 m。
    hazard :
        可选 ``HazardModel``。``None`` 表示无火无烟，HazardLevel=0。
    start_nodes :
        各消防员出生点。``None`` 时按出口顺序取前 ``n_agents`` 个。
    n_agents :
        消防员人数，标量。
    w_dist, w_hazard, w_priority, w_redundancy :
        评分权重，与公式符号一一对应。
    v_hall, v_room :
        走廊/室内标称速度（m/s）。
    t_sweep, t_tag :
        扫荡确认与挂牌耗时（s）。
    dt :
        仿真步长（s）。
    """

    def __init__(
        self,
        graph: nx.Graph,
        hazard: HazardModel | None = None,
        *,
        start_nodes: tuple[NodeId, ...] | None = None,
        n_agents: int = 2,
        w_dist: float = 1.0,
        w_hazard: float = 5.0,
        w_priority: float = 2.0,
        w_redundancy: float = 80.0,
        w_tabu: float = 50.0,
        w_mom: float = 15.0,
        tabu_tenure: int = 3,
        v_hall: float = 1.5,
        v_room: float = 0.8,
        t_sweep: float = 20.0,
        t_tag: float = 5.0,
        dt: float = 1.0,
    ) -> None:
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if v_hall <= 0.0 or v_room <= 0.0:
            raise ValueError("speeds must be positive")
        if tabu_tenure < 1:
            raise ValueError("tabu_tenure must be >= 1")

        self.graph: nx.Graph = graph
        self.hazard: HazardModel | None = hazard
        self.w_dist: float = float(w_dist)
        self.w_hazard: float = float(w_hazard)
        self.w_priority: float = float(w_priority)
        self.w_redundancy: float = float(w_redundancy)
        self.w_tabu: float = float(w_tabu)
        self.w_mom: float = float(w_mom)
        self.tabu_tenure: int = int(tabu_tenure)
        self.v_hall: float = float(v_hall)
        self.v_room: float = float(v_room)
        self.t_sweep: float = float(t_sweep)
        self.t_tag: float = float(t_tag)
        self.dt: float = float(dt)
        self.rooms: tuple[NodeId, ...] = tuple(searchable_nodes(graph))
        exits = exit_nodes(graph)
        if start_nodes is None:
            if len(exits) < n_agents:
                raise ValueError("not enough exit nodes for n_agents")
            start_nodes = tuple(exits[:n_agents])
        if len(start_nodes) != n_agents:
            raise ValueError("start_nodes length must equal n_agents")
        self.start_nodes: tuple[NodeId, ...] = start_nodes
        self._agents: list[_Agent] = [
            _Agent(agent_id=f"F{i + 1}", node=start_nodes[i]) for i in range(n_agents)
        ]
        self.cleared: dict[NodeId, float] = {}
        self.time_s: float = 0.0
        self.n_steps: int = 0
        self._last_progress_step: int = 0
        self._reset_runtime()

    def vulnerable_count(self, room: NodeId) -> float:
        """VulnerableCount(r)：节点 ``vulnerable_weight``；办公室缺省按 1 人计。"""
        weight = float(self.graph.nodes[room].get("vulnerable_weight", 0.0))
        if weight > 0.0:
            return weight
        return 1.0 if self.graph.nodes[room].get("search_required") else 0.0

    def hazard_level(self, room: NodeId) -> float:
        """HazardLevel(r) = sigma(r) + 1_{fire}(r)，无灾害模型时为 0。"""
        if self.hazard is None:
            return 0.0
        fire_term = 1.0 if self.hazard.is_on_fire(room) else 0.0
        return float(self.hazard.smoke_density(room) + fire_term)

    def shortest_path_len(self, source: NodeId, target: NodeId) -> float | None:
        """ShortestPathLen：可通行子图上以 ``length`` 为权的最短路长（m）。"""
        if source == target:
            return 0.0
        walkable = self._walkable_graph()
        if source not in walkable or target not in walkable:
            return None
        try:
            return float(
                nx.shortest_path_length(walkable, source, target, weight="length")
            )
        except nx.NetworkXNoPath:
            return None

    def _node_pos(self, node: NodeId) -> tuple[float, float]:
        pos = self.graph.nodes[node]["pos"]
        return float(pos[0]), float(pos[1])

    def _circulation_hubs(self, node: NodeId) -> set[NodeId]:
        """节点自身及邻接中的走廊/通道/出口，用于识别同列兄弟房间。"""
        hubs: set[NodeId] = set()
        ntype = str(self.graph.nodes[node].get("node_type", ""))
        if ntype in _CIRCULATION_TYPES:
            hubs.add(node)
        for nbr in self.graph.neighbors(node):
            nb = str(nbr)
            if str(self.graph.nodes[nb].get("node_type", "")) in _CIRCULATION_TYPES:
                hubs.add(nb)
        return hubs

    def momentum_bonus(self, agent: _Agent, room: NodeId) -> float:
        """动量 M(a,r) 属于 [0, 1]：同枢纽兄弟房为 1，否则为行进方向余弦的正部。"""
        here_hubs = self._circulation_hubs(agent.node)
        room_hubs = self._circulation_hubs(room)
        if here_hubs & room_hubs:
            return 1.0
        if agent.tabu_rooms:
            last_cleared = agent.tabu_rooms[-1]
            if self._circulation_hubs(last_cleared) & room_hubs:
                return 1.0
        if agent.last_edge is None:
            return 0.0
        u, v = agent.last_edge
        x0, y0 = self._node_pos(u)
        x1, y1 = self._node_pos(v)
        hx, hy = x1 - x0, y1 - y0
        gx, gy = self._node_pos(room)
        ax, ay = self._node_pos(agent.node)
        dx, dy = gx - ax, gy - ay
        h_norm = (hx * hx + hy * hy) ** 0.5
        g_norm = (dx * dx + dy * dy) ** 0.5
        if h_norm < 1e-12 or g_norm < 1e-12:
            return 0.0
        cosine = (hx * dx + hy * dy) / (h_norm * g_norm)
        return float(max(0.0, cosine))

    def score(
        self,
        agent_node: NodeId,
        room: NodeId,
        *,
        already_targeted: bool,
        agent: _Agent | None = None,
        remaining: list[NodeId] | None = None,
    ) -> float | None:
        """计算 Score'(r)；房间不可达时返回 ``None``。"""
        dist = self.shortest_path_len(agent_node, room)
        if dist is None:
            return None
        redundancy = 1.0 if already_targeted else 0.0
        tabu_flag = 0.0
        mom = 0.0
        if agent is not None:
            mom = self.momentum_bonus(agent, room)
            if room in agent.tabu_rooms:
                leftover = remaining if remaining is not None else [
                    r for r in self.rooms if r not in self.cleared
                ]
                has_nontaboo = any(r not in agent.tabu_rooms for r in leftover)
                if has_nontaboo:
                    tabu_flag = 1.0
        return float(
            -self.w_dist * dist
            - self.w_hazard * self.hazard_level(room)
            + self.w_priority * self.vulnerable_count(room)
            - self.w_redundancy * redundancy
            - self.w_tabu * tabu_flag
            + self.w_mom * mom
        )

    @classmethod
    def from_config(
        cls,
        graph: nx.Graph,
        config: PlannerConfig,
        hazard: HazardModel | None = None,
    ) -> ScoringPlanner:
        """由 ``PlannerConfig`` 装配规划器。"""
        return cls(
            graph,
            hazard,
            start_nodes=config.start_nodes,
            n_agents=config.n_agents,
            w_dist=config.w_dist,
            w_hazard=config.w_hazard,
            w_priority=config.w_priority,
            w_redundancy=config.w_redundancy,
            w_tabu=config.w_tabu,
            w_mom=config.w_mom,
            tabu_tenure=config.tabu_tenure,
            v_hall=config.v_hall,
            v_room=config.v_room,
            t_sweep=config.t_sweep,
            t_tag=config.t_tag,
            dt=config.dt,
        )

    def _reset_runtime(self) -> None:
        """清空时钟、已清房间与轨迹，消防员回到出生点。"""
        self.cleared = {}
        self.time_s = 0.0
        self.n_steps = 0
        self._last_progress_step = 0
        self._agents = [
            _Agent(
                agent_id=f"F{i + 1}",
                node=self.start_nodes[i],
                visited=[self.start_nodes[i]],
                tabu_rooms=deque(maxlen=self.tabu_tenure),
                last_edge=None,
            )
            for i in range(len(self.start_nodes))
        ]

    def all_rooms_cleared(self) -> bool:
        """是否全部 ``search_required`` 节点已挂牌。"""
        return all(room in self.cleared for room in self.rooms)

    def agent_states(self) -> list[dict[str, str | None]]:
        """当前各消防员节点与相位（供演示日志）。"""
        rows: list[dict[str, str | None]] = []
        for agent in self._agents:
            rows.append(
                {
                    "agent_id": agent.agent_id,
                    "node": agent.node,
                    "phase": agent.phase,
                    "target": agent.target,
                }
            )
        return rows

    def trajectories(self) -> dict[str, list[NodeId]]:
        """消防员 id -> 按时间顺序的节点轨迹（含起点，相邻去重）。"""
        return {agent.agent_id: list(agent.visited) for agent in self._agents}

    def tick(self) -> TickStatus:
        """推进一个 ``dt``，供外部逐步日志循环调用。"""
        remaining = [r for r in self.rooms if r not in self.cleared]
        if not remaining:
            return "cleared"

        self.n_steps += 1
        self._replan_blocked_moves(self.cleared)
        self._assign_idle_agents(self.cleared)

        progressed = self._step_agents(self.dt, self.time_s, self.cleared)
        if self.hazard is not None:
            self.hazard.step(self.dt)
        self.time_s += self.dt

        busy = any(agent.phase != "idle" for agent in self._agents)
        newly_cleared = remaining != [r for r in self.rooms if r not in self.cleared]
        if progressed or busy or newly_cleared:
            self._last_progress_step = self.n_steps

        if self.all_rooms_cleared():
            return "cleared"
        if self._is_deadlocked(self.cleared) or (
            self.n_steps - self._last_progress_step
        ) >= 200:
            return "deadlock"
        return "running"

    def run(self, *, max_steps: int = 5000) -> ClearanceResult:
        """推进直到全楼清空、死锁或达到 ``max_steps``。"""
        self._reset_runtime()
        status: TickStatus = "running"
        for _ in range(max_steps):
            status = self.tick()
            if status == "cleared":
                last_t = max(self.cleared.values()) if self.cleared else self.time_s
                return ClearanceResult(
                    t_clear=last_t,
                    n_steps=self.n_steps,
                    dt=self.dt,
                    room_clear_times=dict(self.cleared),
                    success=True,
                    deadlock=False,
                    message="all searchable nodes tagged",
                )
            if status == "deadlock":
                return ClearanceResult(
                    t_clear=self.time_s,
                    n_steps=self.n_steps,
                    dt=self.dt,
                    room_clear_times=dict(self.cleared),
                    success=False,
                    deadlock=True,
                    message="deadlock: no feasible assignment or stalled progress",
                )

        return ClearanceResult(
            t_clear=self.time_s,
            n_steps=self.n_steps,
            dt=self.dt,
            room_clear_times=dict(self.cleared),
            success=False,
            deadlock=False,
            message=f"timeout after {max_steps} steps",
        )

    def _walkable_graph(self) -> nx.Graph:
        if self.hazard is None:
            return self.graph
        blocked = {n for n in self.graph.nodes if self.hazard.is_impassable(str(n))}
        keep = [n for n in self.graph.nodes if n not in blocked]
        return self.graph.subgraph(keep)

    def _edge_length(self, u: NodeId, v: NodeId) -> float:
        return float(self.graph.edges[u, v]["length"])

    def _edge_speed(self, u: NodeId, v: NodeId) -> float:
        kind = str(self.graph.edges[u, v].get("kind", "hallway"))
        base = self.v_room if kind == "room_access" else self.v_hall
        if self.hazard is None:
            return base
        mult_u = self.hazard.firefighter_speed_multiplier(str(u))
        mult_v = self.hazard.firefighter_speed_multiplier(str(v))
        if mult_u <= 0.0 or mult_v <= 0.0:
            return 0.0
        return base * min(mult_u, mult_v)

    def _sweep_duration(self, room: NodeId) -> float:
        if self.hazard is None:
            vis = 1.0
        else:
            vis = max(self.hazard.visibility(room), 1e-6)
        return self.t_sweep / vis

    def _targeted_rooms(self, cleared: dict[NodeId, float]) -> set[NodeId]:
        return {
            agent.target
            for agent in self._agents
            if agent.target is not None and agent.target not in cleared
        }

    def _assign_idle_agents(self, cleared: dict[NodeId, float]) -> None:
        remaining = [r for r in self.rooms if r not in cleared]
        for agent in self._agents:
            if agent.phase != "idle":
                continue
            targeted = self._targeted_rooms(cleared)
            best_room: NodeId | None = None
            best_score = float("-inf")
            for room in remaining:
                value = self.score(
                    agent.node,
                    room,
                    already_targeted=room in targeted,
                    agent=agent,
                    remaining=remaining,
                )
                if value is None:
                    continue
                if value > best_score or (
                    abs(value - best_score) < 1e-12 and best_room is not None and room < best_room
                ):
                    best_score = value
                    best_room = room
                elif best_room is None and value == best_score:
                    best_room = room
            if best_room is None:
                continue
            self._dispatch(agent, best_room)

    def _dispatch(self, agent: _Agent, room: NodeId) -> None:
        agent.target = room
        if agent.node == room:
            agent.path = []
            agent.edge_remaining = 0.0
            agent.phase = "sweep"
            agent.phase_remaining = self._sweep_duration(room)
            return
        walkable = self._walkable_graph()
        path = self._choose_path(agent, room, walkable)
        if path is None:
            agent.target = None
            agent.phase = "idle"
            return
        agent.path = path[1:]
        agent.phase = "move"
        agent.edge_remaining = self._edge_length(agent.node, agent.path[0])
        agent.phase_remaining = 0.0

    def _first_hop_reverses(self, agent: _Agent, path: list[NodeId]) -> bool:
        """路径第一步是否把上一跳边反向（环上震荡的典型动作）。"""
        if agent.last_edge is None or len(path) < 2:
            return False
        origin, current = agent.last_edge
        return path[0] == current and path[1] == origin

    def _choose_path(
        self,
        agent: _Agent,
        dest: NodeId,
        walkable: nx.Graph,
    ) -> list[NodeId] | None:
        """在最短路集合中优先不反向 ``last_edge``；无替代时允许回退（叶子渴望）。"""
        if agent.node not in walkable or dest not in walkable:
            return None
        if walkable.number_of_nodes() > 24:
            return self._choose_path_large(agent, dest, walkable)
        try:
            raw_paths = nx.all_shortest_paths(
                walkable, agent.node, dest, weight="length"
            )
            candidates = [[str(n) for n in p] for p in raw_paths]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        if not candidates:
            return None
        non_rev = [p for p in candidates if not self._first_hop_reverses(agent, p)]
        if non_rev:
            return non_rev[0]
        return candidates[0]

    def _choose_path_large(
        self,
        agent: _Agent,
        dest: NodeId,
        walkable: nx.Graph,
    ) -> list[NodeId] | None:
        """大图只用 Dijkstra；若第一步反向则删该边再求一次，失败则保留回退。"""
        try:
            path = [str(n) for n in nx.shortest_path(walkable, agent.node, dest, weight="length")]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        if not self._first_hop_reverses(agent, path) or agent.last_edge is None:
            return path
        origin, current = agent.last_edge
        alt_graph = walkable.copy()
        if alt_graph.has_edge(current, origin):
            alt_graph.remove_edge(current, origin)
        try:
            return [str(n) for n in nx.shortest_path(alt_graph, agent.node, dest, weight="length")]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return path

    def _replan_blocked_moves(self, cleared: dict[NodeId, float]) -> None:
        for agent in self._agents:
            if agent.phase != "move" or agent.target is None:
                continue
            if agent.target in cleared:
                agent.phase = "idle"
                agent.target = None
                agent.path = []
                continue
            if self.shortest_path_len(agent.node, agent.target) is None:
                agent.phase = "idle"
                agent.target = None
                agent.path = []
                continue
            walkable = self._walkable_graph()
            path = self._choose_path(agent, agent.target, walkable)
            if path is None:
                agent.phase = "idle"
                agent.target = None
                agent.path = []
                continue
            if path[1:] != agent.path:
                agent.path = path[1:]
                if agent.path:
                    agent.edge_remaining = self._edge_length(agent.node, agent.path[0])

    def _step_agents(
        self,
        dt: float,
        time_s: float,
        cleared: dict[NodeId, float],
    ) -> bool:
        progressed = False
        for agent in self._agents:
            leftover = dt
            while leftover > 1e-12:
                if agent.phase == "idle":
                    break
                before_node = agent.node
                before_phase = agent.phase
                leftover = self._advance_one_agent(agent, leftover, time_s + (dt - leftover), cleared)
                if agent.node != before_node or agent.phase != before_phase:
                    progressed = True
                if agent.phase == "idle" and leftover > 1e-12:
                    self._assign_idle_agents(cleared)
                    if agent.phase == "idle":
                        break
        return progressed

    def _advance_one_agent(
        self,
        agent: _Agent,
        leftover: float,
        action_start: float,
        cleared: dict[NodeId, float],
    ) -> float:
        if agent.phase == "move":
            return self._advance_move(agent, leftover)
        if agent.phase == "sweep":
            consume = min(leftover, agent.phase_remaining)
            agent.phase_remaining -= consume
            leftover -= consume
            if agent.phase_remaining <= 1e-12 and agent.target is not None:
                agent.phase = "tag"
                agent.phase_remaining = self.t_tag
            return leftover
        if agent.phase == "tag":
            consume = min(leftover, agent.phase_remaining)
            agent.phase_remaining -= consume
            leftover -= consume
            if agent.phase_remaining <= 1e-12 and agent.target is not None:
                finish_time = action_start + consume
                if agent.target not in cleared:
                    cleared[agent.target] = finish_time
                agent.tabu_rooms.append(agent.target)
                agent.target = None
                agent.phase = "idle"
                agent.path = []
            return leftover
        return leftover

    def _advance_move(self, agent: _Agent, leftover: float) -> float:
        if not agent.path:
            if agent.target is not None and agent.node == agent.target:
                agent.phase = "sweep"
                agent.phase_remaining = self._sweep_duration(agent.target)
            else:
                agent.phase = "idle"
            return leftover
        nxt = agent.path[0]
        speed = self._edge_speed(agent.node, nxt)
        if speed <= 0.0:
            return 0.0
        travel_budget = leftover * speed
        if travel_budget + 1e-12 < agent.edge_remaining:
            agent.edge_remaining -= travel_budget
            return 0.0
        leftover -= agent.edge_remaining / speed
        prev = agent.node
        agent.node = nxt
        agent.last_edge = (prev, nxt)
        if not agent.visited or agent.visited[-1] != agent.node:
            agent.visited.append(agent.node)
        agent.path.pop(0)
        if agent.path:
            agent.edge_remaining = self._edge_length(agent.node, agent.path[0])
            return leftover
        agent.edge_remaining = 0.0
        if agent.target is not None and agent.node == agent.target:
            agent.phase = "sweep"
            agent.phase_remaining = self._sweep_duration(agent.target)
        else:
            agent.phase = "idle"
        return leftover

    def _is_deadlocked(self, cleared: dict[NodeId, float]) -> bool:
        remaining = [r for r in self.rooms if r not in cleared]
        if not remaining:
            return False
        if any(agent.phase != "idle" for agent in self._agents):
            return False
        for agent in self._agents:
            for room in remaining:
                if self.shortest_path_len(agent.node, room) is not None:
                    return False
        return True

#!/usr/bin/env python3
"""HiMCM 2025 Problem A / Req 4：多场景 × 烟况蒙特卡洛扫荡实验。

配置
----
- Office：2 名消防员（Figure 1 基线对照）
- Daycare：2 / 3 / 4 名消防员
- Warehouse：3 / 4 / 5 名消防员
- 工况：``clear`` 无烟理想；``smoke`` 动态火/烟扩散（随机起火点）

每个 (场景, 人数, 工况) 独立重复 ``N_MC=30`` 次。指标：

- ``t_clear``：清空总耗时 Total Sweep Time（s）；失败则为该次仿真终止时刻
- ``vulnerable_success_rate``：已清空房间脆弱权重 / 全图脆弱权重
- ``redundancy_rate``：可搜寻节点的重复进入次数 / 房间数

结果写入 ``results/experiment_summary.csv``（真实仿真，禁止事后手填）。

运行::

    PYTHONPATH=. python experiments/run_sweep_experiments.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment.hazards import HazardModel
from src.environment.layouts import (
    exit_nodes,
    get_daycare_layout,
    get_office_layout,
    get_warehouse_layout,
    searchable_nodes,
)
from src.traditional_planner.scoring_planner import PlannerConfig, ScoringPlanner

HazardMode = Literal["clear", "smoke"]
NodeId = str

N_MC: int = 30
CSV_PATH: Path = ROOT / "results" / "experiment_summary.csv"
BASE_SEED: int = 2025

SCENARIO_STAFFING: dict[str, tuple[int, ...]] = {
    "office": (2,),
    "daycare": (2, 3, 4),
    "warehouse": (3, 4, 5),
}

MAX_STEPS: dict[str, int] = {
    "office": 3000,
    "daycare": 8000,
    "warehouse": 5000,
}

DT: dict[str, float] = {
    "office": 1.0,
    "daycare": 1.0,
    "warehouse": 2.0,
}

CSV_FIELDS: tuple[str, ...] = (
    "scenario",
    "n_agents",
    "hazard_mode",
    "run_id",
    "seed",
    "ignite_node",
    "success",
    "deadlock",
    "n_steps",
    "t_clear",
    "vulnerable_success_rate",
    "redundancy_rate",
    "n_rooms_cleared",
    "n_rooms_total",
)


@dataclass(frozen=True)
class ExperimentRow:
    """单次仿真落盘记录。"""

    scenario: str
    n_agents: int
    hazard_mode: str
    run_id: int
    seed: int
    ignite_node: str
    success: bool
    deadlock: bool
    n_steps: int
    t_clear: float
    vulnerable_success_rate: float
    redundancy_rate: float
    n_rooms_cleared: int
    n_rooms_total: int

    def as_dict(self) -> dict[str, Any]:
        """转为 CSV 行（布尔写成 0/1）。"""
        return {
            "scenario": self.scenario,
            "n_agents": self.n_agents,
            "hazard_mode": self.hazard_mode,
            "run_id": self.run_id,
            "seed": self.seed,
            "ignite_node": self.ignite_node,
            "success": int(self.success),
            "deadlock": int(self.deadlock),
            "n_steps": self.n_steps,
            "t_clear": f"{self.t_clear:.4f}",
            "vulnerable_success_rate": f"{self.vulnerable_success_rate:.6f}",
            "redundancy_rate": f"{self.redundancy_rate:.6f}",
            "n_rooms_cleared": self.n_rooms_cleared,
            "n_rooms_total": self.n_rooms_total,
        }


def build_layout(scenario: str) -> nx.Graph:
    """按场景名构造独立图实例。"""
    if scenario == "office":
        return get_office_layout()
    if scenario == "daycare":
        return get_daycare_layout(n_floors=3)
    if scenario == "warehouse":
        return get_warehouse_layout()
    raise ValueError(f"unknown scenario {scenario!r}")


def select_start_nodes(graph: nx.Graph, n_agents: int) -> tuple[NodeId, ...]:
    """出口优先，不足时用楼梯/走廊/通道节点补齐。"""
    exits = [str(n) for n in exit_nodes(graph)]
    extras = [
        str(n)
        for n, data in graph.nodes(data=True)
        if str(n) not in exits
        and str(data.get("node_type", "")) in {"stair", "hallway", "aisle"}
    ]
    pool = exits + extras
    leftover = [str(n) for n in graph.nodes if str(n) not in pool]
    pool.extend(leftover)
    if len(pool) < n_agents:
        raise ValueError(f"graph has {len(pool)} start candidates < n_agents={n_agents}")
    return tuple(pool[:n_agents])


def ignition_candidates(graph: nx.Graph) -> list[NodeId]:
    """随机起火点：房间/货位/走廊，排除出口以免封死入口。"""
    nodes: list[NodeId] = []
    for node, data in graph.nodes(data=True):
        ntype = str(data.get("node_type", ""))
        if ntype == "exit":
            continue
        if ntype in {"room", "bay", "hallway", "aisle"}:
            nodes.append(str(node))
    if not nodes:
        nodes = [str(n) for n in graph.nodes]
    return nodes


def vulnerable_weight(graph: nx.Graph, room: NodeId) -> float:
    """与规划器一致：``vulnerable_weight``，办公室缺省为 1。"""
    weight = float(graph.nodes[room].get("vulnerable_weight", 0.0))
    if weight > 0.0:
        return weight
    if graph.nodes[room].get("search_required"):
        return 1.0
    return 0.0


def vulnerable_success_rate(
    graph: nx.Graph,
    rooms: list[NodeId],
    cleared: dict[NodeId, float],
) -> float:
    """已清空脆弱权重 / 总脆弱权重，取值 [0, 1]。"""
    total = sum(vulnerable_weight(graph, r) for r in rooms)
    if total <= 0.0:
        return 1.0 if rooms and len(cleared) == len(rooms) else 0.0
    saved = sum(vulnerable_weight(graph, r) for r in rooms if r in cleared)
    return float(saved / total)


def redundancy_rate(
    graph: nx.Graph,
    trajectories: dict[str, list[NodeId]],
    n_rooms: int,
) -> float:
    """重复扫荡冗余度：可搜寻节点超额进入次数 / |R|。

    房间被进入 1 次贡献 0；进入 v_r 次贡献 v_r - 1。
    """
    if n_rooms <= 0:
        return 0.0
    visits: Counter[NodeId] = Counter()
    for path in trajectories.values():
        for node in path:
            if graph.nodes[node].get("search_required"):
                visits[str(node)] += 1
    extra = sum(max(count - 1, 0) for count in visits.values())
    return float(extra / n_rooms)


def make_planner_config(
    n_agents: int,
    start_nodes: tuple[NodeId, ...],
    dt: float,
) -> PlannerConfig:
    """实验用评分权重（含禁忌/动量）。"""
    return PlannerConfig(
        n_agents=n_agents,
        start_nodes=start_nodes,
        w_dist=1.0,
        w_hazard=8.0,
        w_priority=2.0,
        w_redundancy=80.0,
        w_tabu=50.0,
        w_mom=15.0,
        tabu_tenure=3,
        v_hall=1.5,
        v_room=0.8,
        t_sweep=20.0,
        t_tag=5.0,
        dt=dt,
    )


def run_one_trial(
    *,
    scenario: str,
    n_agents: int,
    hazard_mode: HazardMode,
    run_id: int,
    seed: int,
) -> ExperimentRow:
    """单次蒙特卡洛：构图、可选点燃、跑规划器、计算三项指标。"""
    graph = build_layout(scenario)
    rooms = searchable_nodes(graph)
    start_nodes = select_start_nodes(graph, n_agents)
    rng = np.random.default_rng(seed)

    ignite_node = ""
    hazard: HazardModel | None = None
    if hazard_mode == "smoke":
        candidates = ignition_candidates(graph)
        ignite_node = str(candidates[int(rng.integers(0, len(candidates)))])
        hazard = HazardModel(
            graph,
            p0=0.08,
            v_fire=0.45,
            smoke_speed_ratio=1.5,
            speed_decay_beta=0.40,
            seed=seed,
        )
        hazard.ignite(ignite_node)

    config = make_planner_config(n_agents, start_nodes, DT[scenario])
    planner = ScoringPlanner.from_config(graph, config, hazard=hazard)
    result = planner.run(max_steps=MAX_STEPS[scenario])

    t_clear = float(result.t_clear)
    v_rate = vulnerable_success_rate(graph, rooms, result.room_clear_times)
    r_rate = redundancy_rate(graph, planner.trajectories(), len(rooms))

    return ExperimentRow(
        scenario=scenario,
        n_agents=n_agents,
        hazard_mode=hazard_mode,
        run_id=run_id,
        seed=seed,
        ignite_node=ignite_node,
        success=bool(result.success),
        deadlock=bool(result.deadlock),
        n_steps=int(result.n_steps),
        t_clear=t_clear,
        vulnerable_success_rate=v_rate,
        redundancy_rate=r_rate,
        n_rooms_cleared=len(result.room_clear_times),
        n_rooms_total=len(rooms),
    )


def iter_jobs() -> list[tuple[str, int, HazardMode, int]]:
    """展开全部 (场景, 人数, 工况, run_id) 任务。"""
    jobs: list[tuple[str, int, HazardMode, int]] = []
    for scenario, staffs in SCENARIO_STAFFING.items():
        for n_agents in staffs:
            for mode in ("clear", "smoke"):
                for run_id in range(N_MC):
                    jobs.append((scenario, n_agents, mode, run_id))
    return jobs


def job_seed(scenario: str, n_agents: int, mode: HazardMode, run_id: int) -> int:
    """可复现种子：场景哈希 + 人数 + 工况 + run_id。"""
    scen_code = {"office": 1, "daycare": 2, "warehouse": 3}[scenario]
    mode_code = 0 if mode == "clear" else 1
    return int(BASE_SEED + 100000 * scen_code + 1000 * n_agents + 100 * mode_code + run_id)


def run_all(csv_path: Path = CSV_PATH) -> Path:
    """执行全部试验并流式写入 CSV。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = iter_jobs()
    n_jobs = len(jobs)
    print(f"Monte Carlo sweep: {n_jobs} trials (N_MC={N_MC}) -> {csv_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        handle.flush()
        for idx, (scenario, n_agents, mode, run_id) in enumerate(jobs, start=1):
            seed = job_seed(scenario, n_agents, mode, run_id)
            row = run_one_trial(
                scenario=scenario,
                n_agents=n_agents,
                hazard_mode=mode,
                run_id=run_id,
                seed=seed,
            )
            writer.writerow(row.as_dict())
            handle.flush()
            print(
                f"[{idx:03d}/{n_jobs}] {scenario} n={n_agents} {mode} "
                f"run={run_id:02d} success={int(row.success)} "
                f"t_clear={row.t_clear:.1f}s "
                f"vuln={row.vulnerable_success_rate:.3f} "
                f"red={row.redundancy_rate:.3f}"
            )
    print(f"Wrote {n_jobs} rows to {csv_path}")
    return csv_path


def main() -> None:
    """命令行入口。"""
    run_all()


if __name__ == "__main__":
    main()

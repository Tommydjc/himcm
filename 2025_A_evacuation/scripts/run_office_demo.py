#!/usr/bin/env python3
"""HiMCM 2025 Requirement 2：Figure 1 办公室双消防员清空可视化演示。

可独立运行::

    PYTHONPATH=. python scripts/run_office_demo.py

每个仿真步对应 15 秒物理时间；清空完成后写出论文插图
``paper_figures/office_sweep_trajectory.png``。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.environment.layouts import exit_nodes, get_office_layout, searchable_nodes
    from src.traditional_planner.scoring_planner import PlannerConfig, ScoringPlanner
except ImportError:
    src_dir = ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from environment.layouts import exit_nodes, get_office_layout, searchable_nodes
    from traditional_planner.scoring_planner import PlannerConfig, ScoringPlanner

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

SECONDS_PER_STEP: float = 15.0
MAX_STEPS: int = 100

DISPLAY_NAME: dict[str, str] = {
    "E_W": "Exit_West",
    "E_E": "Exit_East",
    "H1": "Hallway_1",
    "H2": "Hallway_2",
    "H3": "Hallway_3",
    "N1": "Office_N1",
    "N2": "Office_N2",
    "N3": "Office_N3",
    "S1": "Office_S1",
    "S2": "Office_S2",
    "S3": "Office_S3",
}

NODE_TYPE_COLOR: dict[str, str] = {
    "exit": "#2ca02c",
    "hallway": "#7f7f7f",
    "room": "#1f77b4",
}

AGENT_COLORS: dict[str, str] = {
    "F1": "#d62728",
    "F2": "#9467bd",
}

OFFICE_PLANNER_CONFIG = PlannerConfig(
    n_agents=2,
    start_nodes=("E_W", "E_E"),
    w_dist=1.0,
    w_hazard=5.0,
    w_priority=2.0,
    w_redundancy=80.0,
    v_hall=1.5,
    v_room=0.8,
    t_sweep=20.0,
    t_tag=5.0,
    dt=SECONDS_PER_STEP,
)


def pretty(node: str) -> str:
    """把内部节点 id 映射为轨迹日志中的可读名称。"""
    return DISPLAY_NAME.get(node, node)


def format_trajectory(nodes: list[str]) -> str:
    """输出 ``A -> B -> C`` 形式的轨迹链。"""
    return " -> ".join(pretty(n) for n in nodes)


def assemble_office_planner() -> tuple[nx.Graph, ScoringPlanner, list[str], list[str]]:
    """加载 Figure 1 布局、房间/出口集合，并装配双人评分规划器。"""
    graph = get_office_layout()
    rooms = searchable_nodes(graph)
    exits = exit_nodes(graph)
    planner = ScoringPlanner.from_config(graph, OFFICE_PLANNER_CONFIG, hazard=None)
    return graph, planner, rooms, exits


def print_banner(rooms: list[str], exits: list[str]) -> None:
    """打印演示头信息，避免静默启动。"""
    print("=" * 72)
    print(" HiMCM 2025 Problem A  |  Requirement 2  |  Office Dual-Firefighter Demo")
    print("=" * 72)
    print(f" Rooms ({len(rooms)}): {', '.join(pretty(r) for r in rooms)}")
    print(f" Exits ({len(exits)}): {', '.join(pretty(e) for e in exits)}")
    print(
        f" Agents: F1 @ {pretty('E_W')}, F2 @ {pretty('E_E')}  |  "
        f"dt = {SECONDS_PER_STEP:.0f} s/step  |  max_steps = {MAX_STEPS}"
    )
    print(
        " PlannerConfig: "
        f"w_dist={OFFICE_PLANNER_CONFIG.w_dist}, "
        f"w_hazard={OFFICE_PLANNER_CONFIG.w_hazard}, "
        f"w_priority={OFFICE_PLANNER_CONFIG.w_priority}, "
        f"w_redundancy={OFFICE_PLANNER_CONFIG.w_redundancy}"
    )
    print("-" * 72)


def print_step_status(planner: ScoringPlanner) -> None:
    """打印单步：步号、物理时间、消防员节点与相位。"""
    t_phys = planner.n_steps * SECONDS_PER_STEP
    n_clear = len(planner.cleared)
    n_rooms = len(planner.rooms)
    parts: list[str] = [
        f"[Step {planner.n_steps:03d}]",
        f"t = {t_phys:6.0f} s",
        f"cleared {n_clear}/{n_rooms}",
    ]
    for state in planner.agent_states():
        agent_id = str(state["agent_id"])
        node = pretty(str(state["node"]))
        phase = str(state["phase"])
        target = state["target"]
        target_txt = pretty(str(target)) if target is not None else "-"
        parts.append(f"{agent_id}: {node} ({phase}, tgt={target_txt})")
    print(" | ".join(parts))


def print_requirement2_metrics(planner: ScoringPlanner, status: str) -> None:
    """量化输出 Requirement 2 清空指标。"""
    t_clear = max(planner.cleared.values()) if planner.cleared else planner.time_s
    minutes = t_clear / 60.0
    print("-" * 72)
    print(" Requirement 2  —  Clearance Metrics")
    print("-" * 72)
    print(f" Status            : {status}")
    print(f" Total Steps       : {planner.n_steps}")
    print(f" Estimated T_clear : {t_clear:.1f} 秒 / {minutes:.2f} 分钟")
    print(f" Wall-clock (N*dt) : {planner.n_steps * SECONDS_PER_STEP:.1f} 秒")
    if planner.cleared:
        print(" Room timestamps   :")
        for room, stamp in sorted(planner.cleared.items(), key=lambda kv: kv[1]):
            print(f"   {pretty(room):<12}  {stamp:7.2f} s")
    print()
    print(" Agent trajectory chains:")
    for agent_id, nodes in planner.trajectories().items():
        print(f"   Agent {agent_id}: {format_trajectory(nodes)}")
    print("=" * 72)


def _polyline_with_offset(
    graph: nx.Graph,
    nodes: list[str],
    y_shift: float,
) -> list[tuple[float, float]]:
    """按节点 pos 取样，走廊方向略作纵向错位以免两条动线重合。"""
    points: list[tuple[float, float]] = []
    for node in nodes:
        x0, y0 = graph.nodes[node]["pos"]
        points.append((float(x0), float(y0) + y_shift))
    return points


def draw_office_trajectory_figure(
    graph: nx.Graph,
    planner: ScoringPlanner,
    save_path: Path,
) -> None:
    """绘制建筑拓扑 + 双消防员动线，300 DPI tight 保存。"""
    pos: dict[Any, tuple[float, float]] = {
        n: (float(d["pos"][0]), float(d["pos"][1])) for n, d in graph.nodes(data=True)
    }
    labels = {n: pretty(str(n)) for n in graph.nodes}
    node_colors = [
        NODE_TYPE_COLOR.get(str(graph.nodes[n].get("node_type", "")), "#8c564b")
        for n in graph.nodes
    ]

    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#4a4a4a", width=1.4, alpha=0.85)
    node_coll = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=900,
        edgecolors="black",
        linewidths=0.8,
    )
    node_coll.set_zorder(3)
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7.5, font_weight="bold")

    y_shifts = {"F1": 0.55, "F2": -0.55}
    for agent_id, nodes in planner.trajectories().items():
        color = AGENT_COLORS.get(agent_id, "#ff7f0e")
        pts = _polyline_with_offset(graph, nodes, y_shifts.get(agent_id, 0.0))
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=color, linewidth=2.4, alpha=0.9, zorder=4, label=f"Agent {agent_id}")
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                if abs(x2 - x1) + abs(y2 - y1) < 1e-9:
                    continue
                arrow = FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    arrowstyle="-|>",
                    mutation_scale=12.0,
                    linewidth=0.0,
                    color=color,
                    zorder=5,
                )
                ax.add_patch(arrow)
        if pts:
            ax.scatter(
                [pts[0][0]],
                [pts[0][1]],
                s=80,
                marker="^",
                color=color,
                edgecolors="black",
                zorder=6,
            )
            ax.scatter(
                [pts[-1][0]],
                [pts[-1][1]],
                s=70,
                marker="o",
                color=color,
                edgecolors="black",
                zorder=6,
            )

    ax.set_title("HiMCM 2025 Req 2: Dual-Firefighter Evacuation Trajectory")
    ax.set_aspect("equal")
    ax.axis("off")

    type_handles = [
        Patch(facecolor=NODE_TYPE_COLOR["exit"], edgecolor="black", label="Exit"),
        Patch(facecolor=NODE_TYPE_COLOR["hallway"], edgecolor="black", label="Hallway"),
        Patch(facecolor=NODE_TYPE_COLOR["room"], edgecolor="black", label="Office"),
    ]
    agent_handles = [
        Line2D([0], [0], color=AGENT_COLORS["F1"], lw=2.4, label="Agent F1 (West)"),
        Line2D([0], [0], color=AGENT_COLORS["F2"], lw=2.4, label="Agent F2 (East)"),
    ]
    ax.legend(handles=type_handles + agent_handles, loc="upper right", framealpha=0.92)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" Figure saved: {save_path}")


def main() -> None:
    """Requirement 2 办公室双人清空：步进日志 + 指标 + 插图。"""
    graph, planner, rooms, exits = assemble_office_planner()
    print_banner(rooms, exits)

    status = "running"
    for _ in range(MAX_STEPS):
        status = planner.tick()
        print_step_status(planner)
        if status == "cleared":
            print(" >>> Building fully cleared. Stopping loop.")
            break
        if status == "deadlock":
            print(" >>> Deadlock detected. Stopping loop.")
            break
    else:
        status = "timeout"
        print(f" >>> Reached MAX_STEPS={MAX_STEPS} without full clearance.")

    print_requirement2_metrics(planner, status)

    figure_path = ROOT / "paper_figures" / "office_sweep_trajectory.png"
    draw_office_trajectory_figure(graph, planner, figure_path)


if __name__ == "__main__":
    main()

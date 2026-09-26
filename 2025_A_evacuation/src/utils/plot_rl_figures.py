#!/usr/bin/env python3
"""评估 GAT-PPO Office 策略，写出 CSV 与论文插图（仅真实 rollout）。

输出::
    results/rl_office_eval.csv
    paper_figures/fig4_rl_office_trajectory.png
    paper_figures/fig5_baseline_vs_rl.png
    paper_figures/fig6_rl_policy_eval.png

运行::

    PYTHONPATH=. python src/utils/plot_rl_figures.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment.layouts import get_office_layout
from src.rl.env_wrapper import GraphEvacuationEnv
from src.rl.gat_net import GATActorCritic
from src.rl.ppo_agent import PPOAgent, select_device
from src.utils.plot_figures import FIG_DIR, WONG_BLUE, WONG_ORANGE, WONG_VERMILLION, _configure_style, mean_ci95

CHECKPOINT = ROOT / "src" / "rl" / "checkpoints" / "gat_ppo_office.pt"
CSV_PATH = ROOT / "results" / "rl_office_eval.csv"
N_EVAL: int = 30
MAX_STEPS: int = 80

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
AGENT_COLORS: dict[str, str] = {"F1": "#d62728", "F2": "#9467bd"}


def load_agent() -> tuple[GraphEvacuationEnv, PPOAgent]:
    """从 checkpoint 装配环境与 PPOAgent。"""
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing {CHECKPOINT}; run src/rl/train.py first")
    device = select_device()
    graph = get_office_layout()
    env = GraphEvacuationEnv(graph, n_agents=2, max_steps=MAX_STEPS, enable_hazard=False)
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    max_actions = int(payload.get("max_actions", env.max_actions))
    net = GATActorCritic(in_dim=6, hidden_dim=64, heads=4, max_actions=max_actions)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    net.load_state_dict(state)
    agent = PPOAgent(net, device=device)
    net.eval()
    return env, agent


def run_episode(
    env: GraphEvacuationEnv,
    agent: PPOAgent,
    *,
    seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    """单回合评估，记录步数、回报、清空与轨迹。"""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    obs, info = env.reset(seed=seed)
    traj = [[str(env.agent_nodes[0])], [str(env.agent_nodes[1])]]
    ep_ret = 0.0
    done = False
    truncated = False
    last_info = info
    n_steps = 0
    while not (done or truncated):
        actions, _logp, _v = agent.act(obs, deterministic=deterministic)
        obs, reward, done, truncated, last_info = env.step(actions)
        ep_ret += float(reward)
        n_steps += 1
        traj[0].append(str(env.agent_nodes[0]))
        traj[1].append(str(env.agent_nodes[1]))
    return {
        "mode": "greedy" if deterministic else "stochastic",
        "seed": seed,
        "steps": n_steps,
        "reward": ep_ret,
        "n_cleared": int(last_info.get("n_cleared", 0)),
        "all_clear": int(bool(last_info.get("all_clear", False))),
        "survived": int(bool(last_info.get("survived", False))),
        "traj_f1": traj[0],
        "traj_f2": traj[1],
    }


def evaluate_and_save_csv(env: GraphEvacuationEnv, agent: PPOAgent) -> list[dict[str, Any]]:
    """贪心与随机各 N_EVAL 次，写入 CSV（轨迹列不入库，仅标量）。"""
    records: list[dict[str, Any]] = []
    for i in range(N_EVAL):
        records.append(run_episode(env, agent, seed=1000 + i, deterministic=True))
    for i in range(N_EVAL):
        records.append(run_episode(env, agent, seed=2000 + i, deterministic=False))
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ["mode", "seed", "steps", "reward", "n_cleared", "all_clear", "survived"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in fields})
    return records


def _best_clear_episode(records: list[dict[str, Any]]) -> dict[str, Any]:
    """优先全清且步数最少的随机回合，否则取清空房间最多者。"""
    stoch = [r for r in records if r["mode"] == "stochastic"]
    cleared = [r for r in stoch if r["all_clear"] == 1]
    if cleared:
        return min(cleared, key=lambda r: int(r["steps"]))
    return max(stoch, key=lambda r: int(r["n_cleared"]))


def plot_rl_trajectory(graph: nx.Graph, episode: dict[str, Any], save_path: Path) -> None:
    """RL 双消防员动线（与规则规划器插图同一套配色）。"""
    pos = {n: (float(d["pos"][0]), float(d["pos"][1])) for n, d in graph.nodes(data=True)}
    labels = {n: DISPLAY_NAME.get(str(n), str(n)) for n in graph.nodes}
    node_colors = [
        NODE_TYPE_COLOR.get(str(graph.nodes[n].get("node_type", "")), "#8c564b") for n in graph.nodes
    ]
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#4a4a4a", width=1.4, alpha=0.85)
    coll = nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_color=node_colors, node_size=900, edgecolors="black", linewidths=0.8
    )
    coll.set_zorder(3)
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7.5, font_weight="bold")

    trajs = {"F1": episode["traj_f1"], "F2": episode["traj_f2"]}
    y_shifts = {"F1": 0.55, "F2": -0.55}
    for aid, nodes in trajs.items():
        color = AGENT_COLORS[aid]
        pts: list[tuple[float, float]] = []
        for node in nodes:
            x0, y0 = graph.nodes[node]["pos"]
            pts.append((float(x0), float(y0) + y_shifts[aid]))
        if len(pts) >= 2:
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=color,
                linewidth=2.2,
                alpha=0.85,
                zorder=4,
            )
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                if abs(x2 - x1) + abs(y2 - y1) < 1e-9:
                    continue
                ax.add_patch(
                    FancyArrowPatch(
                        (x1, y1),
                        (x2, y2),
                        arrowstyle="-|>",
                        mutation_scale=11.0,
                        linewidth=0.0,
                        color=color,
                        zorder=5,
                    )
                )
        if pts:
            ax.scatter([pts[0][0]], [pts[0][1]], s=80, marker="^", color=color, edgecolors="black", zorder=6)
            ax.scatter([pts[-1][0]], [pts[-1][1]], s=70, marker="o", color=color, edgecolors="black", zorder=6)

    ax.set_title("GAT-PPO Office trajectory (stochastic rollout, two firefighters)")
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [
        Patch(facecolor=NODE_TYPE_COLOR["exit"], edgecolor="black", label="Exit"),
        Patch(facecolor=NODE_TYPE_COLOR["hallway"], edgecolor="black", label="Hallway"),
        Patch(facecolor=NODE_TYPE_COLOR["room"], edgecolor="black", label="Office"),
        Line2D([0], [0], color=AGENT_COLORS["F1"], lw=2.2, label="Agent F1 (West)"),
        Line2D([0], [0], color=AGENT_COLORS["F2"], lw=2.2, label="Agent F2 (East)"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.92)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_vs_rl(records: list[dict[str, Any]], save_path: Path) -> None:
    """规则规划器 vs GAT-PPO：全清率与平均挂牌房间数（来自真实评估）。"""
    greedy = [r for r in records if r["mode"] == "greedy"]
    stoch = [r for r in records if r["mode"] == "stochastic"]
    labels = ["Scoring planner\n(Office, $m=2$)", "GAT-PPO greedy", "GAT-PPO stochastic"]
    clear_rates = [
        100.0,
        100.0 * float(np.mean([r["all_clear"] for r in greedy])),
        100.0 * float(np.mean([r["all_clear"] for r in stoch])),
    ]
    rooms = [
        6.0,
        float(np.mean([r["n_cleared"] for r in greedy])),
        float(np.mean([r["n_cleared"] for r in stoch])),
    ]
    colors = [WONG_BLUE, WONG_ORANGE, WONG_VERMILLION]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    x = np.arange(3)
    axes[0].bar(x, clear_rates, color=colors, edgecolor="black", width=0.62)
    axes[0].set_xticks(x, labels, fontsize=8.5)
    axes[0].set_ylabel("Full-clearance success rate (percentage)")
    axes[0].set_ylim(0.0, 110.0)
    axes[0].set_title("Clearance success")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    for i, v in enumerate(clear_rates):
        axes[0].text(i, v + 2.0, f"{v:.0f}%", ha="center", fontsize=9)
    axes[1].bar(x, rooms, color=colors, edgecolor="black", width=0.62)
    axes[1].set_xticks(x, labels, fontsize=8.5)
    axes[1].set_ylabel("Mean rooms tagged (count, max 6)")
    axes[1].set_ylim(0.0, 7.0)
    axes[1].set_title("Rooms tagged before timeout")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    for i, v in enumerate(rooms):
        axes[1].text(i, v + 0.12, f"{v:.2f}", ha="center", fontsize=9)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Office $m=2$: scoring planner vs GAT-PPO (ideal, no smoke)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rl_eval_box(records: list[dict[str, Any]], save_path: Path) -> None:
    """贪心 vs 随机策略的步数与回报分布。"""
    greedy = [r for r in records if r["mode"] == "greedy"]
    stoch = [r for r in records if r["mode"] == "stochastic"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    box_kw = {"patch_artist": True, "medianprops": {"color": "black", "linewidth": 1.3}}
    b1 = axes[0].boxplot(
        [[float(r["steps"]) for r in greedy], [float(r["steps"]) for r in stoch]],
        **box_kw,
    )
    b2 = axes[1].boxplot(
        [[float(r["reward"]) for r in greedy], [float(r["reward"]) for r in stoch]],
        **box_kw,
    )
    axes[0].set_xticklabels(["Greedy", "Stochastic"])
    axes[1].set_xticklabels(["Greedy", "Stochastic"])
    for box, color in zip(b1["boxes"], (WONG_ORANGE, WONG_VERMILLION)):
        box.set_facecolor(color)
        box.set_alpha(0.85)
    for box, color in zip(b2["boxes"], (WONG_ORANGE, WONG_VERMILLION)):
        box.set_facecolor(color)
        box.set_alpha(0.85)
    axes[0].set_ylabel("Episode length (graph steps)")
    axes[0].set_title("Steps to terminate")
    axes[1].set_ylabel("Undiscounted return")
    axes[1].set_title("Episode reward")
    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("GAT-PPO Office evaluation ($N=30$ per decoding mode)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize(records: list[dict[str, Any]]) -> None:
    """终端打印评估摘要，便于写入论文表格。"""
    for mode in ("greedy", "stochastic"):
        sub = [r for r in records if r["mode"] == mode]
        steps = np.asarray([r["steps"] for r in sub], dtype=float)
        rew = np.asarray([r["reward"] for r in sub], dtype=float)
        rooms = np.asarray([r["n_cleared"] for r in sub], dtype=float)
        clear = np.asarray([r["all_clear"] for r in sub], dtype=float)
        sm, slo, shi = mean_ci95(steps)
        rm, rlo, rhi = mean_ci95(rew)
        print(
            f"{mode:12} N={len(sub)} clear={clear.mean():.2%} "
            f"rooms={rooms.mean():.2f} steps={sm:.1f} [{slo:.1f},{shi:.1f}] "
            f"R={rm:.1f} [{rlo:.1f},{rhi:.1f}]"
        )


def main() -> None:
    """评估 checkpoint 并写出 RL 论文图。"""
    _configure_style()
    env, agent = load_agent()
    records = evaluate_and_save_csv(env, agent)
    summarize(records)
    best = _best_clear_episode(records)
    print(
        f"trajectory episode: mode={best['mode']} seed={best['seed']} "
        f"steps={best['steps']} cleared={best['n_cleared']} all_clear={best['all_clear']}"
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_rl_trajectory(env.graph, best, FIG_DIR / "fig4_rl_office_trajectory.png")
    plot_baseline_vs_rl(records, FIG_DIR / "fig5_baseline_vs_rl.png")
    plot_rl_eval_box(records, FIG_DIR / "fig6_rl_policy_eval.png")
    print(f"Wrote {CSV_PATH} and RL figures under {FIG_DIR}")


if __name__ == "__main__":
    main()

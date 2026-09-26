"""四策略同气象种子对决：野化 / 14 日刈割 / 结籽前一刀 / PPO。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.decision.tradeoff_model import first_true_bloom_day, prisms_mow_day
from src.rl.env_wrapper import ACTION_NAMES, DandelionEcoEnv
from src.rl.ppo_agent import PPOAgent

PACK_ROOT: Path = Path(__file__).resolve().parents[2]
FIG_DIR: Path = PACK_ROOT / "paper_figures"
WONG = ("#009E73", "#E69F00", "#D55E00", "#0072B2")


PolicyFn = Callable[[NDArray[np.float32], int, DandelionEcoEnv], int]


def _wild(_obs: NDArray[np.float32], _day: int, _env: DandelionEcoEnv) -> int:
    return 0


def _routine14(_obs: NDArray[np.float32], day: int, _env: DandelionEcoEnv) -> int:
    return 2 if day > 0 and day % 14 == 0 else 0


def _human_once(_obs: NDArray[np.float32], day: int, env: DandelionEcoEnv) -> int:
    cut = prisms_mow_day(env.forcings)
    return 2 if day == cut else 0


def rollout(
    env: DandelionEcoEnv,
    policy: PolicyFn,
    weather_seed: int,
) -> dict[str, object]:
    """跑满 365 日，记录回报与动作。"""
    obs, _info = env.reset(options={"weather_seed": weather_seed})
    rewards: list[float] = []
    bees: list[float] = []
    turfs: list[float] = []
    costs: list[float] = []
    actions: list[int] = []
    terminated = False
    while not terminated:
        action = policy(obs, env.day_index, env)
        obs, reward, terminated, _trunc, info = env.step(action)
        rewards.append(float(reward))
        bees.append(float(info["bee"]))
        turfs.append(float(info["turf"]))
        costs.append(float(info["cost"]))
        actions.append(int(info["action"]))
    return {
        "net_utility": float(np.sum(rewards)),
        "pollination": float(np.sum(bees)),
        "turf_damage": float(np.mean(turfs)),
        "cost": float(np.sum(costs)),
        "cover_end": float(turfs[-1]),
        "n_mows": int(info["n_mows"]),
        "n_sprays": int(info["n_sprays"]),
        "actions": np.asarray(actions, dtype=np.int64),
        "bees": np.asarray(bees, dtype=np.float64),
    }


def evaluate_and_plot(ckpt: Path, weather_seed: int = 2023) -> Path:
    """四策略评估、写 CSV、出两张 300 DPI 图。"""
    env = DandelionEcoEnv(weather_seed=weather_seed)
    agent = PPOAgent()
    if ckpt.is_file():
        agent.load(ckpt)

    def learned(obs: NDArray[np.float32], _day: int, _env: DandelionEcoEnv) -> int:
        return agent.greedy(obs)

    specs: list[tuple[str, str, PolicyFn]] = [
        ("rewilding", "A rewild", _wild),
        ("routine_14d", "B routine 14d", _routine14),
        ("human_once", "C pre-seed cut", _human_once),
        ("ppo", "D learned PPO", learned),
    ]
    rows: list[dict[str, object]] = []
    action_series: dict[str, NDArray[np.int64]] = {}
    for name, label, fn in specs:
        stats = rollout(env, fn, weather_seed)
        action_series[name] = stats["actions"]  # type: ignore[assignment]
        rows.append(
            {
                "policy": name,
                "label": label,
                "pollination": stats["pollination"],
                "turf_damage": stats["turf_damage"],
                "cost": stats["cost"],
                "net_utility": stats["net_utility"],
                "cover_end": stats["cover_end"],
                "n_mows": stats["n_mows"],
                "n_sprays": stats["n_sprays"],
                "weather_seed": weather_seed,
            }
        )
    table = pd.DataFrame(rows)
    out_csv = PACK_ROOT / "results" / "rl_policy_evaluation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    _style()
    plot_action_timeline(action_series["ppo"], env, FIG_DIR)
    plot_radar(table, FIG_DIR)
    print(f"wrote {out_csv.relative_to(PACK_ROOT)}")
    print(table.to_string(index=False))
    return out_csv


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "savefig.dpi": 300,
            "text.usetex": False,
            "mathtext.default": "regular",
        }
    )


def plot_action_timeline(
    actions: NDArray[np.int64],
    env: DandelionEcoEnv,
    fig_dir: Path,
) -> Path:
    """365 日动作甘特热力。"""
    heat = np.zeros((4, 365), dtype=np.float64)
    for t, act in enumerate(actions):
        heat[int(act), t] = 1.0
    fig, ax = plt.subplots(figsize=(10.4, 2.8), constrained_layout=True)
    ax.imshow(heat, aspect="auto", cmap="Blues", extent=(1, 365, 3.5, -0.5))
    ax.set_yticks(range(4))
    ax.set_yticklabels(ACTION_NAMES)
    ax.set_xlabel("day")
    ax.set_title("PPO action timeline")
    bloom = first_true_bloom_day(env.base_forcings)
    cut = prisms_mow_day(env.base_forcings)
    ax.axvline(bloom + 1, color="#009E73", ls=":", lw=1.0)
    ax.axvline(cut + 1, color="#D55E00", ls="--", lw=1.0)
    ax.axvspan(59.0, 120.0, color="#0072B2", alpha=0.08)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_rl_action_timeline.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _minmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if abs(hi - lo) < 1.0e-12:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def plot_radar(table: pd.DataFrame, fig_dir: Path) -> Path:
    """五轴雷达：授粉、草坪健康、成本控制、抑草、净效益。"""
    poll = _minmax(table["pollination"].to_numpy(dtype=np.float64))
    turf_h = _minmax(-table["turf_damage"].to_numpy(dtype=np.float64))
    cheap = _minmax(-table["cost"].to_numpy(dtype=np.float64))
    suppress = _minmax(-table["cover_end"].to_numpy(dtype=np.float64))
    net = _minmax(table["net_utility"].to_numpy(dtype=np.float64))
    radar = np.column_stack((poll, turf_h, cheap, suppress, net))
    names = ("pollination", "turf health", "low cost", "suppression", "net utility")
    ang = np.linspace(0.0, 2.0 * np.pi, len(names), endpoint=False)
    ang = np.concatenate((ang, ang[:1]))
    fig, ax = plt.subplots(figsize=(5.6, 5.2), subplot_kw={"polar": True}, constrained_layout=True)
    for i, row in table.iterrows():
        vals = np.concatenate((radar[int(i)], radar[int(i)][:1]))
        ax.plot(ang, vals, color=WONG[int(i)], lw=1.7, label=str(row["label"]))
        ax.fill(ang, vals, color=WONG[int(i)], alpha=0.08)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(names)
    ax.set_yticklabels([])
    ax.set_title("Policy benchmark (higher better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), frameon=False, fontsize=8)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_policy_benchmark_radar.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    ckpt = PACK_ROOT / "src" / "rl" / "checkpoints" / "ppo_dandelion_policy.pt"
    evaluate_and_plot(ckpt)


if __name__ == "__main__":
    main()

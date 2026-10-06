# HiMCM 2025 A — Emergency Evacuation Sweeps

图上多智能体清扫：布局拓扑 + 传统规划器清场时间 \(t_{\mathrm{clear}}\)。GAT–PPO 是题包 `src/rl/` 代码路径；评测权威仍是规划器 CSV，不要把训练 return 写成清场秒数。

```yaml
type: problem
year: 2025
status: ingested
sources:
  - 2025_A_evacuation/results/experiment_summary.csv
  - 2025_A_evacuation/results/rl_office_eval.csv
  - 2025_A_evacuation/src/traditional_planner/scoring_planner.py
  - 2025_A_evacuation/src/rl/gat_net.py
  - 2025_A_evacuation/src/rl/ppo_agent.py
```

## 赛题要交什么

官方 PDF：COMAP HiMCM 2025 Problem A（Emergency Evacuation Sweeps）。

1. 房间/走廊图上的搜救清扫，不是把「发现」写成「救出」。
2. 报告清场时间、死锁、脆弱房间成功率。
3. 规划基线与可选 RL 必须分表。

## 题包地图

| 路径 | 用途 |
| --- | --- |
| `src/environment/` | 布局与烟害 |
| `src/traditional_planner/` | 评分规划器 |
| `src/rl/` | GAT 网 + PPO（代码有；增益未在本 CSV 证实） |
| `results/experiment_summary.csv` | 规划器多情景 \(t_{\mathrm{clear}}\) |
| `results/rl_office_eval.csv` | office 上 greedy 对照，80 步、`n_cleared=2`、`all_clear=0` |

## 规划器锚点（clear 模式；`t_clear` 列为重复均值。office 各 run 的 `n_steps` 均为 135；warehouse 单次 `n_steps` 与均值不同。）

| 情景 | n_agents | \(t_{\mathrm{clear}}\) | rooms | success |
| --- | ---: | ---: | ---: | --- |
| office | 2 | 135 | 6/6 | 1 |
| daycare | 2 | 439 | 18/18 | 1 |
| daycare | 3 | 300.67 | 18/18 | 1 |
| daycare | 4 | 255.67 | 18/18 | 1 |
| warehouse | 3 | 424.67 | 40/40 | 1 |
| warehouse | 4 | 333 | 40/40 | 1 |
| warehouse | 5 | 262.33 | 40/40 | 1 |

`hazard_mode=smoke` 行大量 `success=0`、`deadlock=1`。仓库 smoke 偶发 `n_steps=5000`（截断）。不要把 smoke 的 \(t_{\mathrm{clear}}\) 平均成「已清场」。

## 失败模式

- 发现 ≠ 救出。
- 训练 return ≠ 评测 \(t_{\mathrm{clear}}\)。
- GAT–PPO 改进幅度：`unverified`（无优于规划器的对照表）。

## Related

- [4h 交卷](../playbooks/4h-combat-playbook.md)
- [评价 vs 仿真](../comparisons/evaluation-vs-simulation.md)
- [Notebook vs 模块](../comparisons/notebook-hell-vs-modular.md)

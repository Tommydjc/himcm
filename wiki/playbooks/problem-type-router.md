# 题型路由（模式 A–F）

新题先归类再建模。迁移的是流水线与禁令，不是旧年 CSV 数字。

```yaml
type: playbook
year: cross
status: contest-ready
sources:
  - STUDIES.md
  - wiki/comparisons/evaluation-vs-simulation.md
  - wiki/synthesis.md
  - 2020_A_summer_job_factor/results/model_comparison_metrics.csv
  - 2021_A_solar_storage/results/optimal_sizing_solution.csv
  - 2022_A_honeybee_dynamics/results/orchard_pollination_recommendation.csv
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
  - 2025_A_evacuation/results/experiment_summary.csv
```

## 快速决策

| ID | 模式 | 题干信号 | 基线（必须先） | 加分（后上） | 代表题 / 入口 |
| --- | --- | --- | --- | --- | --- |
| **A** | 评价 / 排序 | 多方案、多准则、选拔、排名 | 指标表 → AHP/熵权 → SAW/TOPSIS → OAT | 小样本 LOOCV（弱互证） | 2024；[AHP](../methods/ahp-eigenvalue.md)、[SAW](../methods/saw-topsis.md) |
| **B** | 问卷 / 潜变量 | 量表、岗位匹配、因子 | EFA/降维 → 权重 → 聚类或分类 + CV | LLM 语义因子 | 2020；[EFA](../methods/factor-analysis-llmfactor.md)、[熵权](../methods/entropy-weight-objective-scoring.md) |
| **C** | 资源优化 / MILP | 容量、成本、可靠性、8760h | 时序负荷/发电 → 选型优化 → KPI | 多智能体协议层 | 2021；[CrewAI](../methods/crewai-cognitive-demand-response.md) |
| **D** | 动力学 + 灵敏度 | 种群/状态随时间、参数影响 | ODE/可微仿真 → 基线轨迹 → 弹性/分岔 | 下游运筹（密度、配额） | 2022；[DFM](../methods/dfm-torch-sim.md)、[Autograd](../methods/autograd-elasticity.md) |
| **E** | 空间机理 / 扩散 | 传播、覆盖、格子/连续空间 | 核/PDE/CA → 情景对照 → 影响因子 | 管理帕累托、可选 RL | 2023；[WALD](../methods/wald-plume.md)、[Lefkovitch](../methods/lefkovitch-self-thinning.md) |
| **F** | 图搜救 / 仿真 MDP | 清扫、路径、多智能体、清场时间 | 环境图 → 规划器基线 \(t_{\mathrm{clear}}\) | RL（须与基线分表） | 2025；[2025 问题](../problems/2025-a-evacuation.md) |

## 组合题

常见组合与主次：

| 组合 | 主模式 | 辅模式 | 注意 |
| --- | --- | --- | --- |
| 排名 + 预测 | A | B / 弱 ML | ML 不抢终裁（见 2024 failure-modes） |
| 动力学 + 管理 | D | 运筹 | 轨迹宇宙与 OR 表分列 |
| 空间入侵 + 管理 | E | 帕累托 | 情景气候 ≠ 观测 |
| 仿真 + 智能策略 | F | RL | 规划器 CSV 先于训练 return |

## 输出模板（开题粘贴）

```text
主模式：
辅模式：
基线方法与 wiki 链接：
基线验收（将写入哪份 results/*.csv）：
明确砍掉：
禁令自检（cross-year-failure-modes）：
```

## 数字纪律

路由表中的代表年数字（如 Flag SAW 0.5360、\(t_{\mathrm{clear}}=135\)）**仅作对照锚点**，禁止写入新题论文。新题数字只认新题包 CSV。

## Related

- [Contest Hub](contest-hub.md)
- [跨年失败模式](../concepts/cross-year-failure-modes.md)
- [评价 vs 仿真](../comparisons/evaluation-vs-simulation.md)
- [HMML 映射](../comparisons/hmml-vs-gohimcm.md)
- [synthesis](../synthesis.md)

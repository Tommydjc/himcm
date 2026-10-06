# 评价题 vs 仿真 / 机理题

跨年对照：2022 生态动力学+OR；2023 空间扩散；2024 MCDM；2025 图上搜救。各年数字不许互相填表。

```yaml
type: comparison
year: cross
status: ingested
sources:
  - 2022_A_honeybee_dynamics/results/orchard_pollination_recommendation.csv
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
  - 2024_A_olympic_sde/paper/himcm_paper.tex
  - 2025_A_evacuation/prompts/cursor_log.md
```

## 对照

| | 2022 A Bees | 2023 A Dandelion | 2024 A SDE | 2025 A Evacuation |
| --- | --- | --- | --- | --- |
| 题型 | 动力学 / 灵敏度 / 授粉 OR | 机理 / 空间扩散 + 影响因子 | 评价 / MCDM + ML | 仿真 / 图上搜救 |
| 主引擎 | DFM–Torch + Autograd + 果园 OR | WALD/混合核 + Lefkovitch | 锁定 AHP–SAW | 规划器 + 可选 RL |
| 权威数字 | `2022_A_honeybee_dynamics/results/*.csv` | `2023_A_dandelion_prisms/results/*.csv` | `2024_A_olympic_sde/results/*.csv` | `2025_A_evacuation/results/*.csv` |
| Wiki 状态 | [problem](../problems/2022-a-honeybee-dynamics.md) ingested | [problem](../problems/2023-a-dandelion-prisms.md) ingested | [problem](../problems/2024-a-olympic-sde.md) ingested | [problem](../problems/2025-a-evacuation.md) ingested |
| 典型陷阱 | 混 Khoury/Torch 轨迹；示例 Summary | 情景当观测；周步/日步平均；旧 TAROF=0 | 九行名次当 2032；LOOCV 当高精度 | 发现=救出；训练 return 当评测 |

2022 方法入口：[DFM](../methods/dfm-torch-sim.md)、[果园 OR](../methods/orchard-hive-or.md)。2023：[WALD](../methods/wald-plume.md)、[Lefkovitch](../methods/lefkovitch-self-thinning.md)。2024：[AHP](../methods/ahp-eigenvalue.md)、[SAW](../methods/saw-topsis.md)。不要把 2024 的 0.5360 写进疏散，也不要把 2023 的 0.4511 写进奥运会排序，或把 2022 的 \(K^\star=40\) 写进蒲公英割草。

## Related

- [总览 synthesis](../synthesis.md)
- [2022 问题](../problems/2022-a-honeybee-dynamics.md)
- [2023 问题](../problems/2023-a-dandelion-prisms.md)
- [2024 问题](../problems/2024-a-olympic-sde.md)
- [2025 问题](../problems/2025-a-evacuation.md)
- [2021 问题](../problems/2021-a-solar-storage.md)
- [2020 问题](../problems/2020-a-summer-job.md)
- [Notebook vs 模块](notebook-hell-vs-modular.md)

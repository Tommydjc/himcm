# Wiki 目录

GoHiMCM 知识库入口。赛时 / Query 先读本页，再打开 3–8 篇子页；对数字时回题包 `results/*.csv`。

```yaml
type: index
year: cross
status: ingested
sources:
  - STUDIES.md
  - AGENTS.md
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
```

## 题

- [2023 A Dandelion Prisms](problems/2023-a-dandelion-prisms.md) — 机理/空间仿真 + 影响因子（draft）
- [2024 A Olympic SDE](problems/2024-a-olympic-sde.md) — MCDM + 小样本 ML 互证（ingested，非 contest-ready）
- 2025 A Evacuation — 题包 `2025_A_evacuation/`；本目录尚未挂 problem 页，数字只认该包 CSV

## 方法（2023）

- [WALD 羽流](methods/wald-plume.md)
- [Lefkovitch 自疏](methods/lefkovitch-self-thinning.md)
- [Fisher–KPP](methods/fisher-kpp.md)
- [割草帕累托](methods/bioeconomic-pareto.md)

## 方法（2024）

- [AHP 特征值](methods/ahp-eigenvalue.md)
- [SAW / TOPSIS](methods/saw-topsis.md)
- [OAT 敏感性](methods/oat-sensitivity.md)
- [LOOCV logistic](methods/loocv-logistic.md)
- [AHP vs |β|](methods/ahp-vs-beta.md)

## 概念

- [IOC 六块与七叶](concepts/ioc-six-plus-xlsx.md)
- [两套宇宙](concepts/two-universes.md)
- [2024 失败模式](concepts/2024-a-failure-modes.md)
- [2023 失败模式](concepts/2023-a-failure-modes.md)
- [COMAP AI 披露](concepts/comap-ai-disclosure.md)

## Playbook

- [2024 研究闭环](playbooks/2024-a-study-loop.md)
- [2024 复现命令](playbooks/2024-repro-eval.md)
- [2023 复现命令](playbooks/2023-repro-sim.md)
- [CSV 到图](playbooks/results-to-figures.md)

## Source（2023）

- [monthly_metrics](sources/monthly-metrics.md)
- [bioeconomic_pareto](sources/bioeconomic-pareto.md)
- [impact_factors](sources/impact-factors.md)

## Source（2024）

- [historical_ranking](sources/historical-ranking.md)
- [brisbane_2032_ranking](sources/brisbane-2032-ranking.md)
- [sensitivity_weight_shock](sources/sensitivity-weight-shock.md)
- [ml_prediction_2032](sources/ml-prediction-2032.md)
- [ahp_vs_ml_weights](sources/ahp-vs-ml-weights.md)

## 对照与总览

- [评价 vs 仿真](comparisons/evaluation-vs-simulation.md)
- [synthesis](synthesis.md)
- [log](log.md)

## Related

- [2023 问题](problems/2023-a-dandelion-prisms.md)
- [2024 问题](problems/2024-a-olympic-sde.md)
- [synthesis](synthesis.md)

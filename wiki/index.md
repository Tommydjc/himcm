# Wiki 目录

GoHiMCM 知识库入口。赛时 / Query 先读本页，再打开 3–8 篇子页；对数字时回题包 `results/*.csv`。

```yaml
type: index
year: cross
status: ingested
sources:
  - STUDIES.md
  - AGENTS.md
  - 2022_A_honeybee_dynamics/results/orchard_pollination_recommendation.csv
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
```

## 题

- [2022 A Honeybee Dynamics](problems/2022-a-honeybee-dynamics.md) — 可微蜂群 + 弹性 + CCD + 20 英亩授粉 OR（ingested）
- [2023 A Dandelion Prisms](problems/2023-a-dandelion-prisms.md) — 机理/空间仿真 + 影响因子（ingested）
- [2024 A Olympic SDE](problems/2024-a-olympic-sde.md) — MCDM + 小样本 ML 互证（ingested，非 contest-ready）
- 2025 A Evacuation — 题包 `2025_A_evacuation/`；problem 页未挂时数字只认该包 CSV（STUDIES 标 ingested）

## 方法（2022）

- [DFM Torch 仿真](methods/dfm-torch-sim.md)
- [Autograd 弹性](methods/autograd-elasticity.md)
- [CCD 分岔](methods/ccd-bifurcation.md)
- [果园蜂箱 OR](methods/orchard-hive-or.md)

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
- [2022 失败模式](concepts/2022-a-failure-modes.md)
- [2023 失败模式](concepts/2023-a-failure-modes.md)
- [2024 失败模式](concepts/2024-a-failure-modes.md)
- [COMAP AI 披露](concepts/comap-ai-disclosure.md)

## Playbook

- [2022 研究闭环](playbooks/2022-a-study-loop.md)
- [2022 复现命令](playbooks/2022-repro-bee.md)
- [2023 研究闭环](playbooks/2023-a-study-loop.md)
- [2023 复现命令](playbooks/2023-repro-sim.md)
- [2024 研究闭环](playbooks/2024-a-study-loop.md)
- [2024 复现命令](playbooks/2024-repro-eval.md)
- [CSV 到图](playbooks/results-to-figures.md)

## Source（2022）

- [ccd_healthy_vs_stress](sources/ccd-healthy-vs-stress.md)
- [autograd_elasticity_ranking](sources/autograd-elasticity-ranking.md)
- [ccd_tipping_point](sources/ccd-tipping-point.md)
- [orchard_pollination_optimization](sources/orchard-pollination-optimization.md)

## Source（2023）

- [monthly_metrics](sources/monthly-metrics.md)
- [monthly_population_metrics](sources/monthly-population-metrics.md)
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

- [2022 问题](problems/2022-a-honeybee-dynamics.md)
- [2023 问题](problems/2023-a-dandelion-prisms.md)
- [2024 问题](problems/2024-a-olympic-sde.md)
- [synthesis](synthesis.md)

# Wiki 目录

GoHiMCM 知识库入口。赛时 / Query 先读本页，再打开 3–8 篇子页；对数字时回题包 `results/*.csv`。

```yaml
type: index
year: cross
status: ingested
sources:
  - STUDIES.md
  - AGENTS.md
  - 2020_A_summer_job_factor/results/entropy_weights.csv
  - 2021_A_solar_storage/results/optimal_sizing_solution.csv
  - 2022_A_honeybee_dynamics/results/orchard_pollination_recommendation.csv
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
  - 2025_A_evacuation/results/experiment_summary.csv
```

## 题（白皮书短页 + 已 ingest 细页）

- [2025 A Evacuation](problems/2025-a-evacuation.md) — 图清扫 + 规划器 \(t_{\mathrm{clear}}\)；GAT–PPO 代码路径未证实优于基线
- [2024 A Olympics](problems/2024-a-olympics.md) — 短页；细表 [2024-a-olympic-sde](problems/2024-a-olympic-sde.md)
- [2023 A Dandelion](problems/2023-a-dandelion.md) — 短页；细表 [2023-a-dandelion-prisms](problems/2023-a-dandelion-prisms.md)
- [2022 A Honeybee](problems/2022-a-honeybee.md) — 短页；细表 [2022-a-honeybee-dynamics](problems/2022-a-honeybee-dynamics.md)
- [2021 A Solar storage](problems/2021-a-solar-storage.md) — MCMC 负荷 + MILP + CrewAI DSR
- [2020 A Summer job](problems/2020-a-summer-job.md) — EFA + LLMFactor + 熵权 + K-Means + LOOCV（SYNTHETIC_FALLBACK）

## 方法（跨年新页）

- [EFA + LLMFactor](methods/factor-analysis-llmfactor.md)
- [PyTorch Autograd 灵敏度](methods/pytorch-autograd-sensitivity.md)
- [CrewAI 认知需求响应](methods/crewai-cognitive-demand-response.md)
- [Conv2D 空间 CA](methods/conv2d-spatiotemporal-ca.md)
- [熵权](methods/entropy-weight-objective-scoring.md)

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

- [4h 交卷](playbooks/4h-combat-playbook.md)
- [规格驱动 Cursor](playbooks/spec-driven-cursor-coding.md)
- [ASD-STE100 写作](playbooks/asd-ste100-academic-writing.md)
- [2022 研究闭环](playbooks/2022-a-study-loop.md)
- [2022 复现命令](playbooks/2022-repro-bee.md)
- [2023 研究闭环](playbooks/2023-a-study-loop.md)
- [2023 复现命令](playbooks/2023-repro-sim.md)
- [2024 研究闭环](playbooks/2024-a-study-loop.md)
- [2024 复现命令](playbooks/2024-repro-eval.md)
- [CSV 到图](playbooks/results-to-figures.md)

## Source（2020）

- [factor_loadings](sources/2020-factor-loadings.md)
- [entropy_weights](sources/2020-entropy-weights.md)
- [model_comparison](sources/2020-model-comparison.md)

## Source（2021）

- [optimal_sizing](sources/2021-optimal-sizing.md)
- [crewai_adaptive](sources/2021-crewai-adaptive.md)

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

## Source（2025）

- [experiment_summary](sources/2025-experiment-summary.md)

## 对照与总览

- [评价 vs 仿真](comparisons/evaluation-vs-simulation.md)
- [合成数据陷阱](comparisons/synthetic-data-trap.md)
- [Notebook vs 模块](comparisons/notebook-hell-vs-modular.md)
- [被动停电谬误](comparisons/passive-blackout-fallacy.md)
- [synthesis](synthesis.md)
- [log](log.md)
- [STUDIES 指针](STUDIES.md)

## Related

- [2025 问题](problems/2025-a-evacuation.md)
- [2024 问题](problems/2024-a-olympics.md)
- [2023 问题](problems/2023-a-dandelion.md)
- [2022 问题](problems/2022-a-honeybee.md)
- [2021 问题](problems/2021-a-solar-storage.md)
- [2020 问题](problems/2020-a-summer-job.md)
- [synthesis](synthesis.md)

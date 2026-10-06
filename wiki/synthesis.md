# 跨年方法总论

十年题包共用「人写 Raw、Agent 写 Wiki」。定量结论必须能回到当年 `YYYY_P_slug/results/*.csv`。

```yaml
type: synthesis
year: cross
status: ingested
sources:
  - STUDIES.md
  - 2020_A_summer_job_factor/results/model_comparison_metrics.csv
  - 2021_A_solar_storage/results/optimal_sizing_solution.csv
  - 2022_A_honeybee_dynamics/results/orchard_pollination_recommendation.csv
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
  - 2025_A_evacuation/results/experiment_summary.csv
  - AGENTS.md
```

## 生态动力学 / 授粉运筹

2022 A 用可微五维蜂群（DFM–Torch）做弹性与 CCD 分岔，再在 20 英亩扁桃园上做蜂箱密度 OR。入口：[2022 问题](problems/2022-a-honeybee-dynamics.md)、[Autograd 弹性](methods/autograd-elasticity.md)、[果园 OR](methods/orchard-hive-or.md)。健康峰 \(66{,}507\) / 越冬 \(38{,}622\)；硬拐点 \(\mu_F^*=0.305\)；\(K^\star=40\)、YieldRatio \(0.848\)。

## 机理 / 空间扩散

2023 A 把冠毛羽流与 Lefkovitch 周步卷积在 1 公顷格网上，三气候对照后做三物种影响因子；日步中心宇宙与周步西缘分列。入口：[2023 问题](problems/2023-a-dandelion-prisms.md)。温带 12 月周步覆盖 **0.4511**；TAROF impact **0.1512** 只在三行宇宙内成立。气候 CSV 默认是情景。

## 评价 / MCDM

2024 A 把 IOC 纲领收成七叶，用锁定 AHP–SAW 做回测与布里斯班短名单，再用 N=6 LOOCV 作弱分类对照。入口：[2024 问题](problems/2024-a-olympic-sde.md)、[AHP](methods/ahp-eigenvalue.md)、[SAW](methods/saw-topsis.md)。2032 决策叙事：Flag football SAW 第一（0.5360）；青年权=0 时逆转。不要用 logistic 均值把 Cricket 写成执委会终裁。

## 离网储能 / 认知削负荷

2021 A：8760 h + MILP 选型。现行 CSV：3× FREEDOH + 1× lead-carbon，capex \$9{,}700，LPSP 0.2489，停电 2725 h。CrewAI 只改 72 h 窗内负荷协议。入口：[2021 问题](problems/2021-a-solar-storage.md)、[CrewAI](methods/crewai-cognitive-demand-response.md)、[被动停电](comparisons/passive-blackout-fallacy.md)。

## 问卷因子 / 岗位匹配

2020 A：15 列是 `SYNTHETIC_FALLBACK`。KMO 0.445，\(m=3\) 方差 42.6%，熵权 \(w=(0.479,0.319,0.202)\)，LOOCV Top-1 0.20。入口：[2020 问题](problems/2020-a-summer-job.md)、[EFA](methods/factor-analysis-llmfactor.md)、[熵权](methods/entropy-weight-objective-scoring.md)、[合成数据](comparisons/synthetic-data-trap.md)。

## 仿真 / 搜救

2025 A：规划器 office 2 体 \(t_{\mathrm{clear}}=135\)（`experiment_summary.csv`）。GAT–PPO 在 `src/rl/`，增益 `unverified`。入口：[2025 问题](problems/2025-a-evacuation.md)。跨年结构见 [评价 vs 仿真](comparisons/evaluation-vs-simulation.md)。

## 赛时

读 [index](index.md) → 复制 `_template/` 建新年目录 → 在新目录建模 → 当年 `prompts/cursor_log.md` 记 AI → 不改旧年论文交差。

## Related

- [目录](index.md)
- [2025 问题](problems/2025-a-evacuation.md)
- [2024 问题](problems/2024-a-olympics.md)
- [2023 问题](problems/2023-a-dandelion.md)
- [2022 问题](problems/2022-a-honeybee.md)
- [2021 问题](problems/2021-a-solar-storage.md)
- [2020 问题](problems/2020-a-summer-job.md)
- [评价 vs 仿真](comparisons/evaluation-vs-simulation.md)

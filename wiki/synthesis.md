# 跨年方法总论

十年题包共用「人写 Raw、Agent 写 Wiki」。定量结论必须能回到当年 `YYYY_P_slug/results/*.csv`。

```yaml
type: synthesis
year: cross
status: ingested
sources:
  - STUDIES.md
  - 2022_A_honeybee_dynamics/results/orchard_pollination_recommendation.csv
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
  - AGENTS.md
```

## 生态动力学 / 授粉运筹

2022 A 用可微五维蜂群（DFM–Torch）做弹性与 CCD 分岔，再在 20 英亩扁桃园上做蜂箱密度 OR。入口：[2022 问题](problems/2022-a-honeybee-dynamics.md)、[Autograd 弹性](methods/autograd-elasticity.md)、[果园 OR](methods/orchard-hive-or.md)。健康峰 \(66{,}507\) / 越冬 \(38{,}622\)；硬拐点 \(\mu_F^*=0.305\)；\(K^\star=40\)、YieldRatio \(0.848\)。

## 机理 / 空间扩散

2023 A 把冠毛羽流与 Lefkovitch 周步卷积在 1 公顷格网上，三气候对照后做三物种影响因子；日步中心宇宙与周步西缘分列。入口：[2023 问题](problems/2023-a-dandelion-prisms.md)。温带 12 月周步覆盖 **0.4511**；TAROF impact **0.1512** 只在三行宇宙内成立。气候 CSV 默认是情景。

## 评价 / MCDM

2024 A 把 IOC 纲领收成七叶，用锁定 AHP–SAW 做回测与布里斯班短名单，再用 N=6 LOOCV 作弱分类对照。入口：[2024 问题](problems/2024-a-olympic-sde.md)、[AHP](methods/ahp-eigenvalue.md)、[SAW](methods/saw-topsis.md)。2032 决策叙事：Flag football SAW 第一（0.5360）；青年权=0 时逆转。不要用 logistic 均值把 Cricket 写成执委会终裁。

## 仿真 / 搜救

2025 A 是图上多智能体清扫（规划基线 + 可选 RL）。方法与清场秒数只存在于 `2025_A_evacuation/` Raw；本页不抄疏散数字。跨年结构见 [评价 vs 仿真](comparisons/evaluation-vs-simulation.md)。

## 赛时

读 [index](index.md) → 复制 `_template/` 建新年目录 → 在新目录建模 → 当年 `prompts/cursor_log.md` 记 AI → 不改旧年论文交差。

## Related

- [目录](index.md)
- [2022 问题](problems/2022-a-honeybee-dynamics.md)
- [2023 问题](problems/2023-a-dandelion-prisms.md)
- [2024 问题](problems/2024-a-olympic-sde.md)
- [评价 vs 仿真](comparisons/evaluation-vs-simulation.md)

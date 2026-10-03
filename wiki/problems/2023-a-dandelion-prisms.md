# HiMCM 2023 A — Dandelion Prisms

2023 A 是机理/空间仿真题：西缘外侧一株绒球期蒲公英，在 1 公顷格网上做三气候扩散，并用锁定 SAW 算入侵影响因子。割草帕累托、日步伴生宇宙、PPO 扩展与 HOA 信是课程层；权威周步数字只认当前 `results/*.csv`。

```yaml
type: problem
year: 2023
status: ingested
sources:
  - 2023_A_dandelion_prisms/README.md
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2023_A_dandelion_prisms/results/impact_factors.csv
  - 2023_A_dandelion_prisms/results/bioeconomic_pareto.csv
  - 2023_A_dandelion_prisms/results/kpp_check.csv
  - 2023_A_dandelion_prisms/results/monthly_population_metrics.csv
  - 2023_A_dandelion_prisms/prompts/cursor_log.md
```

## 赛题要交什么

官方 PDF：COMAP HiMCM 2023 Problem A。

1. **Req 1 扩散**：1/2/3/6/12 月；温带 / 干旱 / 热带。权威表 `monthly_metrics.csv`。温带月末 `cover_frac`：0 / 0 / 0 / 0.0328 / **0.4511**；12 月 `front_m=99.5`，`n_adult≈4674`。
2. **官方 Req 2 影响因子**：三物种宇宙 min-max SAW。`impact_factors.csv`：PUMON **0.9333**，REJAP **0.5700**，TAROF **0.1512**（表内最低，不是「自然界无害」）。TAROF spread 叶来自温带覆盖 0.4511。
3. **课程扩展**：`bioeconomic_pareto.csv`、日步 `monthly_population_metrics.csv`（中心一株宇宙，day365 cover 100%，与周步 0.4511 分列）、`rl_policy_evaluation.csv`、HOA 信。

气候 CSV 默认是**情景强迫**；Open-Meteo 日表可下载但不自动覆盖权威周步结果。

## 题包地图

| 路径 | 用途 |
| --- | --- |
| `src/physics/` | WALD / 混合核羽流 |
| `src/biology/` | Lefkovitch、物候 |
| `src/simulation/` | 周步引擎 + 日步格网 |
| `src/theory/` | Fisher–KPP |
| `src/decision/` | 影响因子、割草帕累托、日步权衡 |
| `src/rl/` | PPO 课程扩展（非官方主问） |
| `src/scripts/` | 出表出图 |
| `results/` | 论文数字唯一来源 |
| `paper/` | 主文 / Summary / HOA |
| `prompts/cursor_log.md` | AI 披露底稿 |

## 研究做到哪一步

周步全表、KPP、影响因子、帕累托、日步伴生、论文套件与 PPO 评测表均已落地。**status=`ingested`，不是 contest-ready**：情景气候、三行影响宇宙、RL 未宣称优于全部田间基线。勿把旧 draft Wiki 的 0.2097 / TAROF=0 当作现行结果。

## Related

- [WALD 羽流](../methods/wald-plume.md)
- [Lefkovitch 自疏](../methods/lefkovitch-self-thinning.md)
- [Fisher–KPP](../methods/fisher-kpp.md)
- [割草帕累托](../methods/bioeconomic-pareto.md)
- [2023 研究闭环](../playbooks/2023-a-study-loop.md)
- [复现命令](../playbooks/2023-repro-sim.md)
- [月末指标](../sources/monthly-metrics.md)

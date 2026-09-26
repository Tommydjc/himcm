# 评价题 vs 仿真题

跨年对照：2024 是 MCDM + 小样本互证；2025 是图上多智能体搜救。两套数字不许互相填表。

```yaml
type: comparison
year: cross
status: ingested
sources:
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
  - 2024_A_olympic_sde/paper/himcm_paper.tex
  - 2025_A_evacuation/prompts/cursor_log.md
```

## 对照

| | 2023 A Dandelion | 2024 A SDE | 2025 A Evacuation |
| --- | --- | --- | --- |
| 题型 | 机理 / 空间扩散 + 影响因子 | 评价 / 规划附加运动 | 仿真 / 图上搜救清扫 |
| 主引擎 | WALD 卷积 + Lefkovitch | 锁定 AHP–SAW | 规划器（评分/解析界）+ 可选 RL |
| 权威数字 | `2023_A_dandelion_prisms/results/*.csv` | `2024_A_olympic_sde/results/*.csv` | `2025_A_evacuation/results/*.csv` |
| 本 Wiki 状态 | [problem](../problems/2023-a-dandelion-prisms.md) draft | [problem](../problems/2024-a-olympic-sde.md) ingested | 本题 ingest 未在本页写入清场秒数 |
| 典型陷阱 | 情景当观测；幼苗当开花株；三行 \(I=0\) 当无害 | 九行名次当 2032；LOOCV 当高精度 | 发现=救出；训练 return 当评测 |

2023 方法入口：[WALD](../methods/wald-plume.md)、[Lefkovitch](../methods/lefkovitch-self-thinning.md)。2024 方法入口：[AHP](../methods/ahp-eigenvalue.md)、[SAW](../methods/saw-topsis.md)。2025 方法页若尚未 ingest，查询时回题包 Raw，不要把 2024 的 0.5360 写进疏散，也不要把 2023 的 0.2097 写进奥运会排序。

## Related

- [总览 synthesis](../synthesis.md)
- [2023 问题](../problems/2023-a-dandelion-prisms.md)
- [2024 问题](../problems/2024-a-olympic-sde.md)

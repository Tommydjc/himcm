# Source：brisbane_2032_ranking.csv

三候选宇宙的锁定 SAW。这是 2032 短名单的决策表。ML 均值不得改写本表名次。

```yaml
type: source
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
  - 2024_A_olympic_sde/paper/himcm_paper.tex
  - 2024_A_olympic_sde/paper/ioc_letter.tex
```

## 摘录

| Rank | SDE_Name | Code | Role | Score |
| --- | --- | --- | --- | --- |
| 1 | Flag football | AFB | emerging_youth | 0.5360 |
| 2 | Cricket | CKT | traditional_host | 0.4842 |
| 3 | Squash | SQU | established_niche | 0.3960 |

CSV 原值：Flag `0.5360000001069045`，Cricket `0.48419999994346224`，Squash `0.3960000003719304`。

政策信建议在**承认青年权**的前提下以 Flag 为第一候选；青年权归零后的逆转见 [敏感性](../sources/sensitivity-weight-shock.md)。

## Related

- [OAT](../methods/oat-sensitivity.md)
- [两套宇宙](../concepts/two-universes.md)
- [ML 概率](../sources/ml-prediction-2032.md)
- [2024 问题](../problems/2024-a-olympic-sde.md)

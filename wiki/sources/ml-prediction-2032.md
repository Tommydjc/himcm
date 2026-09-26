# Source：ml_prediction_2032.csv

2032 三候选的 logistic / RF 入席概率。B=500、seed=42。区间跨过 0.5；tier 不是执委会投票。

```yaml
type: source
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/results/ml_prediction_2032.csv
  - 2024_A_olympic_sde/paper/sections/sec4_ml_validation.tex
```

## 摘录（论文四位）

| sde_name | prob_logistic | prob_rf | ci_lower | ci_upper | ml_tier |
| --- | --- | --- | --- | --- | --- |
| Cricket | 0.7984 | 0.6535 | 0.2689 | 0.9541 | high |
| Squash | 0.4215 | 0.5913 | 0.1152 | 0.7042 | low |
| Flag football | 0.6384 | 0.6079 | 0.1158 | 0.8994 | medium |

CSV 行序是 Cricket、Squash、Flag，不是 SAW 名次。logistic 均值第一是 Cricket；SAW 第一仍是 Flag。二者都把 Flag 放在 Squash 之上。

LOOCV 诊断见 [LOOCV](../methods/loocv-logistic.md)，不在本文件。逐折概率见 `results/ml_prediction_prob.csv`。

## Related

- [LOOCV](../methods/loocv-logistic.md)
- [AHP vs ML 权](../sources/ahp-vs-ml-weights.md)
- [2032 SAW](../sources/brisbane-2032-ranking.md)

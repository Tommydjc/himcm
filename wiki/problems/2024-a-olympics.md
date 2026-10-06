# HiMCM 2024 A — Olympic additional SDE

IOC 增设项目：锁定 AHP–SAW 排序 + N=6 LOOCV 逻辑回归作弱互证。决策引擎是 AHP–SAW，不是分类器。细表见已 ingest 的 [2024-a-olympic-sde](2024-a-olympic-sde.md)。

```yaml
type: problem
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
  - 2024_A_olympic_sde/results/historical_ranking.csv
  - 2024_A_olympic_sde/results/sensitivity_weight_shock.csv
  - 2024_A_olympic_sde/results/ml_prediction_2032.csv
  - 2024_A_olympic_sde/paper/himcm_paper.tex
```

## 赛题要交什么

1. 因子：IOC 六块叙述 → 代码七叶。
2. 历史九行回测。
3. 2032 三候选：Cricket / Squash / Flag football。
4. 权重敏感性。
5. 政策信。

## 2032 锁定 SAW（三行 min-max）

| SDE | Rank | SAW |
| --- | ---: | ---: |
| Flag football | 1 | 0.5360 |
| Cricket | 2 | 0.4842 |
| Squash | 3 | 0.3960 |

青年权 \(=0\) 会逆转名次（`sensitivity_weight_shock.csv`）。

## ML 透镜（不要写成终裁）

`ml_prediction_2032.csv` 逻辑回归均值：Cricket 0.798、Flag 0.638、Squash 0.422。区间很宽。Wiki 旧页记录 LOOCV Acc=0.3333、AUC=0.5000、Brier=0.2475。禁止用 ML 把 Cricket 写成执委会决定。

## Related

- [2024-a-olympic-sde 细页](2024-a-olympic-sde.md)
- [AHP](../methods/ahp-eigenvalue.md)
- [LOOCV logistic](../methods/loocv-logistic.md)
- [合成数据陷阱](../comparisons/synthetic-data-trap.md)

# Source：2020 model_comparison_metrics.csv

三维因子上 Softmax vs MLP 的 LOOCV。Top-1 两模型均为 0.20，多数类 0.22。

```yaml
type: source
year: 2020
status: ingested
sources:
  - 2020_A_summer_job_factor/results/model_comparison_metrics.csv
```

## 摘录

| 模型 | Top-1 | Top-3 |
| --- | ---: | ---: |
| Softmax | 0.20 | 0.38 |
| MLP | 0.20 | 0.48 |
| majority | 0.22 | — |

禁止写成竞赛模板 Top-3 92%。

## Related

- [合成数据陷阱](../comparisons/synthetic-data-trap.md)
- [2020 问题](../problems/2020-a-summer-job.md)

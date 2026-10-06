# Source：2020 factor_loadings.csv

50×15 合成李克特上 PC-EFA + Varimax 的 \(\Lambda^{\ast}\)。不是真实实地问卷。

```yaml
type: source
year: 2020
status: ingested
sources:
  - 2020_A_summer_job_factor/results/factor_loadings.csv
  - 2020_A_summer_job_factor/results/factor_interpretation.md
  - 2020_A_summer_job_factor/data/raw/DATA_SOURCE.txt
```

## 摘录

竞赛抽 \(m=3\)，累计方差 42.6%。F1 列领先 `safety_level`（约 +0.750）。完整列名与载荷以 CSV 为准。Kaiser \(\lambda>1\) 给出 6 个因子，见 `efa_engine` 日志，勿把 6 轴与 3 轴混表。

## Related

- [2020 问题](../problems/2020-a-summer-job.md)
- [EFA + LLMFactor](../methods/factor-analysis-llmfactor.md)

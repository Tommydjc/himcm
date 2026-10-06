# Synthetic data trap

`testrandomdata`、手搓 Likert、与官方题表列数不符的镜像，都不能写成「实地问卷」。provenance 写进 `DATA_SOURCE.txt` 与论文 Limitations。

```yaml
type: comparison
year: cross
status: ingested
sources:
  - 2020_A_summer_job_factor/data/raw/DATA_SOURCE.txt
  - 2020_A_summer_job_factor/data/raw/OriginalData.csv
  - 2020_A_summer_job_factor/results/model_comparison_metrics.csv
  - 2024_A_olympic_sde/results/historical_ranking.csv
```

## 对照

| 源 | 是什么 | 可写进主结论？ |
| --- | --- | --- |
| 2020 `OriginalData.csv` | 50×15 合成李克特，`SYNTHETIC_FALLBACK` | 否（只可作管线演示） |
| 2020 GitHub 7 列 | 400 行竞赛镜像，不是 15 项 | 可作对照，不可与 15 项混表 |
| BLS / O*NET | 职业统计，不是学生问卷 | 可作岗位特征，不可当 KMO 样本 |
| 2024 九行历史 | 真回测表 | 可以，但 N=6 LOOCV 弱 |

## 反模式

竞赛模板 KMO 0.782 / 方差 71.4% / Top-3 92%：**本仓 2020 实测拒绝**。实测 KMO 0.445、方差 42.6%、Top-1 0.20。

## Related

- [2020 暑期工](../problems/2020-a-summer-job.md)
- [EFA + LLMFactor](../methods/factor-analysis-llmfactor.md)
- [2024 奥运](../problems/2024-a-olympics.md)

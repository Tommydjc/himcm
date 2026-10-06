# HiMCM 2020 A — Summer job match

15 项 Likert 上的 EFA + LLMFactor 岗位坐标 + 熵权 + 3D K-Means + Softmax/MLP LOOCV + Streamlit。**15 列问卷是 SYNTHETIC_FALLBACK**，不是 HiMCM2020 GitHub 的 7 维表。

```yaml
type: problem
year: 2020
status: ingested
sources:
  - 2020_A_summer_job_factor/data/raw/DATA_SOURCE.txt
  - 2020_A_summer_job_factor/results/factor_loadings.csv
  - 2020_A_summer_job_factor/results/entropy_weights.csv
  - 2020_A_summer_job_factor/results/model_comparison_metrics.csv
  - 2020_A_summer_job_factor/results/cluster_centroids.csv
  - 2020_A_summer_job_factor/results/job_factor_matrix.csv
  - 2020_A_summer_job_factor/paper/himcm_paper.tex
```

## 赛题要交什么

1. 从问卷抽潜因子。
2. 给暑期岗位打分/排序。
3. 向学生解释（本包：Streamlit + 2 页指南）。

## 实测（禁止套模板假数）

| 量 | 值 | 不采用的模板 |
| --- | --- | --- |
| KMO | 0.445 | 0.782 |
| Bartlett | \(\chi^2=170.05\), df=105, \(p=6.10\times 10^{-5}\) | \(\chi^2=348.6\) |
| \(m=3\) 方差份额 | 42.6% | 71.4% |
| 熵权 \(w\) | (0.479, 0.319, 0.202) | (0.412, 0.354, 0.234) |
| LOOCV Top-1 | Softmax=MLP=0.20；多数类 0.22 | — |
| LOOCV Top-3 | 0.38 vs 0.48 | 92% vs 84% |

Kaiser \(\lambda>1\) 给出 6 个特征值 \(>1\)。竞赛建模仍抽 \(m=3\)。

抽出轴：F1 网络–技能（safety 0.750 领先，不是纯时薪）；F2 自主–可达；F3 负荷（体力正、上司公平负）。

GitHub 镜像：400×7，KMO 0.511，Bartlett \(p=0.052\)（`github_survey_suitability.txt`）。

## 题包地图

`src/factor_analysis/`，`src/llm_factor/`，`src/entropy_weight/`，`src/clustering/`，`src/classification/`，`src/web_app/`。

## Related

- [EFA + LLMFactor](../methods/factor-analysis-llmfactor.md)
- [熵权](../methods/entropy-weight-objective-scoring.md)
- [合成数据陷阱](../comparisons/synthetic-data-trap.md)
- [ASD-STE100 写作](../playbooks/asd-ste100-academic-writing.md)

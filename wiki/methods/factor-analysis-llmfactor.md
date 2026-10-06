# EFA / Varimax and LLMFactor

Pearson \(R\) 上主成分抽取 + Kaiser 正规化 Varimax，再用本地 LLM 给 \(\Lambda^{\ast}\) 和岗位文本命名/打分。代码入口：`2020_A_summer_job_factor/src/factor_analysis/` 与 `src/llm_factor/`。

```yaml
type: method
year: 2020
status: ingested
sources:
  - 2020_A_summer_job_factor/results/factor_loadings.csv
  - 2020_A_summer_job_factor/results/factor_interpretation.md
  - 2020_A_summer_job_factor/results/job_factor_matrix.csv
  - 2020_A_summer_job_factor/src/factor_analysis/efa_engine.py
  - 2020_A_summer_job_factor/src/llm_factor/llm_factor_agent.py
```

## 适用

\(N\) 中、\(p\) 中的李克特相关阵，且已做 KMO / Bartlett。本包 KMO=0.445，仍抽 \(m=3\) 是竞赛设计，不是适宜性通过。

## 公式

未旋转 \(\Lambda=V_m\mathrm{diag}(\sqrt{\lambda_m})\)。
\(\Lambda^{\ast}=\Lambda T\)，\(T^{\top}T=I\)。
Thompson：\(W=R^{-1}\Lambda^{\ast}\)，\(F=ZW\)。

本包 \(m=3\) 占 trace 42.6%。F1 领先项是 `safety_level`（+0.750），不是 hourly wage。

## LLMFactor

Ollama `qwen2.5:7b` 读每列 \(|\lambda^{\ast}|\) 前 4 名。岗位 8 份 txt → \([-3,3]^3\)。Ollama 失败则关键词投影 mock。禁止把竞赛「时薪/履历/日晒」三元组硬贴到错位的列。

## 失败模式

KMO<0.70 仍强行三因子叙事；用设计名代替载荷表。

## Related

- [2020 暑期工](../problems/2020-a-summer-job.md)
- [熵权](entropy-weight-objective-scoring.md)
- [合成数据陷阱](../comparisons/synthetic-data-trap.md)

# Source：ahp_vs_ml_weights.csv

六叶 AHP（重新归一）对 logistic \(|\beta|\)（L1）。Difference \(=w^{\mathrm{ML}}-w^{\mathrm{AHP}}\)。

```yaml
type: source
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/results/ahp_vs_ml_weights.csv
  - 2024_A_olympic_sde/paper/sections/sec4_ml_validation.tex
  - 2024_A_olympic_sde/src/ml/plot_and_report.py
```

## 摘录（论文四位）

| Criterion | AHP | ML | Difference | reading |
| --- | --- | --- | --- | --- |
| GLOBAL_REACH | 0.2000 | 0.1295 | −0.0705 | AHP_higher_than_ML |
| GENDER_PARITY | 0.1778 | 0.0000 | −0.1778 | AHP_higher_than_ML |
| YOUTH_APPEAL | 0.2444 | 0.1285 | −0.1159 | AHP_higher_than_ML |
| INFRA_COST | 0.1333 | 0.2813 | 0.1480 | ML_higher_than_AHP |
| BROADCAST_VAL | 0.1556 | 0.4113 | 0.2557 | ML_higher_than_AHP |
| Events_Count | 0.0889 | 0.0494 | −0.0395 | AHP_higher_than_ML |

CSV 无余弦列。用上述两列权向量做点积余弦得 **0.7193**，与 `sec4_ml_validation.tex` / `plot_and_report.py` 一致。

## Related

- [AHP vs |β| 方法](../methods/ahp-vs-beta.md)
- [LOOCV](../methods/loocv-logistic.md)

# AHP 主观权与 logistic |β|

在六叶 IOC 特征上，把锁定 AHP 权重新 L1 归一，与全样本标准化 L2 logistic 的 \(|\beta|\) L1 归一对照。\(\Delta w_j=w_j^{\mathrm{ML}}-w_j^{\mathrm{AHP}}\)。

```yaml
type: method
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/src/ml/importance.py
  - 2024_A_olympic_sde/results/ahp_vs_ml_weights.csv
  - 2024_A_olympic_sde/paper/sections/sec4_ml_validation.tex
  - 2024_A_olympic_sde/tests/test_importance.py
```

## 对齐规则

`IOC_FEATURE_COLS`：GLOBAL_REACH, GENDER_PARITY, YOUTH_APPEAL, INFRA_COST, BROADCAST_VAL, Events_Count。丢掉截距与惯性两列。Appearances 不在这六列中。

六名标签运动的 `GENDER_PARITY` 均为 50%，故该叶 ML 权重为 0（`ahp_vs_ml_weights.csv`）。

由 CSV 两列权向量计算的余弦为 0.7193（与 `sec4_ml_validation.tex` 一致）。

最大正缺口：BROADCAST_VAL \(\Delta w=+0.2557\)（ML 更高）。TeX 六叶表（四位）：

| Criterion | AHP | ML | \(\Delta w\) |
| --- | --- | --- | --- |
| GLOBAL_REACH | 0.2000 | 0.1295 | −0.0705 |
| GENDER_PARITY | 0.1778 | 0.0000 | −0.1778 |
| YOUTH_APPEAL | 0.2444 | 0.1285 | −0.1159 |
| INFRA_COST | 0.1333 | 0.2813 | 0.1480 |
| BROADCAST_VAL | 0.1556 | 0.4113 | 0.2557 |
| Events_Count | 0.0889 | 0.0494 | −0.0395 |

互证只是部分一致：SAW 金牌 Flag；logistic 均值第一 Cricket。政策建议仍跟 AHP–SAW 与青年权敏感性。

## 代码入口

```bash
cd 2024_A_olympic_sde
PYTHONPATH=. python -m src.ml.importance
```

## Related

- [AHP 特征值](../methods/ahp-eigenvalue.md)
- [LOOCV](../methods/loocv-logistic.md)
- [权重对照 source](../sources/ahp-vs-ml-weights.md)

# AHP 特征值权重

对正互反判断矩阵求 Perron 根对应的归一化特征向量，再用 \(CR=CI/RI\) 做一致性检验。本包锁定矩阵由优先向量外积生成，故 \(CR=0\)。

```yaml
type: method
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/src/mcda/ahp.py
  - 2024_A_olympic_sde/data/raw/ahp_judgment_locked.csv
  - 2024_A_olympic_sde/data/raw/README_AHP.md
  - 2024_A_olympic_sde/paper/himcm_paper.tex
  - 2024_A_olympic_sde/tests/test_ahp_cr.py
```

## 公式与适用条件

判断矩阵 \(A=(a_{ij})\) 须 \(a_{ij}>0\)、\(a_{ii}=1\)、\(a_{ji}=1/a_{ij}\)。权重 \(w\) 是最大实特征对（Perron）的归一化向量，**不用**列和归一化近似。

\[
CI=\frac{\lambda_{\max}-n}{n-1},\qquad CR=\frac{CI}{RI(n)}.
\]

`SAATY_RI` 覆盖 \(n=1..12\)；通过阈 `CR_THRESHOLD=0.10`。不一致时 `warnings.warn` 且 `is_consistent=False`，不静默改矩阵。

锁定叶权重（`README_AHP.md` 与论文）：

\[
w=(0.10,\,0.08,\,0.18,\,0.16,\,0.22,\,0.12,\,0.14)
\]

顺序：Appearances, Events_Count, GLOBAL_REACH, GENDER_PARITY, YOUTH_APPEAL, INFRA_COST, BROADCAST_VAL。\(a_{ij}=w_i/w_j\)，故完全一致。

`HOST_POPULARITY` 不在七准则内。

## 代码入口

- `2024_A_olympic_sde/src/mcda/ahp.py`：`compute_ahp_weights`
- 矩阵：`2024_A_olympic_sde/data/raw/ahp_judgment_locked.csv`
- 测试：`PYTHONPATH=. python -m pytest tests/test_ahp_cr.py -q`

## 失败模式

- 用列和近似代替精确主特征对。
- 用 `np.random` 填判断矩阵当「专家打分」。
- 把 `HOST_POPULARITY` 塞进 AHP 七叶。
- 把六叶对照表（`ahp_vs_ml_weights.csv` 已重新归一、且无 Appearances）误当成七叶锁定向量。

## Related

- [SAW / TOPSIS](../methods/saw-topsis.md)
- [AHP vs |β|](../methods/ahp-vs-beta.md)
- [IOC 六块与七叶](../concepts/ioc-six-plus-xlsx.md)
- [2024 失败模式](../concepts/2024-a-failure-modes.md)

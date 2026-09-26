# 小样本 LOOCV Logistic

在 6 条有标签历史行上做留一交叉验证，并对 2032 三候选做 B=500 分层自助区间。这是弱分类器、强对照：用来读入席概率与 \(|\beta|\)，不是替换 AHP–SAW。

```yaml
type: method
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/src/ml/dataset.py
  - 2024_A_olympic_sde/src/ml/classifier.py
  - 2024_A_olympic_sde/results/ml_prediction_2032.csv
  - 2024_A_olympic_sde/results/ml_prediction_prob.csv
  - 2024_A_olympic_sde/paper/sections/sec4_ml_validation.tex
  - 2024_A_olympic_sde/tests/test_classifier.py
  - 2024_A_olympic_sde/tests/test_dataset.py
```

## 标签与泄漏约束

N=6：ATH / SWM / SKB / SRF 为 y=1，BKG / KTE 为 y=0。2032 候选不进训练。Cricket / Squash / Flag 的惯性列强制为 0。BSB/SBL 无 IOC 六列，不编造后并入 X。

禁止 80/20。`StandardScaler` 只在每一折训练集上 `fit`。

Logistic：C=0.8、liblinear、L2。RF 为对照（depth=3、80 树，`random_state=0`）。自助法 `random_state=42`，B=500。

## 实测（禁止改写成「高精度」）

`sec4_ml_validation.tex`：accuracy \(=0.3333\)，ROC-AUC \(=0.5000\)，Brier \(=0.2475\)。常数 \(\hat p=0.5\) 的 Brier 为 0.25，故接近无信息基线。

2032 logistic 均值（四位，同 TeX / CSV）：

| SDE | \(P_{\mathrm{logit}}\) | CI low | CI high | tier |
| --- | --- | --- | --- | --- |
| Cricket | 0.7984 | 0.2689 | 0.9541 | high |
| Flag football | 0.6384 | 0.1158 | 0.8994 | medium |
| Squash | 0.4215 | 0.1152 | 0.7042 | low |

三个区间都跨过 0.5。ML 均值把 Cricket 排第一，**不能**据此改写 SAW 的 Flag 第一。

## 代码入口

```bash
cd 2024_A_olympic_sde
PYTHONPATH=. python -m src.ml.classifier
PYTHONPATH=. python -m src.ml
```

## Related

- [AHP vs |β|](../methods/ahp-vs-beta.md)
- [2032 概率 source](../sources/ml-prediction-2032.md)
- [失败模式](../concepts/2024-a-failure-modes.md)
- [COMAP 披露](../concepts/comap-ai-disclosure.md)

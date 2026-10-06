# Entropy-weight objective scoring

对已得分的矩阵做列 min-max、列份额 \(p_{ij}\)、熵 \(E_j\)、\(D_j=1-E_j\)、\(w=D/\sum D\)。禁止手写常数权。2020 代码：`2020_A_summer_job_factor/src/entropy_weight/entropy_engine.py`。

```yaml
type: method
year: 2020
status: ingested
sources:
  - 2020_A_summer_job_factor/results/entropy_weights.csv
  - 2020_A_summer_job_factor/results/job_entropy_utility.csv
  - 2020_A_summer_job_factor/src/entropy_weight/entropy_engine.py
```

## 公式

\[
y_{ij}=\frac{f_{ij}-\min f_j}{\max f_j-\min f_j}+\varepsilon,\quad
E_j=-\frac{1}{\ln N}\sum_i p_{ij}\ln p_{ij}.
\]

本包 \(\varepsilon=10^{-4}\)，\(N=50\)，\(w=(0.479,0.319,0.202)\)。F3 仅在效用 \(U=w_1F_1+w_2F_2-w_3F_3\) 取负。Tutor \(U=+0.213\) 第一；Lifeguard \(U=-0.681\) 第八。

GitHub 原 `entropy.py` 在 7 列、无 \(\varepsilon\)、无成本符号上加权。不要混用两套 \(w\)。

## 失败模式

AHP 拍脑袋权冒充熵权；先翻转成本列再算熵却不声明 \(p\) 会变。

## Related

- [2020 暑期工](../problems/2020-a-summer-job.md)
- [EFA + LLMFactor](factor-analysis-llmfactor.md)

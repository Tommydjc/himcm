# Autograd 弹性排名

在展开的 365 日计算图上对叶子参数求 \(\partial J/\partial\theta\)，并报告弹性 \(S_\theta=(\theta/J)(\partial J/\partial\theta)\)。

```yaml
type: method
year: 2022
status: ingested
sources:
  - 2022_A_honeybee_dynamics/src/sensitivity/autograd_sens.py
  - 2022_A_honeybee_dynamics/results/autograd_elasticity_ranking.csv
```

## 目标

- \(J_{\mathrm{pop}}=H+F\) @ day 365（表内 \(J^\star=35{,}179\)）
- \(J_{\mathrm{nectar}}=N\) @ day 270（表内 \(J^\star=106{,}745\)）

## 实测 Top（\(J_{\mathrm{pop}}\)）

| 参数 | \(\theta\) | \(S\) |
| --- | ---: | ---: |
| `laying_rate_max` | 1600 | \(+0.925\) |
| `forager_mortality` | 0.14 | \(-0.311\) |
| `social_inhibition_k` | 10 | \(+0.294\) |

\(J_{\mathrm{nectar}}\) 首位为 `nectar_intake_rate`，\(S=+0.829\)。完整表：`autograd_elasticity_ranking.csv`。

## 失败模式

- 用有限差分手填弹性却声称 Autograd。
- 把峰值日附近的局部尖峰梯度误报为全年弹性主序。

## Related

- [DFM Torch](../methods/dfm-torch-sim.md)
- [弹性 source](../sources/autograd-elasticity-ranking.md)
- [2022 问题](../problems/2022-a-honeybee-dynamics.md)

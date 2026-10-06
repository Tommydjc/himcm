# PyTorch autograd sensitivity

把 ODE / 日步仿真展开成计算图，用反向模式求 \(\partial J/\partial\theta\)，避免有限差分步长。2022 入口：`2022_A_honeybee_dynamics/src/sensitivity/autograd_sens.py`。更短的弹性表说明见 [autograd-elasticity](autograd-elasticity.md)。

```yaml
type: method
year: 2022
status: ingested
sources:
  - 2022_A_honeybee_dynamics/results/autograd_elasticity_ranking.csv
  - 2022_A_honeybee_dynamics/src/autograd_engine/torch_sim.py
  - 2022_A_honeybee_dynamics/src/sensitivity/autograd_sens.py
```

## 适用

参数 \(\theta\) 进入全年可微动态；\(J\) 是年末或花期标量。

## 弹性

\[
S_\theta=\frac{\theta}{J}\frac{\partial J}{\partial\theta}.
\]

\(J_{\mathrm{pop}}\) 实测 Top：`laying_rate_max` \(+0.925\)，`forager_mortality` \(-0.311\)（`autograd_elasticity_ranking.csv`）。

## 失败模式

对不可微分支（硬截断、离散事件）直接 `.backward()`；混用差分与 Autograd 两套表不声明协议。

## Related

- [2022 蜂群](../problems/2022-a-honeybee.md)
- [DFM Torch](dfm-torch-sim.md)
- [autograd-elasticity](autograd-elasticity.md)

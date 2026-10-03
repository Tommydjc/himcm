# DFM 可微蜂群仿真

五维状态 \(y=[B,H,F,P,N]\)（育虫、内勤、外勤、花粉克、蜜克）在 PyTorch 上按日展开，供 Autograd 与胁迫实验共用同一前向。

```yaml
type: method
year: 2022
status: ingested
sources:
  - 2022_A_honeybee_dynamics/src/autograd_engine/torch_sim.py
  - 2022_A_honeybee_dynamics/results/ccd_healthy_vs_stress_snapshot.csv
```

## 公式要点

叶子参数：\(\{L_{\max},\mu_F,\eta_P,\eta_N,\sigma_F\}\)。激活为 \(C^\infty\)：Holling-III、Softplus 投影、季节波 \(\Omega(t)\)。胁迫经 `configure_stress` 写入缓冲，不改叶子身份。

健康基线快照（`ccd_healthy_vs_stress_snapshot.csv`，`stress_label=healthy_baseline`）：\(\mu_F=0.10\)，夏季峰 \(H+F=66{,}507\)，越冬 \(38{,}622\)。

## 代码入口

```bash
cd 2022_A_honeybee_dynamics
PYTHONPATH=. python -c "from src.autograd_engine.torch_sim import DifferentiableColonySimulator; print('ok')"
```

## 失败模式

- 把早期 `colony_annual_dynamics.csv`（成人峰约 \(27{,}546\)）与 Torch 健康峰混成一条「官方轨迹」。
- 用硬 `clamp` 替代 Softplus，导致边界梯度为零却声称做过全局灵敏度。

## Related

- [2022 问题](../problems/2022-a-honeybee-dynamics.md)
- [Autograd 弹性](../methods/autograd-elasticity.md)

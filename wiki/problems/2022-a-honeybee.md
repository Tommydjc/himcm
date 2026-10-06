# HiMCM 2022 A — Honeybee dynamics

可微五维蜂群（DFM–Torch）+ Autograd 弹性 + CCD 分岔 + 20 英亩授粉 OR。细表见 [2022-a-honeybee-dynamics](2022-a-honeybee-dynamics.md)。

```yaml
type: problem
year: 2022
status: ingested
sources:
  - 2022_A_honeybee_dynamics/results/ccd_healthy_vs_stress_snapshot.csv
  - 2022_A_honeybee_dynamics/results/autograd_elasticity_ranking.csv
  - 2022_A_honeybee_dynamics/results/ccd_tipping_point.csv
  - 2022_A_honeybee_dynamics/results/orchard_pollination_recommendation.csv
```

## 赛题要交什么

1. 全年蜂口与储蜜。
2. 参数灵敏度。
3. \(81{,}000\,\mathrm{m}^2\) 地块配箱。
4. 非技术咨询信。

## 锚点

健康峰 \(H+F=66{,}507\)（day 238）、越冬 \(38{,}622\)。
\(J_{\mathrm{pop}}\) 弹性：`laying_rate_max` \(+0.925\)，`forager_mortality` \(-0.311\)。
CCD 软拐点 \(\mu_F^*=0.11\)，硬拐点 \(0.305\,\mathrm{day}^{-1}\)。
果园 \(K^\star=40\)，YieldRatio \(0.848\)。

禁止把早期 Khoury 三分室 CSV 与 Torch 健康峰写成同一轨迹。

## Related

- [2022-a-honeybee-dynamics 细页](2022-a-honeybee-dynamics.md)
- [Autograd 灵敏度](../methods/pytorch-autograd-sensitivity.md)
- [DFM Torch](../methods/dfm-torch-sim.md)
- [Notebook vs 模块](../comparisons/notebook-hell-vs-modular.md)

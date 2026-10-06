# HiMCM 2023 A — Dandelion spread

西缘一株绒球期蒲公英：WALD / 混合核羽流 + 周步 Lefkovitch 卷积格网 + Fisher–KPP 对照。细数字见 [2023-a-dandelion-prisms](2023-a-dandelion-prisms.md)。

```yaml
type: problem
year: 2023
status: ingested
sources:
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2023_A_dandelion_prisms/results/impact_factors.csv
  - 2023_A_dandelion_prisms/results/kpp_check.csv
  - 2023_A_dandelion_prisms/results/bioeconomic_pareto.csv
```

## 赛题要交什么

1. 1/2/3/6/12 月、三气候覆盖与锋面。
2. 三物种影响因子（SAW）。
3. 课程层：割草帕累托、日步宇宙、PPO（非官方主问）。

## 周步锚点

温带 12 月 `cover_frac` **0.4511**，`front_m=99.5`。
影响因子：PUMON 0.9333，REJAP 0.5700，TAROF 0.1512。
KPP 相对误差见 `kpp_check.csv`（旧 Wiki 记 0.0521）。
日步 365 日 cover 100% 与周步 0.4511 **分列**，禁止平均。

## Related

- [2023-a-dandelion-prisms 细页](2023-a-dandelion-prisms.md)
- [WALD](../methods/wald-plume.md)
- [Conv2D CA](../methods/conv2d-spatiotemporal-ca.md)
- [Fisher–KPP](../methods/fisher-kpp.md)

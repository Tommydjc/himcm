# ccd_tipping_point.csv

三重胁迫下 \(\mu_F\) 软/硬拐点一行表。

```yaml
type: source
year: 2022
status: ingested
sources:
  - 2022_A_honeybee_dynamics/results/ccd_tipping_point.csv
  - 2022_A_honeybee_dynamics/results/ccd_bifurcation_sweep.csv
```

## 摘要

- `mu_star_soft=0.11`（离开 Safe，`n_safe_line=5000`）
- `mu_star_threshold=0.305`（硬崩溃，`n_collapse_line=3000`，`n_at_threshold≈2978`）

完整曲线见 `ccd_bifurcation_sweep.csv`（65 行）。

## Related

- [CCD 方法](../methods/ccd-bifurcation.md)
- [2022 问题](../problems/2022-a-honeybee-dynamics.md)

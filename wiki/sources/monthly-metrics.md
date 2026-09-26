# monthly_metrics.csv

三气候基线的月末覆盖、株数与锋面。只摘要，不另造表。

```yaml
type: source
year: 2023
status: draft
sources:
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2023_A_dandelion_prisms/results/drought_monthly.csv
```

## 摘要

温带 1/2/3/6/12 月 `cover_frac`：0.0000 / 0.0027 / 0.0052 / 0.0729 / 0.2097；12 月 `front_m=46.5`，`n_adult=3443.81`。
干旱 12 月覆盖 0.0273、锋面 22.5 m。
热带 12 月覆盖 0.0064、锋面 3.5 m。
温带干旱扰动（`drought_monthly.csv`）12 月覆盖 0.1156、锋面 33.5 m。

`n_plants` 含幼苗。覆盖阈值为 0.05 株/m²。

## Related

- [2023 问题](../problems/2023-a-dandelion-prisms.md)
- [Lefkovitch](../methods/lefkovitch-self-thinning.md)

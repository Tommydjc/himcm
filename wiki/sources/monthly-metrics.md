# monthly_metrics.csv

三气候基线的月末覆盖、株数与锋面。只摘要，不另造表。

```yaml
type: source
year: 2023
status: ingested
sources:
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2023_A_dandelion_prisms/results/drought_monthly.csv
```

## 摘要（baseline）

温带 1/2/3/6/12 月 `cover_frac`：0.0000 / 0.0000 / 0.0000 / 0.0328 / **0.4511**；12 月 `front_m=99.5`，`n_plants≈4.414\times 10^5`，`n_adult≈4674`。

干旱 12 月：`cover_frac=0.0065`，`front_m=13.5`。

热带 12 月：`cover_frac=0.0232`，`front_m=10.5`。

温带干旱扰动（`drought_monthly.csv`）12 月 `cover_frac=0.2046`。

`n_plants` 含幼苗。日步中心宇宙见 `monthly_population_metrics.csv`（勿与本表平均）。

## Related

- [2023 问题](../problems/2023-a-dandelion-prisms.md)
- [Lefkovitch](../methods/lefkovitch-self-thinning.md)

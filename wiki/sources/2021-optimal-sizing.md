# Source：2021 optimal_sizing_solution.csv

\$10k 预算行：3× FREEDOH + 1× lead-carbon，capex \$9{,}700，LPSP 0.2489，年停电小时 2725。

```yaml
type: source
year: 2021
status: ingested
sources:
  - 2021_A_solar_storage/results/optimal_sizing_solution.csv
  - 2021_A_solar_storage/results/lcos_emerging_tech.csv
```

## 摘录

Tesla Powerwall+ 该行 `units=0`。LCOS 另表：Na-ion 0.161 \$/kWh（\(r=0.07\), \(N=20\)）。与早期论文草稿冲突时以本 CSV 为准。

## Related

- [2021 问题](../problems/2021-a-solar-storage.md)
- [被动停电谬误](../comparisons/passive-blackout-fallacy.md)

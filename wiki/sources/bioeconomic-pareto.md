# bioeconomic_pareto.csv

温带割草网格的 12 月覆盖与相对成本。

```yaml
type: source
year: 2023
status: ingested
sources:
  - 2023_A_dandelion_prisms/results/bioeconomic_pareto.csv
```

## 摘要

22 条政策。不割：`cover_frac=0.4511`，`cost=0`，`on_front=True`。

双周 \(\eta=1\)（`interval_weeks=2`）：`cover_frac=0.0064`，`cost=39`，`on_front=True`。

每周 \(\eta=1\)（`interval_weeks=1`）：`cover_frac=0.0064`，`cost=78`，`on_front=False`（被支配）。

日步四策略权衡在 `bioeconomic_tradeoff.csv`，勿与本周步表混写成同一帕累托前沿。

## Related

- [帕累托方法](../methods/bioeconomic-pareto.md)
- [2023 问题](../problems/2023-a-dandelion-prisms.md)

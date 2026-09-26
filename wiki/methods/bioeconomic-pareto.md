# 割草双目标帕累托

温带一年上最小化 \((\mathrm{cover}_{12},C)\)，\(C=n(1+\eta/2)\)。

```yaml
type: method
year: 2023
status: draft
sources:
  - 2023_A_dandelion_prisms/src/decision/pareto.py
  - 2023_A_dandelion_prisms/results/bioeconomic_pareto.csv
```

## 要点

- 不割：覆盖 0.2097，成本 0，在前沿上。
- 双周 \(\eta=1\)：覆盖 0.0077，成本 39，在前沿上。
- 每周 \(\eta=1\)：覆盖同为 0.0077，成本 78，被支配。

课程扩展，不是官方 Req 2。官方 Req 2 是影响因子 SAW。

## Related

- [2023 问题](../problems/2023-a-dandelion-prisms.md)
- [帕累托 source](../sources/bioeconomic-pareto.md)

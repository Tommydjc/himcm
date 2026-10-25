# Source：harness_benchmark_results.csv

五扰动 × 两控制器黑盒跑分；均值见 companion JSON。

```yaml
type: source
year: 2018
status: ingested
sources:
  - 2018_B_smart_house_mpc/results/harness_benchmark_results.csv
  - 2018_B_smart_house_mpc/results/harness_benchmark_summary.json
  - 2018_B_smart_house_mpc/paper_figures/fig_harness_radar_benchmark.png
```

## 摘录（行级）

| Ctrl | Scen | M1 \$ | M2 °C·h | M3 |
| --- | --- | ---: | ---: | ---: |
| bang | A | 1.682 | 6.174 | 42 |
| mpc | A | 1.611 | 5.669 | 3 |
| bang | B | 5.549 | 18.043 | 130 |
| mpc | B | 5.347 | 16.979 | 7 |
| bang | D | 1.720 | 0.538 | 133 |
| mpc | D | 1.662 | 8.321 | 16 |

均值：Bang \(M_1=2.427\) / \(M_3=77.6\)；MPC \(M_1=2.340\) / \(M_3=6.4\)。\(M_4\)：Bang 0.865，MPC 0.874。

## Related

- [Harness 方法](../methods/black-box-climate-harness.md)
- [2018 问题](../problems/2018-b-smart-house.md)

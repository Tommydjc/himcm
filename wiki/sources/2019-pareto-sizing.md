# Source：optimal_kiosk_sizing.csv

Rush 日网格仿真 + 非支配标记；膝点 `is_knee_80k=True`。

```yaml
type: source
year: 2019
status: ingested
sources:
  - 2019_A_charge_des_agent/results/optimal_kiosk_sizing.csv
  - 2019_A_charge_des_agent/paper_figures/fig_pareto_cost_vs_wait.png
```

## 摘录

膝点行：`8,16,0`，`total_cost=35200`，`total_area=38.4`，`mean_wait_time≈0.0115`，`reneging_rate=0`，`is_pareto=True`，`is_knee_80k=True`。  
最便宜 Pareto：`4,8,0`，`mean_wait_time≈5.91`，`reneging_rate≈0.0457`。

## Related

- [Pareto 方法](../methods/pareto-kiosk-sizing.md)
- [2019 问题](../problems/2019-a-airport-charging.md)

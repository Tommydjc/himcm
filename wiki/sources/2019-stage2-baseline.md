# Source：stage2_baseline_metrics.csv

Stage-2 经验包 `c_fast=12, c_slow=15, c_bank=2` 的 normal 日 KPI。

```yaml
type: source
year: 2019
status: ingested
sources:
  - 2019_A_charge_des_agent/results/stage2_baseline_metrics.csv
  - 2019_A_charge_des_agent/results/stage2_baseline_simulation.csv
  - 2019_A_charge_des_agent/paper_figures/stage2_queue_baseline.png
```

## 摘录

`total_passengers=441`，`anxious_passengers=110`，`mean_wait_time=0.0`，`reneging_rate=0.0`，`utilization_fast≈0.0572`，`utilization_slow≈0.2424`，`max_system_load=17`，`scenario=normal`，`horizon_min=960`。

## Related

- [2019 问题](../problems/2019-a-airport-charging.md)
- [BDI–DES](../methods/bdi-des-airport-charging.md)

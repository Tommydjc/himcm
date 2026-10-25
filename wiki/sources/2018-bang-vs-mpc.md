# Source：bang_vs_mpc_summary.json

Teaching 黄金日 Bang-Bang vs 预测矩阵 MPC 锁定对照。

```yaml
type: source
year: 2018
status: ingested
sources:
  - 2018_B_smart_house_mpc/results/bang_vs_mpc_summary.json
  - 2018_B_smart_house_mpc/results/baseline_bang_bang_summary.json
  - 2018_B_smart_house_mpc/results/mpc_optimal_summary.json
  - 2018_B_smart_house_mpc/paper_figures/fig_bang_vs_mpc_day.png
```

## 摘录

Bang：energy 6.232 kWh，cost \$1.682，peak_energy 1.514 kWh，peak_cost_share 0.432，starts 42。  
MPC：energy 6.223 kWh，cost \$1.611，peak_energy 1.274 kWh，peak_cost_share 0.380，starts 3；`mean_u_prepeak=0.401`，`mean_u_peak=0.168`。  
相对：cost_savings_pct 4.18；peak_energy_savings_pct 15.80；short_cycle_reduction 39。

## Related

- [2018 问题](../problems/2018-b-smart-house.md)
- [MPC 方法](../methods/2r2c-prediction-matrix-mpc.md)

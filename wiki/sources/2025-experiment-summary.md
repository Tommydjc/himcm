# Source：2025 experiment_summary.csv

规划器多情景 \(t_{\mathrm{clear}}\)。office 2 智能体 clear 模式 135；smoke 行大量失败。

```yaml
type: source
year: 2025
status: ingested
sources:
  - 2025_A_evacuation/results/experiment_summary.csv
  - 2025_A_evacuation/results/rl_office_eval.csv
```

## 摘录

`rl_office_eval.csv`：greedy，80 步，`n_cleared=2`，`all_clear=0`。不是 GAT–PPO 优于规划器的证据。

## Related

- [2025 问题](../problems/2025-a-evacuation.md)
- [评价 vs 仿真](../comparisons/evaluation-vs-simulation.md)

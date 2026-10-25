# Source：thunderstorm_stress_comparison.csv

17:00 起 40% 旅客延误 2–4 h + SOC 加倍消耗；同 n=621 对照经验包与膝点包。

```yaml
type: source
year: 2019
status: ingested
sources:
  - 2019_A_charge_des_agent/results/thunderstorm_stress_comparison.csv
  - 2019_A_charge_des_agent/paper_figures/fig_queue_length_delay_stress.png
```

## 摘录

| label | CapEx | mean_wait | reneging |
| --- | ---: | ---: | ---: |
| baseline_empirical_12_15_2 | 49000 | 4.591 | 0.0229 |
| pareto_knee (8/16/0) | 35200 | 2.347 | 0.0114 |

相对降幅（论文锁定叙述）：等待 −48.9%；弃排 −50.3%；CapEx −28.2%。

## Related

- [Pareto 方法](../methods/pareto-kiosk-sizing.md)
- [2019 问题](../problems/2019-a-airport-charging.md)

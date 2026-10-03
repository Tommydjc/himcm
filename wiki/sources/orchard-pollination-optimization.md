# orchard_pollination_optimization.csv

20 英亩扁桃园 \(K=5..60\) 扫描；推荐行在 companion 表。

```yaml
type: source
year: 2022
status: ingested
sources:
  - 2022_A_honeybee_dynamics/results/orchard_pollination_optimization.csv
  - 2022_A_honeybee_dynamics/results/orchard_pollination_recommendation.csv
```

## 摘要

| \(K\) | hives/acre | yield_ratio | net_profit_usd | hive_health |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 1.00 | 0.7368 | 47579 | 0.975 |
| 40 | 2.00 | 0.8485 | 51394 | 0.975 |
| 60 | 3.00 | 0.8547 | 47826 | 0.735 |

推荐表：`K_star_golden=40`，`yield_at_golden=0.848485`，`net_star_golden=51393.94`，`K_star_global=42`。

## Related

- [果园 OR 方法](../methods/orchard-hive-or.md)
- [2022 问题](../problems/2022-a-honeybee-dynamics.md)

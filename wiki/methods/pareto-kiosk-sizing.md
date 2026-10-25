# CapEx–Wait Pareto 充电亭选型

在面积与预算约束下枚举 (c_fast, c_slow, c_bank)，对 rush 日 DES 取非支配前沿；再在「弃排≤5%」可行集上选 \$80k 膝点。

```yaml
type: method
year: 2019
status: ingested
sources:
  - 2019_A_charge_des_agent/src/optimization/pareto_sizing.py
  - 2019_A_charge_des_agent/results/optimal_kiosk_sizing.csv
  - 2019_A_charge_des_agent/results/thunderstorm_stress_comparison.csv
  - 2019_A_charge_des_agent/paper_figures/fig_pareto_cost_vs_wait.png
  - 2019_A_charge_des_agent/paper_figures/fig_queue_length_delay_stress.png
```

## 约束与目标

- CapEx ≤ \$80{,}000；面积 ≤ 40 m²。  
- 双目标：最小化总造价与均值等待。  
- 服务过滤：reneging_rate ≤ 0.05（否则膝点塌到不可服务包如 4/0/0）。

## 锁定结果

- 网格约 216 个唯一包；非支配点 7 个。  
- **膝点 8/16/0**：\$35{,}200，38.4 m²；rush 等待 0.0115 min，弃排 0。  
- 最便宜 Pareto：4/8/0，等待 5.91 min。  
- 雷暴日相对经验包 12/15/2：膝点等待 −48.9%，弃排 −50.3%，CapEx −28.2%。

## 入口

```text
PYTHONPATH=src python -m optimization.pareto_sizing
PYTHONPATH=src python -m pytest tests/test_pareto_sizing.py -q
```

## Related

- [2019 问题](../problems/2019-a-airport-charging.md)
- [BDI–DES](bdi-des-airport-charging.md)
- [source](../sources/2019-pareto-sizing.md)

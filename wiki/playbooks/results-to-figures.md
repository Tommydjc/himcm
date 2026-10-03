# 从 results CSV 出图

论文插图必须从题包 `results/*.csv` 生成，终稿放该题 `paper_figures/`。禁止手绘数字、禁止从聊天抄表。

```yaml
type: playbook
year: cross
status: ingested
sources:
  - 2022_A_honeybee_dynamics/results/orchard_pollination_optimization.csv
  - 2023_A_dandelion_prisms/src/scripts/plot_figures.py
  - 2024_A_olympic_sde/src/scripts/run_eval.py
  - 2024_A_olympic_sde/results/historical_ranking.csv
```

## 2022 A 入口

| 图 | 脚本 / 模块 | 源 CSV |
| --- | --- | --- |
| `fig_autograd_elasticity_bars.png` | `src/sensitivity/autograd_sens.py` | `autograd_elasticity_ranking.csv` |
| `fig_ccd_bifurcation_curve.png` | `src/stress_test/ccd_bifurcation.py` | `ccd_bifurcation_sweep.csv` |
| `fig_colony_dynamics_healthy_vs_ccd.png` | 同上 | `ccd_healthy_vs_stress_snapshot.csv` |
| `fig_20acre_hive_density_tradeoff.png` | `src/pollination/orchard_20acres.py` | `orchard_pollination_optimization.csv` |

## 2023 A 入口

| 图 | 脚本 | 源 CSV |
| --- | --- | --- |
| 扩散/锋面类图 | `src/scripts/plot_figures.py` | `monthly_metrics.csv` 等 |
| 日步权衡图 | `src/scripts/plot_tradeoff_effects.py` | `bioeconomic_tradeoff.csv` |

周步 cover 0.4511 与日步 100% 分图标注，禁止拼成一张「平均覆盖」。

## 2024 A 入口

| 图（题包内） | 脚本 | 源 CSV |
| --- | --- | --- |
| `paper_figures/fig_historical_backtest.png` | `src/scripts/run_eval.py` | `results/historical_ranking.csv` |
| `paper_figures/fig_sensitivity_shock.png` | `src/scripts/run_sensitivity.py` | `results/sensitivity_weight_shock.csv` |
| `paper_figures/fig_ml_2032_probability.png` | `src/ml/plot_and_report.py` | `results/ml_prediction_2032.csv` |
| `paper_figures/fig_ahp_vs_ml_weights.png` | `src/ml/importance.py` | `results/ahp_vs_ml_weights.csv` |

2025 仿真图只属于 `2025_A_evacuation/`，不要写进 2022–2024 论文路径。

## Related

- [2022 复现](../playbooks/2022-repro-bee.md)
- [2023 复现](../playbooks/2023-repro-sim.md)
- [2024 复现](../playbooks/2024-repro-eval.md)

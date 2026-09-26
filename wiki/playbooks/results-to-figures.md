# 从 results CSV 出图

论文插图必须从题包 `results/*.csv` 生成，终稿放该题 `paper_figures/`。禁止手绘数字、禁止从聊天抄表。

```yaml
type: playbook
year: cross
status: ingested
sources:
  - 2024_A_olympic_sde/src/scripts/run_eval.py
  - 2024_A_olympic_sde/src/scripts/run_sensitivity.py
  - 2024_A_olympic_sde/src/ml/plot_and_report.py
  - 2024_A_olympic_sde/results/historical_ranking.csv
  - 2024_A_olympic_sde/results/ml_prediction_2032.csv
```

## 2024 A 入口

| 图（题包内） | 脚本 | 源 CSV |
| --- | --- | --- |
| `paper_figures/fig_historical_backtest.png` | `src/scripts/run_eval.py` | `results/historical_ranking.csv` |
| `paper_figures/fig_sensitivity_shock.png` | `src/scripts/run_sensitivity.py` | `results/sensitivity_weight_shock.csv` |
| `paper_figures/fig_ml_2032_probability.png` | `src/ml/plot_and_report.py` | `results/ml_prediction_2032.csv` |
| `paper_figures/fig_ahp_vs_ml_weights.png` | `src/ml/importance.py` | `results/ahp_vs_ml_weights.csv` |
| `paper_figures/fig_ahp_vs_ml_dumbbell.png` | `src/ml/plot_and_report.py` | `results/ahp_vs_ml_weights.csv` |

LaTeX 片段 `paper/sections/sec4_ml_validation.tex` 由 `plot_and_report.py` 按 CSV 重写，不要手改其中的 LOOCV 数字去「好看」。

2025 仿真图只属于 `2025_A_evacuation/results/` 与该包出图脚本，不要写进 2024 论文路径。

## Related

- [2024 复现](../playbooks/2024-repro-eval.md)
- [LOOCV](../methods/loocv-logistic.md)

# results

周步论文数字由 `PYTHONPATH=. python -m src.scripts.run_all` 写入
`monthly_metrics.csv` / `kpp_check.csv` 等。禁止手改这些 CSV。

日步 5 节点与 Fisher–KPP 对照由
`PYTHONPATH=. python -m src.scripts.run_monthly_analysis` 写入
`monthly_population_metrics.csv`、`daily_front_radius.csv`、`daily_kpp_check.csv`
（不覆盖上述周步表）。图由脚本只读这些文件生成到 `paper_figures/`。

PRISMS 四策略生物经济由 `PYTHONPATH=. python -m src.decision.tradeoff_model`
写入 `bioeconomic_tradeoff.csv`，**不覆盖** 周步 `bioeconomic_pareto.csv`
（论文引用的 \(C=n(1+\eta/2)\) 网格）。
情景对照图：`PYTHONPATH=. python -m src.scripts.plot_tradeoff_effects`
（读 `tradeoff_daily_traces.csv` 与 `snapshots/tradeoff_*_day*.npy`）。

PPO 四策略评估由 `PYTHONPATH=. python -m src.rl.train_rl` 写入
`rl_policy_evaluation.csv`，不覆盖周步 `bioeconomic_pareto.csv`。

# 2023 A 复现命令

在题包根目录、仓库 `venv` 下跑通强迫 → 仿真 → 出图 → 测试。

```yaml
type: playbook
year: 2023
status: draft
sources:
  - 2023_A_dandelion_prisms/README.md
  - 2023_A_dandelion_prisms/src/scripts/run_all.py
  - 2023_A_dandelion_prisms/results/run_summary.json
```

## 步骤

1. `cd 2023_A_dandelion_prisms`
2. `PYTHONPATH=. python -m src.scripts.generate_forcing`
3. `PYTHONPATH=. python -m src.scripts.run_all`
4. `PYTHONPATH=. python -m src.scripts.plot_figures`
5. `python -m unittest discover -s tests -v`

## 验收

- `results/monthly_metrics.csv` 36 行（3 气候 × 12 月）。
- `run_summary.json`：`n_pareto_policies=22`，`impact_top=PUMON`。
- 11 个 unittest OK。
- 不手改 CSV。

## Related

- [2023 问题](../problems/2023-a-dandelion-prisms.md)
- [月末指标](../sources/monthly-metrics.md)

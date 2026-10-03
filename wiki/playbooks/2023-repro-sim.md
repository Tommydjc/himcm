# 2023 A 复现命令

在题包根目录、仓库 `venv` 下跑通强迫 → 仿真 → 出图 → 测试。

```yaml
type: playbook
year: 2023
status: ingested
sources:
  - 2023_A_dandelion_prisms/README.md
  - 2023_A_dandelion_prisms/src/scripts/run_all.py
```

## 步骤

```bash
cd 2023_A_dandelion_prisms
PYTHONPATH=. python -m src.scripts.generate_forcing
PYTHONPATH=. python -m src.scripts.run_all
PYTHONPATH=. python -m src.scripts.plot_figures
python -m unittest discover -s tests -v
```

可选 Open-Meteo 日表（不覆盖情景强迫、默认不改权威周步表）：

```bash
PYTHONPATH=. python scripts/download_real_climate.py
```

## 验收

- `results/monthly_metrics.csv`：温带 12 月 `cover_frac=0.4511`，`front_m=99.5`。
- `impact_factors.csv`：PUMON 0.9333 / REJAP 0.5700 / TAROF 0.1512。
- `kpp_check.csv`：`rel_err≈0.0521`。
- 不手改 CSV。

## Related

- [研究闭环](../playbooks/2023-a-study-loop.md)
- [2023 问题](../problems/2023-a-dandelion-prisms.md)
- [月末指标](../sources/monthly-metrics.md)

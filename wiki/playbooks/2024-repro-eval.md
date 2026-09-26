# 2024 复现评价管道

在题包根目录跑测试与三条实验入口。验收：pytest 全绿；不改已冻结 `results/*.csv`；图从 CSV 写入 `paper_figures/`。

```yaml
type: playbook
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/src/scripts/run_eval.py
  - 2024_A_olympic_sde/src/scripts/run_sensitivity.py
  - 2024_A_olympic_sde/src/ml/__main__.py
  - 2024_A_olympic_sde/src/indicators/loader.py
```

## 步骤

在仓库根之外，进入题包：

```bash
cd 2024_A_olympic_sde
PYTHONPATH=. python -m pytest tests -q
PYTHONPATH=. python -m src.indicators.loader
PYTHONPATH=. python -m src.scripts.run_eval
PYTHONPATH=. python -m src.scripts.run_sensitivity
PYTHONPATH=. python -m src.ml
```

`loader` 会重写 `data/processed/`。权威论文数字以已冻结的 `results/` 为准；ingest 过程不要提交对 `results/*.csv` 的改动。

## 验收

- `tests/` 全绿（AHP CR、score、loader、sensitivity、dataset、classifier、importance、pack layout）。
- 2032 基线仍为 Flag 0.5360 / Cricket 0.4842 / Squash 0.3960，否则先对 CSV，不要改 Wiki 数字去「凑」。
- 图路径见 [CSV 到图](../playbooks/results-to-figures.md)。

## Related

- [2024 问题](../problems/2024-a-olympic-sde.md)
- [研究闭环](../playbooks/2024-a-study-loop.md)
- [SAW](../methods/saw-topsis.md)

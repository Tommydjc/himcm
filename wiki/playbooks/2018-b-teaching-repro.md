# 2018 B Teaching 轨复现清单

按序跑通黄金日 → Harness → MAS → 论文编译。验收以 `results/` 锁定文件为准。

```yaml
type: playbook
year: 2018
status: ingested
sources:
  - 2018_B_smart_house_mpc/experiments/protocols.md
  - 2018_B_smart_house_mpc/results/bang_vs_mpc_summary.json
  - 2018_B_smart_house_mpc/results/harness_benchmark_results.csv
  - 2018_B_smart_house_mpc/results/agent_thermostat_dialogue.json
```

## 步骤

```bash
cd 2018_B_smart_house_mpc

# 1–5 Teaching 内核
PYTHONPATH=src python -m thermal_physics.building_rc
PYTHONPATH=src python -m comfort_tariffs.models
PYTHONPATH=src python -m baseline_thermostat.bang_bang
PYTHONPATH=src python -m control_mpc.prediction_matrices
PYTHONPATH=src python -m control_mpc.mpc_controller

# Prompt 5–6
PYTHONPATH=src python -m eval_harness.benchmark_harness
PYTHONPATH=src python -m agent_inhabitant.cognitive_thermostat

# 测试
PYTHONPATH=src python -m pytest tests/test_building_rc.py tests/test_bang_bang.py \
  tests/test_prediction_matrices.py tests/test_mpc_controller.py \
  tests/test_benchmark_harness.py tests/test_cognitive_thermostat.py -q

# Streamlit（可选）
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src streamlit run src/ui/interactive_app.py \
  --server.port 8507 --server.fileWatcherType none

# 论文
cd paper && pdflatex himcm_paper.tex && pdflatex summary_sheet.tex \
  && pdflatex homeowner_advisory_letter.tex && pdflatex ai_disclosure.tex
```

## 验收标准

| 检查 | 期望 |
| --- | --- |
| `bang_vs_mpc_summary.json` | 成本约 −4.2%；峰电约 −15.8%；启停 42→3 |
| `harness_benchmark_results.csv` | 10 行（2 控制器 × 5 场景） |
| `agent_thermostat_dialogue.json` | 含 consensus 与 comfort_config |
| `paper_figures/fig_bang_vs_mpc_day.png` 等 | 300 DPI 存在 |
| 论文主表 | 无 Nest−64.6%（附录除外） |

## Related

- [2018 问题](../problems/2018-b-smart-house.md)
- [双轨](../concepts/2018-b-dual-track.md)
- [STE100 写作](asd-ste100-academic-writing.md)
- [CSV 到图](results-to-figures.md)

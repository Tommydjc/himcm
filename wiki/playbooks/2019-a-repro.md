# 2019 A 机场充电复现清单

```yaml
type: playbook
year: 2019
status: ingested
sources:
  - 2019_A_charge_des_agent/results/stage2_baseline_metrics.csv
  - 2019_A_charge_des_agent/results/optimal_kiosk_sizing.csv
  - 2019_A_charge_des_agent/results/thunderstorm_stress_comparison.csv
  - 2019_A_charge_des_agent/prompts/cursor_log.md
```

## 步骤

```bash
cd 2019_A_charge_des_agent

PYTHONPATH=src python -m pytest tests/ -q
PYTHONPATH=src python -m scripts.run_stage2_simulation
PYTHONPATH=src python -m optimization.pareto_sizing

# 可选：反思 / 规划（需本机 Ollama 或走规则降级）
PYTHONPATH=src python -m des_agent.reflection_agent
PYTHONPATH=src python -m des_agent.planning_agent

# Streamlit
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src streamlit run src/des_env/interactive_app.py \
  --server.port 8506 --server.fileWatcherType none

cd paper && pdflatex himcm_paper.tex && pdflatex summary_sheet.tex \
  && pdflatex airport_authority_letter.tex && pdflatex ai_disclosure.tex
```

## 验收

| 文件 | 期望 |
| --- | --- |
| `stage2_baseline_metrics.csv` | 12/15/2 normal；wait≈0；reneging=0 |
| `optimal_kiosk_sizing.csv` | 膝点行 `is_knee_80k=True` → 8/16/0 |
| `thunderstorm_stress_comparison.csv` | 两行 baseline vs pareto_knee |
| `paper_figures/*.png` | 四图存在 |

## Related

- [2019 问题](../problems/2019-a-airport-charging.md)
- [STE100](asd-ste100-academic-writing.md)

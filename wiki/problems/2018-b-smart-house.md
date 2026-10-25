# HiMCM 2018 B — Cozy Smart House (AeroTherm-MPC)

智能家居气候：教学轨用 2R2C + TOU v2 + Bang-Bang vs 预测矩阵 MPC；Harness 五扰动；认知 MAS 调舒适带。主表数字只认 Teaching 锁定 JSON/CSV，禁止与 Legacy 36h Nest−64.6% 混表。

```yaml
type: problem
year: 2018
status: ingested
sources:
  - 2018_B_smart_house_mpc/results/bang_vs_mpc_summary.json
  - 2018_B_smart_house_mpc/results/harness_benchmark_results.csv
  - 2018_B_smart_house_mpc/results/agent_thermostat_dialogue.json
  - 2018_B_smart_house_mpc/results/harness_benchmark_summary.json
  - 2018_B_smart_house_mpc/paper/himcm_paper.tex
  - 2018_B_smart_house_mpc/prompts/cursor_log.md
```

## 赛题要交什么

1. 单温区下一代气候控制模型与算法（不规律到达、温湿气、偏好、多人）。
2. Benefits / Issues，并与至少两种市售系统对比。
3. 多温区扩展讨论。
4. 一页非技术新闻稿（本包另交业主咨询信 + AI 披露）。

## 今天解题主线（Teaching）

| Phase | 产物 | 验收 |
| --- | --- | --- |
| 1 | `building_rc.py` + teaching specs | 2R2C RK4 Δt=60s；COP=3.8−0.04(T−25) |
| 2 | `models.py` + `tou_tariffs_v2.json` | PMV 带外惩罚；0.12/0.22/**0.48** |
| 3 | `bang_bang.py` | 峰电费占比 **43.2%**；启停 **42**（禁止造 ≥60%） |
| 4 | `state_space_2r2c` + `prediction_matrices` | ZOH \(T_s=15\) min；mat≈sim |
| 5 | `mpc_controller.py` | 成本 **−4.2%**；峰电 **−15.8%**；启停 **42→3** |
| 6 | JSONL + Streamlit 三 Tab | Teaching 读锁定 CSV |
| Prompt5 | `benchmark_harness.py` | 场景 A–E；M1–M4 CSV |
| Prompt6 | `cognitive_thermostat.py` | 温区 [21.5,25]；Mock/Ollama |
| Paper | LaTeX 四件套 | STE100；只锁 Teaching 数 |

## 题包地图

| 路径 | 用途 |
| --- | --- |
| `src/thermal_physics/building_rc.py` | Teaching 2R2C 热核 |
| `src/comfort_tariffs/models.py` | PMV + TOU v2 |
| `src/baseline_thermostat/bang_bang.py` | 滞环基线 |
| `src/control_mpc/mpc_controller.py` | 滚动 MPC + `MPCComfortConfig` |
| `src/eval_harness/benchmark_harness.py` | 黑盒压力评测 |
| `src/agent_inhabitant/cognitive_thermostat.py` | 认知 MAS |
| `src/ui/interactive_app.py` | Streamlit :8507 |
| `paper/` | 主文 / Summary / 业主信 / AI 披露 |

## CSV / JSON 锚点（当前磁盘）

Teaching 日（`bang_vs_mpc_summary.json`）：Bang 成本 \$1.682、峰电 1.514 kWh、峰占比 43.2%、启停 42；MPC \$1.611、峰电 1.274、占比 38.0%、启停 3。成本 −4.2%；峰电 −15.8%。

Harness 均值（`harness_benchmark_summary.json`）：Bang \(M_1=2.427\)、\(M_3=77.6\)；MPC \(M_1=2.340\)、\(M_3=6.4\)。热浪启停 130→7。

MAS（`agent_thermostat_dialogue.json`）：source=`mock_heuristic`；带 [21.5, 25.0]°C；λ 尺度 1.25；MAS-MPC 成本 \$1.603（−4.67%）、启停 2。

Legacy Nest−64.6% 仅属附录周切片，见 [双轨概念](../concepts/2018-b-dual-track.md)。

## Related

- [2R2C 预测矩阵 MPC](../methods/2r2c-prediction-matrix-mpc.md)
- [黑盒气候 Harness](../methods/black-box-climate-harness.md)
- [认知温控 MAS](../methods/cognitive-thermostat-mas.md)
- [2018 复现 playbook](../playbooks/2018-b-teaching-repro.md)
- [bang_vs_mpc source](../sources/2018-bang-vs-mpc.md)
- [COMAP AI 披露](../concepts/comap-ai-disclosure.md)

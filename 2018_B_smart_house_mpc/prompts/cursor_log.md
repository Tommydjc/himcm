# Cursor AI Interaction Log（COMAP AI Disclosure）

| 时间戳 | 任务目标 | 涉及文件 | 人工确认要点 |
|---|---|---|---|
| 2026-10-25 | 锁定 2018B 五条设计依据并搭建题包脚手架 | `2018_B_smart_house_mpc/` 全树 | 多物理场 ODE；2R2C 移峰；ETA+τ_lead；纳什议价；News Release 非信件 |
| 2026-10-25 | 实现 2R2C+湿+过敏原单温区与多温区图耦合 | `single_zone_rc.py`；`multi_zone_coupling.py` | RH>60% 触发 G_bio；VAV 风门和=1；Laplacian 内墙导热 |
| 2026-10-25 | 竞品对照 + 滚动 MPC + Geofence 到达预测 | `competitors.py`；`mpc_solver.py`；`arrival_predictor.py` | 同物理沙盒对比 programmable/Nest/Ecobee；SciPy L-BFGS-B |
| 2026-10-25 | MAS 议价 / Harness / 全流水线 | `inhabitant_mas.py`；`test_harness.py`；`run_full_pipeline.py` | Ollama 降级非对称纳什；热浪/电价暴涨/漂移 |
| 2026-10-25 | 修正墙体太阳得热尺度并重跑 36h 基准 | `building_specs.json`；`single_zone_rc.py` | 有效受照面积；MPC vs Nest 能耗 −64.6%、不舒适 −86.9% |
| 2026-10-25 | LaTeX 主文 + Summary + AP News Release | `paper/himcm_paper.tex` 等 | STE100；数字引自 competitor_benchmarks；Req3 新闻通稿非信件 |
| 2026-10-25 | Streamlit 动态沙盒（竞品+MPC） | `src/ui/interactive_app.py` :8507 | 五控制器/热浪电价工况；Plotly 温湿气+HVAC；fileWatcher=none |
| 2026-10-25 | Phase1 教学 2R2C 热核 | `building_rc.py`；`building_specs_teaching.json`；`test_building_rc.py` | 180m²/2R2C；RK4 Δt=60s；COP=3.8−0.04(T−25)；自由漂浮→35℃+能量守恒；导出 Ac,Bc,Ec |
| 2026-10-25 | Phase2 PMV惩罚 + TOU v2 | `comfort_tariffs/models.py`；`tou_tariffs_v2.json`；`test_comfort_tou_v2.py` | |PMV|>0.5 二次惩罚；谷/平/峰 0.12/0.22/0.48（峰谷比 4×）；24°C PMV≈−0.21 |
| 2026-10-25 | Phase3 Bang-Bang 高温日 | `bang_bang.py`；`weather_summer_day_1min.csv`；`baseline_bang_bang_*.csv/json` | 滞环[22,24]/Q=5kW；1440min；峰电费占比43.2%；启停42次；下调α_solar使5kW可控 |
| 2026-10-25 | Phase4 状态空间+预测矩阵 | `state_space_2r2c.py`；`prediction_matrices.py`；`test_prediction_matrices.py` | ZOH Van Loan Ts=900s；X=𝒜x+ℬU+ℰD；mat vs sim err≈7e-15；因果下三角 |
| 2026-10-25 | Phase5 滚动MPC vs Bang-Bang | `mpc_controller.py`；`mpc_optimal_*.csv/json`；`bang_vs_mpc_summary.json`；`fig_bang_vs_mpc_day.png` | H=24/Ts=15min；预测矩阵代价；峰电−15.8%；电费−4.2%；启停42→3 |
| 2026-10-25 | Phase6 JSONL+Streamlit+论文刷新 | `run_logger.py`；`interactive_app.py` 三Tab；`himcm_paper/summary/news`；`test_run_logger.py` | Teaching主表锁 bang_vs_mpc；Nest−64.6%附录；logs/*.jsonl |
| 2026-10-25 | Prompt5 黑盒评测Harness | `benchmark_harness.py`；`harness_benchmark_results.csv`；雷达+追踪图；`test_benchmark_harness.py` | 场景A–E；M1电费/M2违规°C·h/M3启停/M4方差 |
| 2026-10-25 | Prompt6 认知MAS温控博弈 | `cognitive_thermostat.py`；`agent_thermostat_dialogue.json`；`MPCComfortConfig`注入 | Ollama qwen2.5:7b→HTTP→Mock；老人底线21.5℃ |
| 2026-10-25 | 论文四件套刷新 | `himcm_paper.tex`；`summary_sheet.tex`；`homeowner_advisory_letter.tex`；`ai_disclosure.tex` | STE100；引用 harness/MAS/bang_vs_mpc 锁定数字；≤25页 |
| 2026-10-25 | Wiki ingest 全日轨 | `wiki/problems/2018-b-smart-house.md` 等 9 页 + index/log/synthesis/STUDIES | 只读 Raw；Teaching 锚点入库；题型路由增 G |


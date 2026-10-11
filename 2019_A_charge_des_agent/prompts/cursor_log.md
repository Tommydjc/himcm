# Cursor AI Interaction Log（COMAP AI Disclosure）

| 时间戳 | 任务目标 | 涉及文件 | 人工确认要点 |
|---|---|---|---|
| 2026-10-11 | BDI 旅客认知状态机（arXiv:2602.13878） | `src/bdi_agents/passenger_bdi.py`；`tests/test_passenger_bdi.py` | 25% 电量焦虑先验；`T_board~U(45,120)`；`U_charge` 硬约束误机；IDLE→SEEK→QUEUE→RENEGING→RENT/ABORT；T_board=10 且队列>5 时 100% 弃排自检 |
| 2026-10-11 | SimPy 航站楼充电 DES 沙盒（BDI–DES 映射） | `src/des_env/airport_sim.py`；`tests/test_airport_sim.py` | 三资源池 65W/15W/充电宝；NHPP 16h 三波峰×3；`passenger_process` 弃排+登机强制拔电；`run_airport_simulation` 返回 wait/queue/reneging/util/rentals |
| 2026-10-11 | 阶段 2 基线仿真脚本修正并执行 | `src/scripts/run_stage2_simulation.py`；`airport_sim` 时序/别名 | 原文件是未展开 shell 模板；对接真实 API；导出 CSV+PNG；路径锚定题包根目录 |
| 2026-10-11 | 修复基线图误用纯等待队长 + Streamlit 动态沙盒 | `airport_sim.py` 时序字段；`interactive_app.py`；重跑 stage2 | 根因：过配置下 queue_length≡0；改为 system_load/in_service/λ(t)；情景 normal/rush/storm |
| 2026-10-11 | FastMCP 封装机场 DES（Tools/Resource/进程内调用） | `src/mcp_server/server.py`；`tests/test_mcp_server.py` | `run_charging_simulation`/`validate_infrastructure_budget`；`config://airport_constraints`；`get_in_process_tools()` 避跨进程 |
| 2026-10-11 | LangGraph 自反思调优智能体（Ollama/规则降级） | `src/des_agent/reflection_agent.py`；`results/des_agent_reflection_log.json` | propose→validate→simulate→reflect→check；预算驳回最多 3 次；目标 wait≤5 且弃排≤5%；Ollama 宕机走规则 |
| 2026-10-11 | 注册 Cursor MCP：airport-charging-des | `.cursor/mcp.json`；`src/mcp_server/__main__.py` | stdio 入口纯 `mcp.run()`；PYTHONPATH=题包 src；工具含仿真/预算校验/约束 Resource |
| 2026-10-11 | 宏观规划智能体：双 LLM + 调度 reflection | `src/des_agent/planning_agent.py`；`autonomous_planning_summary.json` | 四子任务（40k/80k/定价弹性×2）→ reflection 局部搜索 → Pareto 汇总；Ollama 降级规则 |
| 2026-10-11 | Pareto 前沿 + 雷暴延误压力测试与三图 | `src/optimization/pareto_sizing.py`；`airport_sim` storm 注入 | CapEx–Wait 非支配排序；17:00 40% 延误 2–4h+电量减半；fig_pareto/stress/sankey 300DPI |
| 2026-10-11 | Pareto 膝点修正：弃排≤5% 过滤 | `pareto_sizing.py`；`test_pareto_sizing.py` | 纯双目标会塌到 4/0/0；服务可行集上再取 $80k 膝点；网格跳过 <4 桩退化包 |
| 2026-10-11 | LaTeX 论文套件（STE100 + 真实 KPI） | `paper/himcm_paper.tex`；`summary_sheet.tex`；`airport_authority_letter.tex`；`ai_disclosure.tex` | 六章主文+1页摘要+管理信+AI披露；数字只引自 results CSV；术语统一 BDI/fast kiosk/reneging |

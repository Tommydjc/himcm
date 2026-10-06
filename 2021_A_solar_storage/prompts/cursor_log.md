# Cursor AI Interaction Log（COMAP AI Disclosure）

| 时间戳 | 任务目标 | 涉及文件 | 人工确认要点 |
|---|---|---|---|
| 2026-10-04 | 8760 h 光伏 Hay–Davies 出力 + MCMC 家庭负荷，写出净功率表 | `src/generation/solar_pv.py`；`src/load_profiler/mcmc_load.py`；`src/scripts/run_pv_load.py` | 辐射来自 Open-Meteo `solar_irradiance_annual.csv`；家电来自 `appliances_catalog.csv`；`net=pv-load`；未手改 CSV |
| 2026-10-04 | 商用电池 SOC 状态机 + EFC/DoD 吞吐 SOH 衰减 | `src/battery_model/battery_dynamics.py`；`tests/test_battery_dynamics.py` | 解析 `commercial_batteries.csv`；补齐 Enphase/铅碳基线；充放电能量损耗断言；未手改电池 CSV |
| 2026-10-04 | SciPy HiGHS MILP 容量配置 + 代表日 EMS，8760 h 评估 | `src/optimization/sizing_milp.py` | \(x_k\in\{0..4\}\)；节点平衡/SOC/LPSP；禁止电池组合枚举；写出 `optimal_sizing_solution.csv` |
| 2026-10-04 | 预算 Pareto 扫描 + 四类储能 LCOS + 300 DPI 三图 | `src/techno_economic/lcos_and_pareto.py` | \$4k–\$18k 调用既有 MILP；LCOS 吞吐来自 8760 CSV；Na-ion/VRFB/水泥为文献量级假设非实验伪造 |
| 2026-10-04 | 家庭多智能体 EMS：三级暴雪预警 + 家长/居家办公博弈 + DSR 回灌 | `src/agents/household_ems.py` | SOC 20% 警戒线；保护电脑/路由/冰箱；行动 JSON 回灌 8760 h；对话为确定性模板非在线 LLM |
| 2026-10-04 | crewAI + 本地 Ollama 家庭减载协商基座（无模型则启发式 Mock） | `src/agent_sim/models.py`；`src/agent_sim/crew_engine.py` | `LoadSheddingAgreement` Pydantic 契约；`OLLAMA_MODEL` 可覆盖；Ollama 空模型时不崩溃 |
| 2026-10-04 | 协商契约→kW 适配器 + Day40–42 暴雪 A/B 闭环 | `src/agent_sim/load_adapter.py`；`src/agent_sim/run_simulation.py` | 8%/°C HVAC；刚性下限 0.3 kW；对照 CSV 来自真实 8760 气象；SOC 用 `step_soc` |
| 2026-10-04 | 关闭 CrewAI Plus ephemeral 云端 trace 链接 | `src/agent_sim/crew_engine.py`；`src/agent_sim/run_marathon_simulation.py` | `CREWAI_TRACING_ENABLED=false`；`Crew(tracing=False)`；对话看终端 / SQLite / md，不依赖 app.crewai.com |
| 2026-10-06 | 25 页内论文套件 + 业主信；数字只来自 results CSV | `paper/himcm_paper.tex`；`paper/summary_sheet.tex`；`paper/letter_to_homeowners.tex` | \$10k=3 FREEDOH+铅碳非 Powerwall；Na LCOS \$0.161；crewAI 16→5 h 非 32.5 h 清零 |
| 2026-10-06 | 80\% ASD-STE100 润色：短句、一词一义、主动语态 | 同上 + `paper/ai_disclosure_local_llm.tex` | 统一 pack/fleet/MILP/adapter/crew/DSR；公式与 CSV 数字未改 |

# HiMCM 2019 A — Airport charging (BDI–DES)

航站楼移动充电亭选型：SimPy 三资源池 DES + BDI 旅客智能体 + CapEx/面积约束下的 Pareto 网格；LangGraph/Ollama 自反思与宏观规划为加分层。数字只认 `2019_A_charge_des_agent/results/*`。

```yaml
type: problem
year: 2019
status: ingested
sources:
  - 2019_A_charge_des_agent/results/stage2_baseline_metrics.csv
  - 2019_A_charge_des_agent/results/optimal_kiosk_sizing.csv
  - 2019_A_charge_des_agent/results/thunderstorm_stress_comparison.csv
  - 2019_A_charge_des_agent/results/autonomous_planning_summary.json
  - 2019_A_charge_des_agent/paper/himcm_paper.tex
  - 2019_A_charge_des_agent/prompts/cursor_log.md
```

## 赛题要交什么

1. 旅客充电行为与排队/弃排模型。
2. 快充亭 / 慢充柜 / 充电宝配比与预算、占地约束。
3. 压力情景（如雷暴延误）下的服务 KPI。
4. 给机场当局的非技术建议信 + AI 披露。

## 解题主线（题包已落地）

| 层 | 模块 | 作用 |
| --- | --- | --- |
| 微观认知 | `bdi_agents/passenger_bdi.py` | SOC 焦虑、登机松弛、弃排意图 |
| 物理 DES | `des_env/airport_sim.py` | 65W / 15W / power-bank；NHPP 16h |
| 选型 OR | `optimization/pareto_sizing.py` | CapEx–Wait 非支配；弃排≤5% 可行膝点 |
| 工具协议 | `mcp_server/` | FastMCP 仿真/预算校验 |
| 自反思 | `des_agent/reflection_agent.py` | propose→validate→simulate→reflect |
| 宏观规划 | `des_agent/planning_agent.py` | 四子任务 + Ollama/规则汇总 |

## 题包地图

| 路径 | 用途 |
| --- | --- |
| `src/des_env/` | SimPy 沙盒 + Streamlit `:8506` |
| `src/bdi_agents/` | BDI 旅客 |
| `src/optimization/` | Pareto 网格 |
| `src/des_agent/` | LangGraph 反思 / 规划 |
| `src/mcp_server/` | Cursor MCP 工具 |
| `paper/` | 主文 / Summary / 当局信 / AI 披露 |
| `paper_figures/` | 基线队列、Pareto、雷暴、Sankey |

## CSV / JSON 锚点（当前磁盘）

**Stage-2 基线**（`12/15/2`，normal）：到客 441；焦虑寻求者 110；均值等待 **0.00** min；弃排 **0.00**；快充利用率 5.72%；慢充 24.24%；最大系统负荷 17。

**Pareto 膝点**（rush、弃排≤5%、\$80k 下）：包 **8/16/0**；CapEx **\$35{,}200**；面积 38.4 m²；rush 均值等待 0.0115 min；弃排 0。最便宜 Pareto 点 4/8/0（等待 5.91 min）。

**雷暴压力**（`thunderstorm_stress_comparison.csv`，n=621）：基线 12/15/2 等待 4.591 / 弃排 0.0229 / CapEx \$49k；膝点 8/16/0 等待 2.347 / 弃排 0.0114 / CapEx \$35.2k（等待 −48.9%，弃排 −50.3%，CapEx −28.2%）。

验收阈值：均值等待 ≤5 min；弃排概率 ≤0.05。

## Related

- [BDI–DES 方法](../methods/bdi-des-airport-charging.md)
- [Pareto 亭选](../methods/pareto-kiosk-sizing.md)
- [LangGraph 反思](../methods/langgraph-des-reflection.md)
- [2019 复现](../playbooks/2019-a-repro.md)
- [stage2 source](../sources/2019-stage2-baseline.md)
- [COMAP AI 披露](../concepts/comap-ai-disclosure.md)

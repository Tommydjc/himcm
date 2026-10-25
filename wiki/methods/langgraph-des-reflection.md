# LangGraph DES 自反思与宏观规划

把机场 DES 封成 FastMCP 工具后，用本地 Ollama（qwen2.5:7b）驱动 propose→validate→simulate→reflect 回路；宕机则规则降级。

```yaml
type: method
year: 2019
status: ingested
sources:
  - 2019_A_charge_des_agent/src/des_agent/reflection_agent.py
  - 2019_A_charge_des_agent/src/des_agent/planning_agent.py
  - 2019_A_charge_des_agent/src/mcp_server/server.py
  - 2019_A_charge_des_agent/results/des_agent_reflection_log.json
  - 2019_A_charge_des_agent/results/autonomous_planning_summary.json
```

## 角色分工

| 组件 | 职责 |
| --- | --- |
| FastMCP | `run_charging_simulation` / `validate_infrastructure_budget` |
| reflection_agent | 局部改包直至 wait≤5 且 reneging≤0.05 |
| planning_agent | 拆四子任务（紧急最小 / 配比 / 定价弹性 / 综合）再汇总 |

## 锁定注意

`autonomous_planning_summary.json` 的 `plan_source=ollama`。  
子任务 003（低预算定价）候选可出现高弃排（约 0.637）——属探索点，**不是**运营推荐膝点；运营主推荐仍以 `optimal_kiosk_sizing.csv` 的 8/16/0 为准。

## Related

- [2019 问题](../problems/2019-a-airport-charging.md)
- [CrewAI DSR（2021 对照）](crewai-cognitive-demand-response.md)
- [认知温控 MAS（2018 对照）](cognitive-thermostat-mas.md)
- [AI 披露](../concepts/comap-ai-disclosure.md)

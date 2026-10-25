# 认知温控多智能体（Ollama / Mock）

用家庭角色对话动态修正 MPC 舒适带与 \(\lambda_{\mathrm{comf}}\)，避免硬编码设定值。

```yaml
type: method
year: 2018
status: ingested
sources:
  - 2018_B_smart_house_mpc/src/agent_inhabitant/cognitive_thermostat.py
  - 2018_B_smart_house_mpc/results/agent_thermostat_dialogue.json
  - 2018_B_smart_house_mpc/results/agent_mpc_vs_bang.json
```

## 三角色

| Agent | 诉求 | 硬约束 |
| --- | --- | --- |
| EMS | 峰电移峰 | 可提议抬高下午设定 |
| Senior | 畏寒 | \(T\ge 21.5^\circ\mathrm{C}\) |
| Parent | 账单 | 峰段可短时至 25°C |

## 后端顺序

1. `langchain_ollama` ChatOllama（`qwen2.5:7b`，短超时）  
2. Ollama HTTP `/api/generate`  
3. **Mock 启发式**（保证不崩）

锁定 run：`source=mock_heuristic`。论文须写明 Mock，不可假装直播 LLM。

## 契约 → MPC

Pydantic `ComfortConsensus` → `MPCComfortConfig`：  
\(T_{\min}=21.5\)，\(T_{\max}=25.0\)，`comfort_weight_scale=1.25` → \(\lambda_{\mathrm{comf}}=0.0625\)。

闭环（同日）：MPC 成本 \$1.603（相对 Bang −4.67%）；峰电 −15.6%；启停 42→2。

## 入口

```text
PYTHONPATH=src python -m agent_inhabitant.cognitive_thermostat
```

## Related

- [2018 问题](../problems/2018-b-smart-house.md)
- [MPC](2r2c-prediction-matrix-mpc.md)
- [CrewAI DSR（2021 对照）](crewai-cognitive-demand-response.md)
- [AI 披露](../concepts/comap-ai-disclosure.md)
- [dialogue source](../sources/2018-agent-thermostat.md)

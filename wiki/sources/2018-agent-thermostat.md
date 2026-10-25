# Source：agent_thermostat_dialogue.json

认知 MAS 协商日志 + 注入 MPC 后的同日对照。

```yaml
type: source
year: 2018
status: ingested
sources:
  - 2018_B_smart_house_mpc/results/agent_thermostat_dialogue.json
  - 2018_B_smart_house_mpc/results/agent_mpc_vs_bang.json
```

## 摘录

`bargain.source`：**mock_heuristic**（Ollama 路径存在但本锁定 run 走 Mock）。  
Consensus：`agreed_temp_min=21.5`，`agreed_temp_max=25.0`，`comfort_weight_scale=1.25`。  
`comfort_config.lambda_comf=0.0625`，`T_precool_C=21.5`，`T_peak_C=25.0`。  
闭环：Bang \$1.682 / 启停 42；MAS-MPC \$1.603 / 启停 2；cost_savings_pct 4.67；peak_energy_savings_pct 15.62。

## Related

- [认知 MAS 方法](../methods/cognitive-thermostat-mas.md)
- [AI 披露](../concepts/comap-ai-disclosure.md)
- [2018 问题](../problems/2018-b-smart-house.md)

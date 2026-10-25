# 2019 A 失败模式

机场充电 DES 题包特有坑；与跨年总表互补。

```yaml
type: concept
year: 2019
status: ingested
sources:
  - 2019_A_charge_des_agent/prompts/cursor_log.md
  - 2019_A_charge_des_agent/results/optimal_kiosk_sizing.csv
  - 2019_A_charge_des_agent/results/autonomous_planning_summary.json
```

## 清单

1. **纯双目标无弃排过滤**：膝点塌到几乎不服务的廉价包。  
2. **把 planning LLM 推荐当运营终裁**：003 类高弃排点须标注探索。  
3. **队长图误用 queue_length**：过配置下恒为 0；用 system_load。  
4. **雷暴 KPI 与 normal 基线混比**：须同情景对照。  
5. **依赖云端 LLM**：契约是本地 Ollama；必须有规则降级。  
6. **负排队 / 时钟回拨**：SimPy 纪律。

## Related

- [跨年失败模式](cross-year-failure-modes.md)
- [2019 问题](../problems/2019-a-airport-charging.md)
- [Pareto 方法](../methods/pareto-kiosk-sizing.md)

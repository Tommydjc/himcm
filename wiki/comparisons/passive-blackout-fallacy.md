# Passive blackout fallacy

把「电池到 \(SOC_{\min}\) 即全年同等停电」当成唯一情景，忽略家庭可推迟负荷。2021 对照：`crewai_adaptive_vs_passive.csv` 双列 SOC。

```yaml
type: comparison
year: 2021
status: ingested
sources:
  - 2021_A_solar_storage/results/crewai_adaptive_vs_passive.csv
  - 2021_A_solar_storage/results/optimal_sizing_solution.csv
  - 2021_A_solar_storage/src/agent_sim/crew_engine.py
```

## 对照

| 假定 | 含义 | 本包 |
| --- | --- | --- |
| 被动 | 负荷曲线固定，LPSP 只由容量决定 | 选型表 `lpsp=0.2489`，`outage_hours_year=2725` |
| 自适应 | 角色谈判削减，SOC 轨迹分列 | CSV 有 `soc_passive` 与 `soc_adaptive` |

不要把 72 h 叙事窗的削负荷推广成「年停电小时从 2725 降到 X」——年尺度未在本 CSV 重算则为 `unverified`。

## Related

- [CrewAI DSR](../methods/crewai-cognitive-demand-response.md)
- [2021 微电网](../problems/2021-a-solar-storage.md)

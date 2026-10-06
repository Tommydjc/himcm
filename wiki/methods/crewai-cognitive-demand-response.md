# CrewAI cognitive demand response

事件驱动多角色（butler / parent / remote worker）在本地 Ollama 上谈判可削减负荷，输出必须通过 Pydantic，再写入电池 SOC 轨迹。代码：`2021_A_solar_storage/src/agent_sim/`。

```yaml
type: method
year: 2021
status: ingested
sources:
  - 2021_A_solar_storage/results/crewai_adaptive_vs_passive.csv
  - 2021_A_solar_storage/src/agent_sim/crew_engine.py
  - 2021_A_solar_storage/paper/himcm_paper.tex
```

## 适用

家庭微电网短窗（本包示例：CSV 时间戳 2021-02-09 起约 72 h）。LLM 只改「可推迟 kW」，不改物理 SOC 更新器。

## 耦合

1. 角色对话 → schema（削减协议）。
2. 适配器改 `negotiated_load_kW`。
3. 电池动力学写 `soc_adaptive` vs `soc_passive`。

Ollama 宕机则启发式走同一 schema。CSV 中可见 \(SOC_{\min}\approx 0.178\)。削减总 kWh 以脚本对 CSV 求和为准，Wiki 不另造。

## 失败模式

把对话文本里的 kWh 当测量；让 LLM 改 SOC 公式。见 [被动停电谬误](../comparisons/passive-blackout-fallacy.md)。

## Related

- [2021 微电网](../problems/2021-a-solar-storage.md)
- [COMAP 披露](../concepts/comap-ai-disclosure.md)

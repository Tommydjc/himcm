# HiMCM 2021 A — Off-grid solar storage

离网 PV–电池：8760 h 负荷/辐照 + MILP 选型 + CrewAI 认知削负荷。数字只认当前 `2021_A_solar_storage/results/*.csv`（与早期论文草稿若冲突，以 CSV 为准）。

```yaml
type: problem
year: 2021
status: ingested
sources:
  - 2021_A_solar_storage/results/optimal_sizing_solution.csv
  - 2021_A_solar_storage/results/lcos_emerging_tech.csv
  - 2021_A_solar_storage/results/crewai_adaptive_vs_passive.csv
  - 2021_A_solar_storage/results/annual_pv_load_8760h.csv
  - 2021_A_solar_storage/results/budget_pareto_front.csv
  - 2021_A_solar_storage/paper/himcm_paper.tex
```

## 赛题要交什么

1. 年尺度光伏与负荷。
2. 预算约束下的商用包选型。
3. 停电/LPSP。
4. 家庭可削减负荷（本包用 CrewAI + 本地 LLM，失败则启发式同 schema）。
5. 给房主的非技术信。

## 题包地图

| 路径 | 用途 |
| --- | --- |
| `src/generation/` | PV |
| `src/load_profiler/` | MCMC / 负荷 |
| `src/battery_model/` | SOC |
| `src/optimization/` | 选型 MILP |
| `src/agent_sim/` | CrewAI DSR |
| `src/techno_economic/` | LCOS / Pareto |

## CSV 锚点（当前磁盘）

`optimal_sizing_solution.csv`（\$10k 预算行）：**3× FREEDOH**（15 kWh / 7.5 kW / \$6{,}600）+ **1× lead-carbon**（9.6 kWh / 3 kW / \$3{,}100）。合计 capex \$9{,}700。Tesla Powerwall+ 本行 `units=0`。同表 `lpsp=0.2489`，`outage_hours_year=2725`，`tco_usd≈1.577\times 10^5`。

`lcos_emerging_tech.csv`（\(r=0.07\)，\(N=20\)）：Na-ion **0.161 \$/kWh**；VRFB 0.292；LFP 0.410；cement 1.273。

`crewai_adaptive_vs_passive.csv`：72 h 轨迹含 `soc_passive` / `soc_adaptive`，可见 \(SOC_{\min}\approx 0.178\)。削负荷 kWh **未在本 Wiki 重算**；论文派生值回题包脚本，不在此发明。

8760 行净功率在 `annual_pv_load_8760h.csv`。年电量若需引用，对 CSV 求和，不要抄过期 Summary。

## Related

- [CrewAI DSR](../methods/crewai-cognitive-demand-response.md)
- [被动停电谬误](../comparisons/passive-blackout-fallacy.md)
- [熵权（别年）](../methods/entropy-weight-objective-scoring.md)

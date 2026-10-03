# 2023 A 失败模式

情景气候、两套几何、三行影响宇宙与把 RL 写成主结论是本题最容易写错的几处。

```yaml
type: concept
year: 2023
status: ingested
sources:
  - 2023_A_dandelion_prisms/prompts/cursor_log.md
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2023_A_dandelion_prisms/results/impact_factors.csv
  - 2023_A_dandelion_prisms/results/monthly_population_metrics.csv
  - 2023_A_dandelion_prisms/data/raw/weather/README.md
```

## 六条

1. **气候 CSV 默认不是台站观测**。情景 `*_hourly.csv` / `*_smi.csv` 与 Open-Meteo 日表分轨。
2. **\(n_{\mathrm{plants}}\) 含幼苗**。温带 12 月约 \(4.41\times 10^5\) 株，成株约 4674。
3. **TAROF impact≈0.151 只在三行宇宙**；旧稿「TAROF=0」已过期，勿回写。
4. **周步 cover 0.4511 ≠ 日步 100%**。西缘周卷积与中心日步是两套几何。
5. **热带锋面短是离岸风 + 边界滞留**，不是「热带不长」。
6. **PPO 评测表不是官方 Req**；`rl_policy_evaluation.csv` 显示 learned PPO 与 rewilding 同轨时，不得写成「已击败全部田间策略」。

## Related

- [2023 问题](../problems/2023-a-dandelion-prisms.md)
- [两套宇宙](../concepts/two-universes.md)

# Source：sensitivity_weight_shock.csv

OAT ±20% 与 `youth_weight_zero`。读表时先过滤 `Universe`：`brisbane_candidates` 与 `sde_matrix` 的 Rank 列不可混用。

```yaml
type: source
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/results/sensitivity_weight_shock.csv
  - 2024_A_olympic_sde/paper/himcm_paper.tex
```

## 三候选宇宙（决策相关）

- `Scenario=baseline`：与 [brisbane_2032_ranking](../sources/brisbane-2032-ranking.md) 同序，Flag 0.5360 第 1。
- 所有 `plus20` / `minus20` 行：Flag–Cricket–Squash 名次不变（论文：14 个 ±20% 情景中 Flag 在 13 个里第 1，青年归零那一格除外）。
- `youth_weight_zero`：

| SDE | Rank | Score | Baseline_Rank | Delta_Rank |
| --- | --- | --- | --- | --- |
| Squash | 1 | 0.5077 | 3 | 2 |
| Cricket | 2 | 0.4092 | 2 | 0 |
| Flag football | 3 | 0.4051 | 1 | −2 |

## 九行宇宙（回测相关）

`Universe=sde_matrix` 且 `youth_weight_zero`：Athletics 升至第 1，Swimming 第 2，Breaking 从基线第 2 落到第 8（CSV：Baseline_Rank 2，Rank 8，Delta_Rank −6）。

## Related

- [OAT 方法](../methods/oat-sensitivity.md)
- [两套宇宙](../concepts/two-universes.md)

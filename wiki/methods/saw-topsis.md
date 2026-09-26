# SAW 与轻量 TOPSIS

方案层：先按效益/成本做 min-max+\(\varepsilon\)，再与 AHP 权做简单加权和。TOPSIS 在 `score.py` 中提供，主排名表用 SAW。

```yaml
type: method
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/src/mcda/score.py
  - 2024_A_olympic_sde/src/indicators/schema.py
  - 2024_A_olympic_sde/results/historical_ranking.csv
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
  - 2024_A_olympic_sde/tests/test_score.py
```

## 规范化

\(\varepsilon=\) `RANGE_EPSILON` \(=10^{-12}\)。对同一因子、**当前表中全部方案**取 min/max：

- 效益：\(\tilde x=(x-\min)/(\max-\min+\varepsilon)\)
- 成本：\(\tilde x=(\max-x)/(\max-\min+\varepsilon)\)

`schema.py` 中成本叶：`Events_Count`、`INFRA_COST`。其余锁定七叶为效益。权重须非负且和为 1。

## 两套宇宙不可比

min-max 依赖对照集。九行 `sde_matrix` 与三行 `brisbane_candidates` 的同一运动综合分不能横比名次。见 [两套宇宙](../concepts/two-universes.md)。

锁定三候选 SAW（`brisbane_2032_ranking.csv`）：Flag football 0.5360 第 1，Cricket 0.4842 第 2，Squash 0.3960 第 3。

九行回测第一名 Skateboarding，`Norm_Score` 0.6879（`historical_ranking.csv`）。

## 代码入口

- `normalize_indicators` / SAW：`2024_A_olympic_sde/src/mcda/score.py`
- 历史：`PYTHONPATH=. python -m src.scripts.run_eval`
- 2032：`PYTHONPATH=. python -m src.scripts.run_sensitivity`

## Related

- [AHP 特征值](../methods/ahp-eigenvalue.md)
- [OAT 敏感性](../methods/oat-sensitivity.md)
- [2032 排序](../sources/brisbane-2032-ranking.md)
- [历史回测](../sources/historical-ranking.md)

# 两套评价宇宙

九行历史矩阵与三行 2032 候选必须分开读名次。min-max 在不同对照集上，同一运动的综合分不可横比。

```yaml
type: concept
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/results/historical_ranking.csv
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
  - 2024_A_olympic_sde/results/sensitivity_weight_shock.csv
  - 2024_A_olympic_sde/data/processed/sde_matrix.csv
  - 2024_A_olympic_sde/data/processed/brisbane_candidates.csv
  - 2024_A_olympic_sde/paper/himcm_paper.tex
```

## 九行宇宙

文件：`historical_ranking.csv`（由 `sde_matrix.csv` + 锁定 AHP 算出）。第一名 Skateboarding，Norm_Score 0.6879。Cricket 在该表为第 9（0.2646）。**这不是「2032 推荐末位」**，只是九项对照下的回测位置。

## 三行宇宙

文件：`brisbane_2032_ranking.csv`。锁定 SAW：Flag 0.5360 第 1、Cricket 0.4842 第 2、Squash 0.3960 第 3。这是政策短名单的决策表。

## 逆转

±20% OAT 不改变三候选名次。`youth_weight_zero` 使 Squash 升至第 1、Flag 降至第 3（Δrank=−2）。九行宇宙下同一冲击把田径/游泳抬到前二、Breaking 从 2 落到 8。

## Related

- [SAW / TOPSIS](../methods/saw-topsis.md)
- [OAT](../methods/oat-sensitivity.md)
- [历史回测](../sources/historical-ranking.md)
- [2032 排序](../sources/brisbane-2032-ranking.md)
- [2023 影响因子三行宇宙](../sources/impact-factors.md)

# Source：historical_ranking.csv

九行宇宙锁定 AHP–SAW 回测榜。min-max 在 9 项上；前六名为历史/近年项，后三行为打了 2032 标签但仍在同一张表里的候选，**不可当成短名单名次**。

```yaml
type: source
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/results/historical_ranking.csv
  - 2024_A_olympic_sde/paper/himcm_paper.tex
```

## 摘录（论文四位）

| Final_Rank | SDE_Name | Code | Norm_Score |
| --- | --- | --- | --- |
| 1 | Skateboarding | SKB | 0.6879 |
| 2 | Breaking | BKG | 0.6560 |
| 3 | Athletics | ATH | 0.6350 |
| 4 | Swimming | SWM | 0.6279 |
| 5 | Surfing | SRF | 0.5997 |
| 6 | Karate | KTE | 0.5675 |
| 7 | Flag football | AFB | 0.5183 |
| 8 | Squash | SQU | 0.3826 |
| 9 | Cricket | CKT | 0.2646 |

论文摘要：近年城市项高于「一届空手道」实验，方向与东京后青年/城市编程一致。Cricket 第 9 只描述九行对照。

脚本：`src/scripts/run_eval.py` → `paper_figures/fig_historical_backtest.png`。

## Related

- [两套宇宙](../concepts/two-universes.md)
- [SAW](../methods/saw-topsis.md)
- [2032 排序](../sources/brisbane-2032-ranking.md)

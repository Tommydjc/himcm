# 一次一项权重扰动

在锁定 AHP 权上，按 IOC 六块分组做 ±20% 冲击后重新归一化；另设青年吸引力权重为 0。主结论：±20% 不改三候选名次；青年权归零会逆转。

```yaml
type: method
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/src/scripts/run_sensitivity.py
  - 2024_A_olympic_sde/results/sensitivity_weight_shock.csv
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
  - 2024_A_olympic_sde/paper/himcm_paper.tex
  - 2024_A_olympic_sde/tests/test_sensitivity.py
```

## 设计

- 宇宙：`brisbane_candidates`（三行）与 `sde_matrix`（九行）分开写行，禁止混读 Rank。
- ±20%：论文称六块准则 OAT；三候选宇宙下 Flag–Cricket–Squash 名次不变。
- `youth_weight_zero`：将 Relevance/Innovation（叶 `YOUTH_APPEAL`）权置 0 再归一。

三候选 `youth_weight_zero`（`sensitivity_weight_shock.csv`）：

| SDE | Baseline rank | Youth=0 rank | Score at youth=0 | Δrank |
| --- | --- | --- | --- | --- |
| Squash | 3 | 1 | 0.5077 | +2 |
| Cricket | 2 | 2 | 0.4092 | 0 |
| Flag football | 1 | 3 | 0.4051 | −2 |

基线分仍以 `brisbane_2032_ranking.csv` 为准：Flag 0.5360、Cricket 0.4842、Squash 0.3960。

九行宇宙同一冲击：Athletics、Swimming 升至 1–2，Breaking 从 2 落到 8（论文摘要；细节以 CSV 该 `Universe=sde_matrix` 段为准）。

## 代码入口

```bash
cd 2024_A_olympic_sde
PYTHONPATH=. python -m src.scripts.run_sensitivity
```

图：`paper_figures/fig_sensitivity_shock.png`。无随机数。

## Related

- [SAW / TOPSIS](../methods/saw-topsis.md)
- [两套宇宙](../concepts/two-universes.md)
- [敏感性 source](../sources/sensitivity-weight-shock.md)
- [2032 排序](../sources/brisbane-2032-ranking.md)

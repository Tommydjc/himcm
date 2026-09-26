# HiMCM 2024 A — Olympic SDE

2024 A 是评价题：把 IOC 纲领落成可算的因子，用锁定 AHP–SAW 做历史回测与布里斯班 2032 三候选排序，再用小样本 LOOCV 作描述性互证。决策引擎是 AHP–SAW，不是分类器。

```yaml
type: problem
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/paper/himcm_paper.tex
  - 2024_A_olympic_sde/paper/ioc_letter.tex
  - 2024_A_olympic_sde/results/brisbane_2032_ranking.csv
  - 2024_A_olympic_sde/results/historical_ranking.csv
  - 2024_A_olympic_sde/results/sensitivity_weight_shock.csv
  - 2024_A_olympic_sde/prompts/cursor_log.md
```

## 赛题要交什么

论文骨架见 `2024_A_olympic_sde/paper/himcm_paper.tex`（Task 1–5）与 `paper/ioc_letter.tex`（Task 6 政策信）：

1. **因子体系**：IOC 叙述六块 → 代码七叶（xlsx 两列 + 研究表五列）。见 [IOC 六块与七叶](../concepts/ioc-six-plus-xlsx.md)。
2. **历史回测**：九行宇宙 `historical_ranking.csv`。见 [历史回测表](../sources/historical-ranking.md)。
3. **2032 三候选排序**：Cricket / Squash / Flag football，min-max 只在这三行上。锁定 SAW：Flag 0.5360 第 1、Cricket 0.4842 第 2、Squash 0.3960 第 3。见 [2032 排序](../sources/brisbane-2032-ranking.md)。
4. **敏感性**：六准则 ±20% OAT；青年权归零会逆转三候选名次。见 [OAT](../methods/oat-sensitivity.md)。
5. **政策信**：名次与敏感性只引用上述 CSV，不把 ML 均值写成执委会终裁。
6. **ML 互证（可选透镜）**：N=6 LOOCV + B=500 自助区间 + AHP vs \(|\beta|\)。弱分类器，禁止写成已推翻 AHP。

本包禁止 RL / DES / 清场仿真。根目录 `paper/` 与 `2025_A_evacuation/` 不是本题方法。

## 题包地图

| 路径 | 用途 |
| --- | --- |
| `src/indicators/` | `schema.py` 因子契约；`loader.py` xlsx + 研究表 → processed |
| `src/mcda/` | AHP Perron 权；SAW / 轻量 TOPSIS |
| `src/ml/` | 无泄漏 dataset、LOOCV、importance、出图 |
| `src/scripts/` | `run_eval.py`、`run_sensitivity.py` |
| `data/raw/` | 官方 xlsx、`sde_factors_research.csv`、锁定判断矩阵 |
| `data/processed/` | `sde_matrix.csv`、`brisbane_candidates.csv` |
| `results/` | 论文数字唯一来源 |
| `paper/` `paper_figures/` | 文稿与由 CSV 生成的图 |
| `tests/` | 护公式与管道 |
| `prompts/cursor_log.md` | COMAP AI 披露底稿 |

## 研究做到哪一步

已 ingest、**不是** contest-ready：主决策表与敏感性已冻结；ML 只有 N=6，且 IOC 五列是研究赋分不是官方表。

顺序见 [2024 研究闭环](../playbooks/2024-a-study-loop.md)：IOC 六块 → 七叶 schema → AHP 锁定 CR=0 → loader → 历史回测 → 2032 SAW → OAT/青年归零 → 论文与 IOC 信 → 无泄漏 dataset → LOOCV+bootstrap → AHP vs \(|\beta|\) → 出图与诚实弱 LOOCV。

## Related

- [AHP 特征值](../methods/ahp-eigenvalue.md)
- [SAW / TOPSIS](../methods/saw-topsis.md)
- [两套宇宙](../concepts/two-universes.md)
- [复现命令](../playbooks/2024-repro-eval.md)
- [2032 排序 source](../sources/brisbane-2032-ranking.md)

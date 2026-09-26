# HiMCM 2023 A — Dandelion Prisms

2023 A 是机理/空间仿真题：西缘外侧一株绒球期蒲公英，在 1 公顷格网上做三气候扩散，并用锁定 SAW 算入侵影响因子。割草帕累托与 HOA 信是课程扩展，不是官方第三问。

```yaml
type: problem
year: 2023
status: draft
sources:
  - 2023_A_dandelion_prisms/paper/himcm_paper.tex
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2023_A_dandelion_prisms/results/impact_factors.csv
  - 2023_A_dandelion_prisms/results/bioeconomic_pareto.csv
  - 2023_A_dandelion_prisms/results/kpp_check.csv
  - 2023_A_dandelion_prisms/prompts/cursor_log.md
```

## 赛题要交什么

官方 PDF：COMAP HiMCM 2023 Problem A。

1. **Req 1 扩散**：1/2/3/6/12 月；温带 / 干旱 / 热带。权威表 `monthly_metrics.csv`。温带月末覆盖率 0 / 0.0027 / 0.0052 / 0.0729 / 0.2097，12 月锋面 46.5 m。
2. **官方 Req 2 影响因子**：三物种宇宙 min-max SAW。`impact_factors.csv`：葛藤 PUMON 0.9333，虎杖 REJAP 0.6935，蒲公英 TAROF 0.0000（表内最低，不是「自然界无害」）。
3. **课程扩展**：`bioeconomic_pareto.csv` 与 `paper/hoa_letter.tex`。双周 \(\eta=1\) 覆盖 0.0077、相对成本 39；每周同覆盖成本 78，被支配。

本包禁止把情景气候写成 NOAA 观测，禁止把 \(n_{\mathrm{plants}}\) 写成开花成株数（含幼苗）。

## 题包地图

| 路径 | 用途 |
| --- | --- |
| `src/physics/` | WALD + 2D 羽流 |
| `src/biology/` | Lefkovitch + 自疏 |
| `src/simulation/` | 100×100 m 卷积引擎 |
| `src/theory/` | Fisher–KPP 验算 |
| `src/decision/` | 影响因子与割草帕累托 |
| `src/scripts/` | 强迫展开、全量出表、出图 |
| `data/parameters/` | 生物物理与气候契约 |
| `results/` | 论文数字唯一来源 |

## 研究做到哪一步

已能一键出表出图，**不是** contest-ready：气候是情景，影响因子只有三行宇宙，论文未完稿排版。

## Related

- [WALD 羽流](../methods/wald-plume.md)
- [Lefkovitch 自疏](../methods/lefkovitch-self-thinning.md)
- [Fisher–KPP](../methods/fisher-kpp.md)
- [复现命令](../playbooks/2023-repro-sim.md)
- [月末指标 source](../sources/monthly-metrics.md)

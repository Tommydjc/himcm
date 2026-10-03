# HiMCM 2022 A — The Need for Bees

2022 A 是蜂群动力学 + 可微灵敏度 + 胁迫分岔 + 20 英亩授粉运筹题。权威数字只认题包 `results/*.csv`；论文图只从 `paper_figures/` 读真实输出。

```yaml
type: problem
year: 2022
status: ingested
sources:
  - 2022_A_honeybee_dynamics/results/ccd_healthy_vs_stress_snapshot.csv
  - 2022_A_honeybee_dynamics/results/autograd_elasticity_ranking.csv
  - 2022_A_honeybee_dynamics/results/ccd_tipping_point.csv
  - 2022_A_honeybee_dynamics/results/orchard_pollination_optimization.csv
  - 2022_A_honeybee_dynamics/results/orchard_pollination_recommendation.csv
  - 2022_A_honeybee_dynamics/paper/himcm_paper.tex
  - 2022_A_honeybee_dynamics/prompts/cursor_log.md
```

## 赛题要交什么

官方 PDF：COMAP HiMCM 2022 Problem A（*The Need for Bees*）。

1. **Req 1 群体动态**：全年蜂口与储蜜轨迹。健康基线（Torch DFM）夏季峰 \(H+F=66{,}507\)（day 238）、越冬 \(38{,}622\)（`ccd_healthy_vs_stress_snapshot.csv`）。
2. **Req 2 灵敏度**：产卵、寿命/死亡率等对目标的影响。Autograd 弹性表见 `autograd_elasticity_ranking.csv`。
3. **Req 3 授粉配箱**：官方地块 \(81{,}000\,\mathrm{m}^2\approx 20\) acre。扁桃园 OR 扫描 \(K\in[5,60]\)，黄金带 1–2 箱/英亩（\(K\in[20,40]\)）。
4. **Req 4 非技术页**：养蜂协会/种植园主咨询信（`paper/grower_advisory_letter.tex`）。

课程扩展：三重胁迫下 CCD 式分岔（`ccd_bifurcation_sweep.csv`）。

## 题包地图

| 路径 | 用途 |
| --- | --- |
| `src/autograd_engine/` | 可微 DFM 全年仿真 `torch_sim.py` |
| `src/sensitivity/` | Autograd 弹性排名 |
| `src/stress_test/` | 农药/瓦螨/霜冻 + \(\mu_F\) 分岔 |
| `src/pollination/` | 20 英亩扁桃园蜂箱密度 OR |
| `results/` | 论文数字唯一来源 |
| `paper_figures/` | 由脚本写出的图 |
| `paper/` | 主文、Summary、咨询信、AI 披露 |
| `tests/` | autograd / CCD / orchard 护栏 |
| `prompts/cursor_log.md` | COMAP AI 披露底稿 |

## 研究做到哪一步

DFM–Torch 基线、弹性表、CCD 软/硬拐点、果园 \(K\) 扫描与 LaTeX 套件均已有真实 CSV/PDF。**status=`ingested`，不是 contest-ready**：气候/花期多为情景；`colony_annual_dynamics.csv` 是早期 Khoury 三分室伴生表，勿与 Torch 健康峰混写成同一轨迹。

核对锚点（均来自上表 CSV）：

- \(J_{\mathrm{pop}}\) 弹性 Top3：\(L_{\max}\,+0.925\)，\(\mu_F\,-0.311\)，\(\sigma_F\,+0.294\)
- CCD：软拐点 \(\mu_F^*=0.11\)，硬拐点 \(0.305\,\mathrm{day}^{-1}\)
- 果园黄金带最优 \(K^\star=40\)，YieldRatio \(0.848\)，净收益 \$51{,}394

## Related

- [DFM Torch 仿真](../methods/dfm-torch-sim.md)
- [Autograd 弹性](../methods/autograd-elasticity.md)
- [CCD 分岔](../methods/ccd-bifurcation.md)
- [果园蜂箱 OR](../methods/orchard-hive-or.md)
- [2022 研究闭环](../playbooks/2022-a-study-loop.md)
- [健康/胁迫快照](../sources/ccd-healthy-vs-stress.md)

# 2022 A 研究闭环

按本节实际顺序检索：读题 → 可微 DFM → 弹性 → CCD → 果园 OR → 论文与披露。每步链 Raw，不把后一步数字提前写成「已证明」。

```yaml
type: playbook
year: 2022
status: ingested
sources:
  - 2022_A_honeybee_dynamics/prompts/cursor_log.md
  - 2022_A_honeybee_dynamics/paper/himcm_paper.tex
```

## 顺序

1. **读题 / 官方地块** → [2022 问题](../problems/2022-a-honeybee-dynamics.md)；Raw：`data/raw/2022_HiMCM_Problem_A.pdf`，面积 \(81{,}000\,\mathrm{m}^2\)。
2. **可微 DFM 五维** → [DFM Torch](../methods/dfm-torch-sim.md)；`src/autograd_engine/torch_sim.py`。
3. **Autograd 弹性** → [弹性方法](../methods/autograd-elasticity.md)；写出 `autograd_elasticity_ranking.csv`。
4. **三重胁迫 + \(\mu_F\) 分岔** → [CCD](../methods/ccd-bifurcation.md)；`ccd_tipping_point.csv`。
5. **20 英亩扁桃园 OR** → [果园 OR](../methods/orchard-hive-or.md)；`orchard_pollination_optimization.csv`。
6. **论文套件与咨询信** → `paper/himcm_paper.tex`、`summary_sheet.tex`、`grower_advisory_letter.tex`、`ai_disclosure.tex`；数字只粘 CSV。
7. **出图** → [CSV 到图](../playbooks/results-to-figures.md)；`paper_figures/fig_*.png`。
8. **Wiki ingest** → 本闭环与 source 页；不改模型、不改权威 CSV。

复现命令：[2022-repro-bee](../playbooks/2022-repro-bee.md)。禁令：[失败模式](../concepts/2022-a-failure-modes.md)。

## Related

- [2022 问题](../problems/2022-a-honeybee-dynamics.md)
- [COMAP 披露](../concepts/comap-ai-disclosure.md)

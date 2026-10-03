# 2023 A 研究闭环

按课上顺序检索：读题 → 羽流/生活史 → 周步出表 → KPP → 影响因子 → 帕累托 → 日步伴生 → 论文/HOA →（可选）PPO → Wiki。

```yaml
type: playbook
year: 2023
status: ingested
sources:
  - 2023_A_dandelion_prisms/prompts/cursor_log.md
  - 2023_A_dandelion_prisms/README.md
  - 2023_A_dandelion_prisms/paper/himcm_paper.tex
```

## 顺序

1. **读题 / 1 ha 西缘一株** → [2023 问题](../problems/2023-a-dandelion-prisms.md)；`README.md`。
2. **WALD 羽流 + Lefkovitch** → [WALD](../methods/wald-plume.md)、[Lefkovitch](../methods/lefkovitch-self-thinning.md)。
3. **周步卷积出 `monthly_metrics.csv`** → [月末指标](../sources/monthly-metrics.md)。
4. **Fisher–KPP 验算** → [KPP](../methods/fisher-kpp.md)；`kpp_check.csv` rel_err \(0.0521\)。
5. **三物种影响因子** → [impact source](../sources/impact-factors.md)。
6. **割草帕累托 + HOA** → [帕累托](../methods/bioeconomic-pareto.md)、`paper/hoa_*.tex`。
7. **日步中心宇宙（伴生）** → [日步 source](../sources/monthly-population-metrics.md)；与周步分列。
8. **论文套件** → 数字只抄当前 CSV；拒用过期 0.2097 / TAROF=0 示例。
9. **（可选）PPO 扩展** → `results/rl_policy_evaluation.csv`；不写入官方主结论。
10. **Wiki ingest** → 本闭环；不改权威 CSV。

复现：[2023-repro-sim](../playbooks/2023-repro-sim.md)。禁令：[失败模式](../concepts/2023-a-failure-modes.md)。

## Related

- [2023 问题](../problems/2023-a-dandelion-prisms.md)
- [COMAP 披露](../concepts/comap-ai-disclosure.md)

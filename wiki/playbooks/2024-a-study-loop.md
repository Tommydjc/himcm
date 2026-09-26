# 2024 A 研究闭环

按课上实际顺序检索：从读题到诚实弱 LOOCV 出图。每一步只链 Raw 与 Wiki，不把后一步的数字提前写成「已证明」。

```yaml
type: playbook
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/prompts/cursor_log.md
  - 2024_A_olympic_sde/paper/himcm_paper.tex
  - 2024_A_olympic_sde/paper/sections/sec4_ml_validation.tex
```

## 顺序

1. **读题 / IOC 六块** → [IOC 六块与七叶](../concepts/ioc-six-plus-xlsx.md)；Raw：`paper/himcm_paper.tex` Introduction。
2. **schema 七叶** → `src/indicators/schema.py`（`INDICATORS`，HOST 解耦）。
3. **AHP 锁定 CR=0** → [AHP](../methods/ahp-eigenvalue.md)；Raw：`data/raw/ahp_judgment_locked.csv`、`src/mcda/ahp.py`。
4. **loader：xlsx + 研究表** → `src/indicators/loader.py`、`data/raw/README_FACTORS.md` → `data/processed/sde_matrix.csv`。
5. **历史回测** → [SAW](../methods/saw-topsis.md)、[历史表](../sources/historical-ranking.md)；`src/scripts/run_eval.py`。
6. **2032 三候选 SAW** → [两套宇宙](../concepts/two-universes.md)、[2032 表](../sources/brisbane-2032-ranking.md)；loader 同时导出 `brisbane_candidates.csv`。
7. **OAT / 青年归零** → [OAT](../methods/oat-sensitivity.md)；`src/scripts/run_sensitivity.py`。
8. **论文 Task 1–5 与 IOC 信** → `paper/himcm_paper.tex`、`paper/ioc_letter.tex`；数字只粘 CSV。
9. **dataset 无泄漏** → `src/ml/dataset.py`（折内 scaler、候选惯性 0）。
10. **LOOCV + bootstrap** → [LOOCV](../methods/loocv-logistic.md)；B=500 seed=42。
11. **AHP vs |β|** → [对照方法](../methods/ahp-vs-beta.md)；`src/ml/importance.py`。
12. **出图与 sec4** → [CSV 到图](../playbooks/results-to-figures.md)；写明 Acc=0.3333 而非假高准确率。

复现命令：[2024-repro-eval](../playbooks/2024-repro-eval.md)。禁令：[失败模式](../concepts/2024-a-failure-modes.md)。

## Related

- [2024 问题](../problems/2024-a-olympic-sde.md)
- [COMAP 披露](../concepts/comap-ai-disclosure.md)

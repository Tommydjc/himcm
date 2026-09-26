# 2024 A 不要再走的路

从 `prompts/cursor_log.md`、`dataset.py` 与论文 sec4 归纳的禁令。本页不引用退役脚本数字当结果。

```yaml
type: concept
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/prompts/cursor_log.md
  - 2024_A_olympic_sde/src/ml/dataset.py
  - 2024_A_olympic_sde/paper/sections/sec4_ml_validation.tex
  - 2024_A_olympic_sde/data/raw/README_FACTORS.md
  - 2024_A_olympic_sde/src/mcda/ahp.py
```

## 清单

1. **不要 80/20 切分** 冒充验证；标签只有 N=6，必须 LOOCV。
2. **不要全局 `scaler.fit`**；折内拟合，否则泄漏。
3. **不要随机判断矩阵**；锁定向量外积，CR=0。
4. **不要把研究赋分写成 IOC 官方表**；xlsx 只有 Appearances / Events_Count。
5. **不要用九行名次当 2032 推荐**；Cricket 在历史表第 9 ≠ 短名单末位。
6. **不要宣称 ML 准确**；Acc=0.3333、AUC=0.5000、Brier=0.2475。
7. **不要用 Cricket 的 logistic 均值改写决策引擎**；SAW 第一仍是 Flag football。
8. **不要把 HOST 塞进七准则综合分**。
9. **不要把 2025 清场时间、RL、DES 写进 2024 页**。
10. **不要用空值编造 BSB/SBL 的 IOC 六列**。

## Related

- [两套宇宙](../concepts/two-universes.md)
- [LOOCV](../methods/loocv-logistic.md)
- [AHP](../methods/ahp-eigenvalue.md)
- [研究闭环](../playbooks/2024-a-study-loop.md)

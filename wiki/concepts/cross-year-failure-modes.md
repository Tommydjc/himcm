# 跨年失败模式（赛时禁令总表）

题型无关的禁令清单。分年细节见各年 failure-modes；本页供 Contest Hub 一页自检。

```yaml
type: concept
year: cross
status: contest-ready
sources:
  - wiki/concepts/2022-a-failure-modes.md
  - wiki/concepts/2023-a-failure-modes.md
  - wiki/concepts/2024-a-failure-modes.md
  - wiki/comparisons/synthetic-data-trap.md
  - wiki/concepts/two-universes.md
  - wiki/concepts/comap-ai-disclosure.md
  - AGENTS.md
```

## 清单（15 条）

1. **禁止串年数字**：不得把旧年 CSV 锚点写进新论文。
2. **禁止两套宇宙横比**：历史回测表 ≠ 决策短名单；周步 ≠ 日步；Torch ≠ 伴生简化表。
3. **禁止合成数据当真**：`SYNTHETIC_FALLBACK` / 手搓 Likert 只能写 Limitations。
4. **禁止无 CSV 先写结论**：先锁 `results/*.csv`，再写 25 页。
5. **禁止训练指标当评测**：return / loss ≠ \(t_{\mathrm{clear}}\) / Acc / LPSP。
6. **禁止发现=救出**（搜救类）：挂牌/发现与人员结局分列。
7. **禁止基线未通就上深度模型**：规划器 / AHP / EFA / MILP 永远先于 GAT–PPO / 长训 Crew。
8. **禁止第三套方法超时**：120–180 窗砍枝，不砍证据。
9. **禁止情景当观测**：气候/花期情景须标明 scenario。
10. **禁止小样本装高精度**：N=6 LOOCV 等弱结果不得改写决策引擎。
11. **禁止全局 scaler 泄漏**：折内拟合。
12. **禁止随机判断矩阵冒充 AHP**：锁定矩阵并报告 CR。
13. **禁止 Agent 自动覆盖权威 CSV**：人审门闩后才 promote。
14. **禁止省略号占位代码**：完整可运行块 + Type Hints。
15. **禁止不披露 AI**：关键回合写入当年 `prompts/cursor_log.md`。

## 分年细页

- [2022](2022-a-failure-modes.md) · [2023](2023-a-failure-modes.md) · [2024](2024-a-failure-modes.md)
- [合成数据陷阱](../comparisons/synthetic-data-trap.md) · [两套宇宙](two-universes.md)

## Related

- [Contest Hub](../playbooks/contest-hub.md)
- [题型路由](../playbooks/problem-type-router.md)
- [COMAP 披露](comap-ai-disclosure.md)

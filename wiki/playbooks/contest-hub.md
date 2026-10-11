# Contest Hub（赛时总控台）

开赛 0–30 分钟只读本页与题干：输出题型模式、基线方法、砍枝清单与 Wiki 链接。完整目录见 [index](../index.md)；数字仍回当年 `results/*.csv`。

```yaml
type: playbook
year: cross
status: contest-ready
sources:
  - AGENTS.md
  - STUDIES.md
  - wiki/playbooks/4h-combat-playbook.md
  - wiki/playbooks/problem-type-router.md
```

## 固定口令（贴给 Agent）

```text
先读 wiki/playbooks/contest-hub.md 与 problem-type-router.md。
输出：主模式 A–F（可多选）+ 基线方法 + 砍枝清单 + 对应 wiki 链接。
禁止引用其他年份 results/*.csv 中的数字。
基线先于深度模型；权威数字只认当年题包 CSV。
```

## 赛时必读（最多 6 页）

| 顺序 | 页 | 用途 |
| --- | --- | --- |
| 1 | [题型路由](problem-type-router.md) | 模式 A–F → 基线 |
| 2 | [4h 交卷](4h-combat-playbook.md) | 时间盒与砍枝 |
| 3 | [跨年失败模式](../concepts/cross-year-failure-modes.md) | 禁令自检 |
| 4 | [规格驱动编码](spec-driven-cursor-coding.md) | 代码纪律 |
| 5 | [Agent 栈职责](../comparisons/agent-stack-roles.md) | Remit / MM / Cursor / Lean |
| 6 | [CSV 到图](results-to-figures.md) | 图只读 results |

论文窗加读：[ASD-STE100](asd-ste100-academic-writing.md)、[COMAP 披露](../concepts/comap-ai-disclosure.md)。

## 开题动作（0–30 min）

1. 复制 `_template/` → `YYYY_P_slug/`（在题包内建 `src/`、`prompts/`、`paper/`）。
2. 导入题面到 `data/raw/`（只读，不覆盖）。
3. 按路由填假设表：主模式 / 辅模式 / 基线验收指标。
4. 若用 Remit / MM Skill：计划以本 Hub 输出为**唯一终裁**，见旁路 playbook。

## 时间盒（摘要）

| 窗 | 做什么 |
| --- | --- |
| 0–30 | 路由 + 假设；不写 RL 长训 |
| 30–120 | 基线 → `results/*.csv` |
| 120–180 | 敏感性 / 一图一表 |
| 180–230 | 论文；可选 1 条 Lean 引理 |
| 230–240 | PDF、AI 披露、打包 |

## 验收

- 计划含模式 ID 与基线方法链接。
- 至少一份当年可复现 CSV。
- `prompts/cursor_log.md` 有本场条目。
- 未把旧年锚点数字写进新论文。

## Related

- [题型路由](problem-type-router.md)
- [4h 交卷](4h-combat-playbook.md)
- [Remit 旁路](remit-assisted-loop.md)
- [MM-Agent 旁路](mm-agent-assisted-loop.md)
- [证据门闩](evidence-gate.md)
- [Lean 引理](lean-lemma-loop.md)
- [目录 index](../index.md)

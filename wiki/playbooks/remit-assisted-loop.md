# Remit 旁路协作闭环

Remit 是仓外工作台：团团/灵灵/点点/墨墨协作 + 人审 + 检查点。权威 CSV 仍只在本仓题包；详见 `tools/remit_bridge/README.md`。

```yaml
type: playbook
year: cross
status: contest-ready
sources:
  - tools/remit_bridge/README.md
  - wiki/playbooks/contest-hub.md
  - wiki/playbooks/4h-combat-playbook.md
  - AGENTS.md
```

## 步骤

1. **仓外**启动 Remit（Docker 或官方 `win_start.bat`），勿 submodule 进本仓根目录。
2. 新建项目；导入当年题包的 `data/raw/`（及空 `src/` 若需要）。
3. 将 [Contest Hub](contest-hub.md) 固定口令贴进项目说明；挂载只读 Wiki 技能（playbook + failure-modes，**不含**旧年 `results/*.csv`）。
4. 团团出计划 → **人确认**（须含模式 A–F + 基线）。
5. 灵灵建模对照 Wiki method；点点改代码提案 → 人审后写入 `YYYY_P_slug/src/`。
6. 跑通后产物进 `results/`（仅人执行 promote）；墨墨写 `paper/`，图只读 CSV。
7. 关键回合追加当年 `prompts/cursor_log.md`（可摘要 Remit 对话）。

## 与 4h 时间盒

| 窗 | Remit 角色 |
| --- | --- |
| 0–30 | 团团 + 人确认计划 |
| 30–120 | 点点 + 本仓脚本；检查点保存基线 |
| 120–180 | 探索实验；失败回退检查点 |
| 180–230 | 墨墨论文工作区 |
| 230–240 | 导出 PDF；本仓打包与披露 |

## 验收

- 计划与 Contest Hub 模板一致。
- 无「未审核自动覆盖 results」。
- [证据门闩](evidence-gate.md) 抽检通过（若已跑）。

## Related

- [Contest Hub](contest-hub.md)
- [Agent 栈职责](../comparisons/agent-stack-roles.md)
- [4h 交卷](4h-combat-playbook.md)
- [证据门闩](evidence-gate.md)

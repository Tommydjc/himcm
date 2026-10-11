# MM-Agent 旁路（许可门闩后）

MM-Agent / Expert Skill 仅作开题与方法候选外脑。默认**不** vendoring 其源码；启用前须完成 `tools/mm_bridge/README.md` 中的许可检查。

```yaml
type: playbook
year: cross
status: draft
sources:
  - tools/mm_bridge/README.md
  - wiki/comparisons/hmml-vs-gohimcm.md
  - wiki/playbooks/contest-hub.md
  - AGENTS.md
```

## 启用条件

- [ ] 已阅读上游 LICENSE / README，确认课程或竞赛使用场景合法。
- [ ] `tools/mm_bridge/LICENSE_GATE.md` 中结论为 `ALLOWED` 或 `SKILLS_ONLY`。
- [ ] 计划终裁仍在 Contest Hub / 人确认（非 MM 自动计划）。

未勾选则**只使用本仓 Wiki 路由**，不要跑 MM 流水线。

## 步骤（放行后）

1. 仓外克隆或安装 Expert Skill（见 mm_bridge README）。
2. 输入：新题文本 + 可选 `data/raw`；**不**喂旧年 results。
3. 输出只写入 `YYYY_P_slug/paper/notes/`（分析草稿）。
4. 用 [题型路由](problem-type-router.md) 与 [HMML 映射](../comparisons/hmml-vs-gohimcm.md) 过滤方法候选。
5. 冲突以 [跨年失败模式](../concepts/cross-year-failure-modes.md) 否决。
6. 编码与 CSV 仍走本仓 / Remit 点点路径；记 `prompts/cursor_log.md`。

## 验收

- notes 有草稿；`results/` 无 MM 直接写入。
- 选用方法能在 Wiki method 中找到入口或标 `unverified` 新方法。

## Related

- [Agent 栈职责](../comparisons/agent-stack-roles.md)
- [HMML 映射](../comparisons/hmml-vs-gohimcm.md)
- [Contest Hub](contest-hub.md)

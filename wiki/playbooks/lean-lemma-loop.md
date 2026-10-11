# Lean 引理闭环（可选 P4）

赛前可练；赛中论文窗最多塞 **1** 条短引理。不通过则改措辞为启发式/数值观察，禁止写「已证明」。

```yaml
type: playbook
year: cross
status: contest-ready
sources:
  - wiki/concepts/formal-methods-boundary.md
  - _template/formal/README.md
  - AGENTS.md
```

## 步骤

1. 从论文草稿抽出 1 条「声称必真」的命题与假设列表 → `paper/notes/lemmas.md`。
2. 对照 [形式化边界](../concepts/formal-methods-boundary.md)：经验分类则停止。
3. 在题包 `formal/`（从 `_template/formal/` 复制）写 Lean 4 骨架。
4. 本地 `lake build` 通过为唯一标准。
5. 正文/附录加 Formal statement + 文件指针；记 `prompts/cursor_log.md`。

## 验收

- `lake build` 成功，或明确放弃并改论文措辞。
- 未把 CSV 数字写成定理。

## Related

- [形式化边界](../concepts/formal-methods-boundary.md)
- [Contest Hub](contest-hub.md)
- [4h 交卷](4h-combat-playbook.md)

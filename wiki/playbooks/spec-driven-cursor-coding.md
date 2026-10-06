# Spec-driven Cursor coding

规格先于补全：Type Hints、Docstring、完整可运行块。禁止 `# ... existing code ...`。根规则：`.cursorrules`；题包 glob 规则不覆盖根 2025 疏散规则。

```yaml
type: playbook
year: cross
status: contest-ready
sources:
  - .cursorrules
  - AGENTS.md
  - _template/README.md
  - 2020_A_summer_job_factor/.cursorrules
```

## 步骤

1. 在题包内新建模块，不在仓库根放业务 `src/`。
2. 环境 / 规划器 / RL（或 generation / MILP / agent）分层。
3. 改代码后跑题包 `pytest`；图只读 `results/`。
4. 关键算法回合追加当年 `prompts/cursor_log.md`。
5. Ingest 只写 `wiki/`，不改权威 CSV。

## 验收

grep 生成补丁无 ellipsis 占位；函数有参数维数说明。

## Related

- [4h 交卷](4h-combat-playbook.md)
- [Notebook vs 模块](../comparisons/notebook-hell-vs-modular.md)
- [ASD-STE100](asd-ste100-academic-writing.md)

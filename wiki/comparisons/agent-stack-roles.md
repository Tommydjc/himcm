# Agent 栈职责对照

GoHiMCM Wiki、Remit、MM-Agent、Cursor Skill、Lean 各管一层。赛时单一计划源：Contest Hub / 团团计划。

```yaml
type: comparison
year: cross
status: contest-ready
sources:
  - AGENTS.md
  - wiki/playbooks/contest-hub.md
  - tools/remit_bridge/README.md
  - tools/mm_bridge/README.md
```

## 对照

| 组件 | 负责 | 不负责 | 何时用 |
| --- | --- | --- | --- |
| **GoHiMCM Wiki** | 题型路由、方法规格、禁令、playbook | 跑仿真、当第二数据表 | 始终；Query 先读 Hub |
| **题包 `YYYY_P_slug/`** | 权威 `src/` + `results/*.csv` + 终稿 | 托管外部 Agent 内核 | 始终 |
| **Remit** | 多 Agent 协作 UI、人审、检查点、LaTeX 工作区 | 定义权威数字；自动 promote CSV | 课程班 / 需要检查点时 |
| **MM-Agent / Expert Skill** | 开题分析、HMML 式方法候选、代码草稿 | 绕过 Wiki；未审写 results | **许可放行后**；仅 notes 草稿 |
| **Cursor + 本仓规则** | 规格驱动改题包代码 | 替代人终裁 | 默认开发路径 |
| **Lean `formal/`** | 1–3 条纯数学引理核验 | 证明 CSV 经验结果 | 可选附录 |

## 单一计划源

```text
Contest Hub 输出（或 Remit 团团经人确认的计划）
    ↑ 可吸收
MM-Agent / Skill 建议（冲突则以 Wiki failure-modes 否决）
```

禁止两套终裁计划并行写同一 `results/`。

## 开源指针（仓外）

- Remit：<https://github.com/zhou2030109-glitch/Remit>（MIT；旁路）
- MM-Agent：<https://github.com/usail-hkust/LLM-MM-Agent>（先读 `tools/mm_bridge/README.md` 许可门闩）
- openai/math：<https://github.com/openai/math>（Lean 组织理念；非数模方法库）

## Related

- [Contest Hub](../playbooks/contest-hub.md)
- [Remit 旁路](../playbooks/remit-assisted-loop.md)
- [MM 旁路](../playbooks/mm-agent-assisted-loop.md)
- [HMML 映射](hmml-vs-gohimcm.md)
- [形式化边界](../concepts/formal-methods-boundary.md)

# HMML 与 GoHiMCM 模式映射

MM-Agent 的 Hierarchical Mathematical Modeling Library（Domain → Subdomain → Method）映射到本仓模式 A–F 与 method 页。吸收可赛时子集；过泛方法标「默认不用」。

```yaml
type: comparison
year: cross
status: draft
sources:
  - STUDIES.md
  - wiki/playbooks/problem-type-router.md
  - tools/mm_bridge/README.md
  - wiki/synthesis.md
```

## 层级对应

| HMML（概念） | GoHiMCM |
| --- | --- |
| Domain | 模式 A–F / STUDIES 题型 |
| Subdomain | method 主题簇（评价、动力学、空间、优化…） |
| Method node | 单篇 `wiki/methods/*.md`（公式+命令+验收+失败模式） |

## 模式映射（摘要）

| GoHiMCM | HMML 侧常见域（示意） | 本仓优先 method |
| --- | --- | --- |
| A 评价 | MCDM / ranking | AHP、SAW/TOPSIS、OAT |
| B 潜变量 | psychometrics / classification | EFA、熵权 |
| C 资源优化 | OR / energy systems | MILP 选型、KPI；Crew 仅协议层 |
| D 动力学 | dynamical systems / sensitivity | DFM–Torch、Autograd 弹性 |
| E 空间扩散 | spatial ecology / PDE | WALD、Lefkovitch、Fisher–KPP |
| F 图仿真 | search / multi-agent | 规划器 \(t_{\mathrm{clear}}\)；RL 分表 |

具体上游节点名随 MM-Agent 版本变化；以本表**题型意图**为准，不硬编码 98 节点编号。

## 赛时规则

- 候选方法必须能链到 Wiki 或显式新建 method（draft）。
- 数字与案例仍来自当年题包，不来自 HMML 叙述。
- 许可未放行时本页只作教学对照，不启用 MM 流水线。

## Related

- [题型路由](../playbooks/problem-type-router.md)
- [MM 旁路](../playbooks/mm-agent-assisted-loop.md)
- [Agent 栈职责](agent-stack-roles.md)

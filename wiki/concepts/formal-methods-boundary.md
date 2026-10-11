# 形式化边界（Lean vs CSV）

Lean 核验「在假设下必真」的引理；CSV 证经验结果。两者不可互换。理念可参考 openai/math 的「已形式化 / 未形式化」分级，但不要把其纯数定理库当数模方法库。

```yaml
type: concept
year: cross
status: contest-ready
sources:
  - wiki/playbooks/lean-lemma-loop.md
  - AGENTS.md
  - _template/formal/README.md
```

## 可以 Lean

- 权重归一化、简单不等式与代数恒等。
- 离散不变量（如已访集合单调）。
- 简化模型下的局部性质（须写清假设 H1–Hk）。

## 不可以 Lean

- \(t_{\mathrm{clear}}\)、SAW 得分、cover、LPSP 等**跑出来的数**。
- 「RL 优于基线」类实验主张。
- 情景气候、合成问卷、研究赋分等数据选择。

## 论文句式

**弱：**「因此权重归一且排序稳定。」  
**强：**「在 H1–H3 下 Lemma 2 保证…；Lean 见 `formal/`。经验排序以 `results/*.csv` 为准。」

## Related

- [Lean 引理闭环](../playbooks/lean-lemma-loop.md)
- [Contest Hub](../playbooks/contest-hub.md)
- [跨年失败模式](cross-year-failure-modes.md)

# Notebook hell vs modular packs

单体 `.ipynb` 把环境、规划器、画图缠在一起，长跑会把核内存打爆，且无法 pytest。本仓按 `_template/README.md` 分层。

```yaml
type: comparison
year: cross
status: ingested
sources:
  - _template/README.md
  - AGENTS.md
  - 2025_A_evacuation/src/traditional_planner/scoring_planner.py
  - 2022_A_honeybee_dynamics/src/autograd_engine/torch_sim.py
  - 2021_A_solar_storage/src/optimization/sizing_milp.py
```

## 对照

| 反模式 | 题包做法 |
| --- | --- |
| 一格跑 8760 h + 画 20 图 | `src/` 脚本写 CSV，`paper_figures/` 另读 |
| 根目录 `src/` | 仅 `YYYY_P_slug/src/` |
| RL 与环境互相 import 副作用 | 2025 `environment` / `traditional_planner` / `rl` 分目录 |
| 改 CSV「让图好看」 | 只增不改已 ingest 证据 |

## Related

- [规格驱动编码](../playbooks/spec-driven-cursor-coding.md)
- [2025 疏散](../problems/2025-a-evacuation.md)
- [2022 蜂群](../problems/2022-a-honeybee.md)

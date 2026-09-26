# HiMCM Studies — 十年考题总览

仓名 **GoHiMCM**；根目录为 Studies 枢纽，题包按年分目录。

理念：[Karpathy LLM Wiki](llm-wiki.md) → Schema：[AGENTS.md](AGENTS.md)

---

## 题包状态

| 目录 | 年份·题 | 题型 | Wiki problem | 状态 |
| --- | --- | --- | --- | --- |
| [2025_A_evacuation/](2025_A_evacuation/) | 2025 A Evacuation Sweeps | 仿真 / 图 MDP / 规划+RL | [2025-a-evacuation](wiki/problems/2025-a-evacuation.md) | ingested |
| [2024_A_olympic_sde/](2024_A_olympic_sde/) | 2024 A Olympic SDE | 评价 / MCDM + ML 互证 | [2024-a-olympic-sde](wiki/problems/2024-a-olympic-sde.md) | ingested |
| [2023_A_dandelion_prisms/](2023_A_dandelion_prisms/) | 2023 A Dandelions | 机理 / 空间扩散 + 影响因子 | [2023-a-dandelion-prisms](wiki/problems/2023-a-dandelion-prisms.md) | draft |
| [_template/](_template/) | — | 空壳 | — | ready |

新开一年：

```text
Copy-Item -Recurse _template YYYY_P_slug
课上填 raw → 冻结 → ingest YYYY_P_slug
```

---

## 知识库入口

| 文件 | 用途 |
| --- | --- |
| [wiki/index.md](wiki/index.md) | 赛时 / Query **先读** |
| [wiki/synthesis.md](wiki/synthesis.md) | 跨年方法总论 |
| [wiki/log.md](wiki/log.md) | Ingest / Query / Lint 时间线 |
| [docs/course/weekly_loop.md](docs/course/weekly_loop.md) | 每周课堂闭环 |

---

## 分层提醒

- 人写题包，LLM 写 Wiki。
- 赛时查 Wiki，在新题包里重做。
- 定量结论必须能回到 `YYYY_P_slug/results/*.csv`。

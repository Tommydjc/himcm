# HiMCM Studies — 十年考题总览

仓名 **GoHiMCM**；根目录为 Studies 枢纽，题包按年分目录。

理念：[Karpathy LLM Wiki](llm-wiki.md) → Schema：[AGENTS.md](AGENTS.md)

---

## 题包状态

| 目录 | 年份·题 | 题型 | Wiki problem | 状态 |
| --- | --- | --- | --- | --- |
| [2025_A_evacuation/](2025_A_evacuation/) | 2025 A Evacuation Sweeps | 仿真 / 图 MDP / 规划+RL | [2025-a-evacuation](wiki/problems/2025-a-evacuation.md) | ingested |
| [2024_A_olympic_sde/](2024_A_olympic_sde/) | 2024 A Olympic SDE | 评价 / MCDM + ML 互证 | [2024-a-olympics](wiki/problems/2024-a-olympics.md) · [细页](wiki/problems/2024-a-olympic-sde.md) | ingested |
| [2023_A_dandelion_prisms/](2023_A_dandelion_prisms/) | 2023 A Dandelions | 机理 / 空间扩散 | [2023-a-dandelion](wiki/problems/2023-a-dandelion.md) · [细页](wiki/problems/2023-a-dandelion-prisms.md) | ingested |
| [2022_A_honeybee_dynamics/](2022_A_honeybee_dynamics/) | 2022 A Need for Bees | 动力学 / Autograd / 授粉 OR | [2022-a-honeybee](wiki/problems/2022-a-honeybee.md) · [细页](wiki/problems/2022-a-honeybee-dynamics.md) | ingested |
| [2021_A_solar_storage/](2021_A_solar_storage/) | 2021 A Solar storage | 8760 h + MILP + CrewAI | [2021-a-solar-storage](wiki/problems/2021-a-solar-storage.md) | ingested |
| [2020_A_summer_job_factor/](2020_A_summer_job_factor/) | 2020 A Summer job | EFA + LLMFactor + 熵权 | [2020-a-summer-job](wiki/problems/2020-a-summer-job.md) | ingested |
| [_template/](_template/) | — | 空壳 | — | ready |

新开一年：

```text
Copy-Item -Recurse _template YYYY_P_slug
课上填 raw → 冻结 → ingest YYYY_P_slug
```

---

## 技能树全景矩阵

| 技能 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| 评价 / 权重 | 熵权 | LCOS | 果园 OR | SAW 影响因子 | AHP–SAW | 规划器评分 |
| 统计 / 小样本 | EFA / LOOCV | MCMC 负荷 | Autograd \(S_\theta\) | KPP 校核 | N=6 LOOCV | 重复 \(t_{\mathrm{clear}}\) |
| 空间 / 图 | — | 8760 时序 | 日步 ODE | Conv2D CA | — | 房间图 |
| 可微 / 优化 | MLP LOOCV | 选型 MILP | DFM–Torch | — | — | GAT–PPO（代码） |
| 多智能体 | — | CrewAI DSR | — | — | — | 2–5 搜救体 |
| LLM 语义 | LLMFactor | 本地谈判 | — | — | — | — |
| 数据纪律 | SYNTHETIC_FALLBACK | CSV>论文草稿 | Torch≠Khoury | 周步≠日步 | ML≠终裁 | 发现≠救出 |

方法入口：[EFA](wiki/methods/factor-analysis-llmfactor.md) · [熵权](wiki/methods/entropy-weight-objective-scoring.md) · [CrewAI](wiki/methods/crewai-cognitive-demand-response.md) · [Autograd](wiki/methods/pytorch-autograd-sensitivity.md) · [Conv2D](wiki/methods/conv2d-spatiotemporal-ca.md) · [AHP](wiki/methods/ahp-eigenvalue.md)

Playbook：[4h](wiki/playbooks/4h-combat-playbook.md) · [Cursor 规格](wiki/playbooks/spec-driven-cursor-coding.md) · [STE100](wiki/playbooks/asd-ste100-academic-writing.md)

反模式：[合成数据](wiki/comparisons/synthetic-data-trap.md) · [Notebook](wiki/comparisons/notebook-hell-vs-modular.md) · [被动停电](wiki/comparisons/passive-blackout-fallacy.md)

---

## CSV 锚点（核对用，非第二数据表）

| 年 | 锚点 | 路径 |
| --- | --- | --- |
| 2025 | office 2-agent \(t_{\mathrm{clear}}=135\) | `2025_A_evacuation/results/experiment_summary.csv` |
| 2024 | Flag SAW 0.5360 | `2024_A_olympic_sde/results/brisbane_2032_ranking.csv` |
| 2023 | 温带 12 月 cover 0.4511 | `2023_A_dandelion_prisms/results/monthly_metrics.csv` |
| 2022 | \(K^\star=40\) YieldRatio 0.848 | `2022_A_honeybee_dynamics/results/orchard_pollination_recommendation.csv` |
| 2021 | capex \$9700，LPSP 0.2489，2725 h | `2021_A_solar_storage/results/optimal_sizing_solution.csv` |
| 2020 | KMO 0.445，\(w=(0.479,0.319,0.202)\)，Top-1 0.20 | `2020_A_summer_job_factor/results/` |

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
- 2020 问卷宽表是 `SYNTHETIC_FALLBACK`，禁止套模板 KMO 0.782。

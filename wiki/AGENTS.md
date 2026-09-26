# AGENTS.md — GoHiMCM Schema（课程仓库契约）

本文件是 **Schema 层**：Agent 与学生都必须遵守。若某次改动改变了目录、页类型或 ingest 规则，必须在**同一次提交**里更新本文件。

仓库对外名称保持不变。根目录只放 Studies 枢纽，**不要**把年份嵌进根级 `src/`、`paper/`、`results/`。

## 1. 三层

| 层 | 谁写 | 规则 |
|---|---|---|
| Raw | 学生（题包） | `YYYY_P_slug/results/*.csv` 是论文数字的唯一来源。已经 ingest 进 Wiki 的表只增不改。 |
| Wiki | 仅 Agent | 人只读。页类型封闭。相对链接必须带 `.md`。 |
| Schema | 本文件 | 目录、ingest / query / lint、题包契约。 |

## 2. 根目录允许出现的内容

- `README.md` `STUDIES.md` `AGENTS.md` `llm-wiki.md` `.cursorrules`
- `_template/` 单题空壳
- `wiki/` Agent 知识库
- `docs/course/` 课表与周循环
- `tools/` 课程级脚本（开题、lint），**不是**某一年的模型代码
- `YYYY_P_slug/` 一年一题
- 共享 `venv/`（可选）

禁止在根目录新建建模用 `src/`、`paper/`、`results/`。

## 3. 题包契约 `YYYY_P_slug/`

从 `_template/` 复制，不要改仓名、不要把上一年整题拷进 `legacy/`。

| 路径 | 约定 |
|---|---|
| `data/raw/` | 只读：原始赛题 PDF/文本、公开表。Agent 与清洗脚本不得覆盖。 |
| `data/processed/` | 清洗、对齐后的中间表。 |
| `src/` | 按**题型**自定，不要写死三套文件夹同时存在。见下。 |
| `experiments/` | 只写「怎么跑」。不得手填权威数字。 |
| `results/` | 权威 CSV / 数组。论文与 Wiki 定量结论只许引用这里。 |
| `logs/` | 运行日志。 |
| `tests/` | 护公式与不变量。 |
| `paper/` `paper_figures/` | 文稿与由 CSV 生成的图。 |
| `prompts/` | 本题 AI 披露日志。 |
| `legacy/` | **本题**退役代码片段，不是上一年整包。 |

### `src/` 题型建议（择一，可增减模块，禁止混层）

- 仿真 / 搜救：`environment` / `traditional_planner` / `rl`
- 评价 / MCDM：`indicators` / `mcda` /（可选）`ml` / `sensitivity`
- 机理 / 优化：`model` / `solver` / `sensitivity`

Python 3.10+；函数与类要有 Type Hints 与 Docstring；禁止用省略号占位；禁止伪造 CSV。

## 4. Wiki 页类型（只允许下列六种）

`problem` | `method` | `concept` | `playbook` | `comparison` | `source`

对应目录：`wiki/problems/` `wiki/methods/` `wiki/concepts/` `wiki/playbooks/` `wiki/comparisons/` `wiki/sources/`。

每页必须包含：

1. 一级标题 H1
2. 紧随其后的一段摘要（普通段落，不是列表）
3. 一个 yaml 代码块，字段：`type` / `year` / `status` / `sources`
4. 文内相对链接带 `.md`

`sources` 必须带题包前缀，例如 `2025_A_evacuation/results/experiment_summary.csv`。

禁止发明数字。没有 raw/CSV 引用的定量结论必须标 `unverified`。

枢纽页（仍由 Agent 维护）：`wiki/index.md` `wiki/log.md` `wiki/synthesis.md`。

## 5. 操作

### Ingest

读题包 `data/raw` 与 `results/*.csv`，更新 5–15 个相关 Wiki 页，刷新 `wiki/index.md`，在 `wiki/log.md` **追加**：

```markdown
## [YYYY-MM-DD] ingest | 标题
```

已 ingest 的权威 CSV 只增不改。

### Query

先读 `wiki/index.md`，回答必须带相对链接引用。稳定的好答案可落成 `comparison` 或 `playbook`。

### Lint

运行 `PYTHONPATH=. python tools/wiki_lint.py`。检查：孤儿页、断链、无 `sources` 的数字、playbook 不可执行。

## 6. 已有题包

- `2025_A_evacuation/` — HiMCM 2025 Problem A（搜救扫荡）。不要覆盖其 `src/` 与权威 CSV。
- `2024_A_olympic_sde/` — 目录占位。无权威结果前，Wiki 不得编造 2024 数字。

# HiMCM Studies — Agent Schema (LLM Wiki)

本文件是仓库的 **Schema**：告诉 Agent 如何维护 `wiki/`、如何对待题包（Raw）、以及竞赛合规边界。  
理念来源：[`llm-wiki.md`](llm-wiki.md)（Karpathy LLM Wiki）。  
题包契约：[`_template/README.md`](_template/README.md)。  
十年总览：[`STUDIES.md`](STUDIES.md)。

仓名 **GoHiMCM**；根目录是 Studies 枢纽，**每年一道** `YYYY_P_slug/` 题包。

---

## 1. 三层架构

| 层 | 路径 | 谁写 | 规则 |
| --- | --- | --- | --- |
| **Raw（题包）** | `2025_A_evacuation/`、`2024_A_olympic_sde/`、未来的 `YYYY_P_slug/` | 学生/建模者 | **只增不改已 ingest 的证据**；不擅自改题包内 `results/*.csv` |
| **Wiki** | `wiki/` | **仅 LLM / Agent** | 人只读；心得放题包 `paper/notes/` 再 Ingest |
| **Schema** | 本文件 + `.cursorrules` | 师生共进化 | 新约定写入本文件同一提交 |

根目录 **不应**再放业务 `src/`；代码只在各年题包内。

---

## 2. Wiki 页面类型（只允许这些）

| type | 目录 | 一页写什么 |
| --- | --- | --- |
| `problem` | `wiki/problems/` | 年份、A/B、题型、要求、交付物路径 |
| `method` | `wiki/methods/` | 公式、适用条件、代码入口、失败模式 |
| `concept` | `wiki/concepts/` | 易错语义、符号、竞赛规范 |
| `playbook` | `wiki/playbooks/` | 可执行步骤 + 验收标准 |
| `comparison` | `wiki/comparisons/` | 跨年/跨方法对照 |
| `source` | `wiki/sources/` | 对某一 raw 产物的摘要（须可追溯） |

特殊页：`wiki/index.md`、`wiki/log.md`、`wiki/synthesis.md`。

### 页面格式

```markdown
# 标题

一段目录用摘要。

```yaml
type: method
year: 2025 | cross
status: draft | ingested | contest-ready
sources:
  - 2025_A_evacuation/results/experiment_summary.csv
```

## 正文
...

## Related
- [Other Page](../methods/other.md)
```

规则：文件名 kebab-case；交叉引用相对路径 + `.md`；禁止自链；**禁止发明数字**（无 raw 则标 `unverified`）。

---

## 3. 三操作

### Ingest

触发：「ingest 某题包 / 某次实验 / 某篇草稿」。

1. 只读题包 raw，不改权威 CSV。  
2. 写/更新 `source` 及相关 problem/method/concept/playbook（可碰 5–15 页）。  
3. 刷新 `wiki/index.md`；相关页补 `## Related`。  
4. 追加 `wiki/log.md`：`## [YYYY-MM-DD] ingest | 标题`  
5. `sources:` 路径必须带题包前缀，例如 `2025_A_evacuation/results/...`。

### Query

1. 先读 `wiki/index.md`（必要时 `synthesis.md`）。  
2. 打开 3–8 页，带引用回答；核对数字时回题包 raw。  
3. 好答案可 save 为 comparison/playbook + log。

**赛时协议：** 读 Wiki → 复制 `_template/` 建**新**题目录 → 在新目录建模 → AI 写入**当年** `prompts/cursor_log.md` → 不改旧年论文交差。

### Lint

查孤儿页、断链、空页、无 sources 定量结论、problem↔method、playbook 可执行性。  
追加：`## [YYYY-MM-DD] lint | wiki lint pass`。

---

## 4. 竞赛纪律

1. 严禁伪造数据；图只从题包 `results/` 读。  
2. Wiki 不是第二份数据表。  
3. `legacy/` = **本题**退役代码，不是上一年整题。  
4. 学生写 raw；Agent 写 wiki。  
5. Python 3.10+，Type Hints + Docstring，完整可运行块（见 `.cursorrules`）。

---

## 5. 题包模块边界

见 [`_template/README.md`](_template/README.md)。  
`src/` 子结构按题型自定（2025：environment/planner/rl；2024：indicators/mcda/sensitivity；2023：physics/biology/simulation/theory/decision；2021：generation/load/battery/optimization；2020：factor_analysis/llm_factor/entropy_weight/clustering/classification/web_app）。

---

## 6. Co-evolution

新约定与本文件同一提交更新。课程流程：[`docs/course/weekly_loop.md`](docs/course/weekly_loop.md)。  
2023 题包规则：[`2023_A_dandelion_prisms/.cursorrules`](2023_A_dandelion_prisms/.cursorrules) 与 [`.cursor/rules/2023-a-dandelion.mdc`](.cursor/rules/2023-a-dandelion.mdc)（只匹配该题包 glob，不覆盖根目录 2025 `.cursorrules`）。  
2020 题包规则：[`2020_A_summer_job_factor/.cursorrules`](2020_A_summer_job_factor/.cursorrules) 与 [`.cursor/rules/2020-a-summer-job.mdc`](.cursor/rules/2020-a-summer-job.mdc)（glob 仅 `2020_A_summer_job_factor/**`）。

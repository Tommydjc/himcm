# llm-wiki.md — Karpathy 风格 LLM Wiki（本仓库实现）

思想：模型不靠聊天记忆，而靠一套**人只读、Agent 可改**的 Markdown 图。查询先读目录，写入走 ingest，质量靠 lint。

## 目录

```
wiki/
  index.md          # 总目录，Query 的第一跳
  log.md            # 只追加 ingest / lint 记录
  synthesis.md      # 跨年综合，禁止无引用数字
  problems/
  methods/
  concepts/
  playbooks/
  comparisons/
  sources/
```

## 页合同

只允许类型：`problem` | `method` | `concept` | `playbook` | `comparison` | `source`。

每页顺序固定：

1. `# 标题`
2. 一段摘要
3. yaml 块（`type` `year` `status` `sources`）
4. 正文；链接写成 `[文字](../methods/foo.md)` 这种带 `.md` 的相对路径

`sources` 必须带题包前缀：

```yaml
sources:
  - 2025_A_evacuation/results/experiment_summary.csv
```

## 定量纪律

- 数字必须能回溯到题包 `results/` 或 `data/raw/`。
- 做不到就写 **unverified**，不要编。
- 已 ingest 的 CSV：只允许追加新文件或新行协议，不改旧行。

## 三种操作

| 操作 | 做什么 |
|---|---|
| Ingest | 读 raw + CSV → 改 5–15 页 → 刷新 `index.md` → 追加 `log.md`：`## [YYYY-MM-DD] ingest \| 标题` |
| Query | 先 `index.md`，回答带链接；好答案可另存 comparison / playbook |
| Lint | `python tools/wiki_lint.py`：孤儿页、断链、无 sources 的数字、playbook 无命令块 |

课结束如何 ingest：见 [`docs/course/weekly_loop.md`](docs/course/weekly_loop.md)。

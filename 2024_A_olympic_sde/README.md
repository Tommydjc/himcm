# 题包模板 `_template`

复制为本年目录后，把本文件改成该题的说明。根目录开题命令见仓库 `README.md`：

```powershell
Copy-Item -Recurse _template YYYY_P_slug
```

## 目录契约

| 路径 | 用途 |
|---|---|
| `data/raw/` | 只读赛题与公开表 |
| `data/processed/` | 清洗结果 |
| `src/` | 按题型自定（见下），不要三套架构同时空转 |
| `experiments/` | 怎么跑 |
| `results/` | 权威 CSV（论文数字唯一来源） |
| `logs/` | 运行日志 |
| `tests/` | 护公式 |
| `paper/` `paper_figures/` | 文稿与由 CSV 生成的图 |
| `prompts/` | AI 披露日志 |
| `legacy/` | 本题退役代码，不是上一年整题 |

## `src/` 题型（择一）

- 仿真/搜救：`environment` / `traditional_planner` / `rl`
- 评价/MCDM：`indicators` / `mcda` /（可选）`ml` / `sensitivity`
- 机理/优化：`model` / `solver` / `sensitivity`

Python 3.10+，Type Hints，禁止省略号占位，禁止伪造 CSV。

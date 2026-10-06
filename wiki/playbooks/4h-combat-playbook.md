# 4-hour contest combat playbook

240 分钟交卷：先锁可复现 CSV，再写 25 页。超时砍枝，不砍证据。

```yaml
type: playbook
year: cross
status: contest-ready
sources:
  - 2025_A_evacuation/results/experiment_summary.csv
  - 2024_A_olympic_sde/paper/himcm_paper.tex
  - 2020_A_summer_job_factor/paper/compile.sh
  - AGENTS.md
```

## 时间盒

| 窗 | 做什么 | 砍枝 |
| --- | --- | --- |
| 0–30 min | 读题、建 `YYYY_P_slug/`、写假设表 | 不写 RL / Crew 长训 |
| 30–120 min | 基线模型跑通，落 `results/*.csv` | 不调超参到「好看」 |
| 120–180 min | 敏感性 / 一图一表 | 砍第三套方法 |
| 180–230 min | 论文 + 摘要页 | 砍装饰图 |
| 230–240 min | 编译 PDF、AI 披露、打包 | 不改已 ingest CSV |

## 验收

- 文中数字能 grep 到题包 CSV。
- `prompts/cursor_log.md` 有本场条目。
- 规划器 / AHP / EFA 基线永远先于 GAT–PPO / LLM。

## Related

- [规格驱动编码](spec-driven-cursor-coding.md)
- [COMAP 披露](../concepts/comap-ai-disclosure.md)
- [2025 疏散](../problems/2025-a-evacuation.md)

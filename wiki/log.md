# Wiki 操作日志

Ingest / Query / Lint 时间线。新条目追加在文件顶部附近的日期节。

```yaml
type: log
year: cross
status: ingested
sources:
  - AGENTS.md
```

## [2026-09-26] rule | 2023 A .cursorrules from Simpson–McCue + PRISM

写入 `2023_A_dandelion_prisms/.cursorrules` 与 `.cursor/rules/2023-a-*.mdc`。锚点 arXiv:2403.01667（KPP 波前协议）与 arXiv:2601.11747（从本题 CSV 抽知识）。未改权威 CSV。

## [2026-09-26] ingest | 2023 A dandelion prisms scaffold

只读 `2023_A_dandelion_prisms/` Raw，写入 problem / methods / concept / playbook / sources，刷新 `wiki/index.md` 与 `wiki/synthesis.md`。未改 `results/*.csv`。status=`draft`（非 contest-ready）。STUDIES 增 2023 行。

核对保留：温带 12 月 cover 0.2097、锋面 46.5 m；干旱 0.0273 / 热带 0.0064；KPP 相对误差 0.0521；影响因子 PUMON 0.9333 / REJAP 0.6935 / TAROF 0.0000；双周 \(\eta=1\) 覆盖 0.0077 成本 39。

## [2026-09-19] ingest | 2024 A study close-out

只读 `2024_A_olympic_sde/` Raw，写入 problem / methods / concepts / playbooks / sources / comparison，刷新 `wiki/index.md` 与 `wiki/synthesis.md`。未改 `results/*.csv`、`src/`、`data/processed/`、判断矩阵。status=`ingested`（非 contest-ready）。STUDIES 2024 行改为 ingested。`prompts/cursor_log.md` 增一行「写入 Wiki、不改模型」。

核对保留：Brisbane SAW Flag 0.5360 / Cricket 0.4842 / Squash 0.3960；youth=0 时 Flag Δrank=−2；LOOCV Acc=0.3333 AUC=0.5000 Brier=0.2475；余弦 0.7193（由 `ahp_vs_ml_weights.csv` 两列算出）。

## Related

- [目录](index.md)
- [2024 问题](problems/2024-a-olympic-sde.md)

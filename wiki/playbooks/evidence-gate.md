# 证据门闩（论文数字 ⊆ CSV）

交卷前抽检：正文中的高信号数字须能在当年 `results/*.csv`（或明确标注的 raw 表）中找到。脚本：`tools/evidence_check.py`。

```yaml
type: playbook
year: cross
status: contest-ready
sources:
  - tools/evidence_check.py
  - AGENTS.md
  - wiki/playbooks/results-to-figures.md
  - 2025_A_evacuation/results/experiment_summary.csv
```

## 步骤

1. 锁定当年题包路径 `YYYY_P_slug/`。
2. 确认权威表在 `results/*.csv`（不要用手改论文里的「好看数」倒推 CSV）。
3. 先跑顾问模式（默认 exit 0，仍打印 missing）：

```text
python tools/evidence_check.py --slug 2025_A_evacuation
```

4. 交卷门闩用严格模式：

```text
python tools/evidence_check.py --slug 2025_A_evacuation --strict
```

5. 处理 `missing`：改论文、补跑写入 CSV，或把**非结果常数**（版心、假设参数）写入 `paper/evidence_allowlist.txt`（一行一个 token，`#` 注释）。
6. 图：按 [CSV 到图](results-to-figures.md) 只读 results 出图。

## 验收

- `--strict` 下 exit code 0。
- 主结论表与 CSV 行一致；allowlist 不含决策数字。

## Related

- [Contest Hub](contest-hub.md)
- [4h 交卷](4h-combat-playbook.md)
- [CSV 到图](results-to-figures.md)
- [跨年失败模式](../concepts/cross-year-failure-modes.md)

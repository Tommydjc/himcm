# 实验协议

输出一律写入 `../results/`。禁止手填数字。

## 基线（Req 1）

- 气候：`temperate` / `arid` / `tropical`
- 干预：无
- 时长：52 周
- 验收：`monthly_metrics.csv` 含 1–12 月；快照周 4/8/13/26/52

## 干旱扰动

- 仅温带，`smi_scale=0.5`
- 对照列 `scenario=drought_smi_x0.5`
- 验收：同月末 `cover_frac` 不高于基线（允许数值噪声 \(10^{-9}\)）

## 割草帕累托

- 间隔周 \(\tau\in\{0,1,2,4,6,8,13,26\}\)，\(\tau=0\) 为不割
- 强度 \(\eta\in\{0.4,0.7,1.0\}\)
- 目标：12 月 `cover_frac` 与 \(C=n(1+\eta/2)\)
- 验收：`bioeconomic_pareto.csv` 的 `on_front` 与 `decision.nondominated` 一致

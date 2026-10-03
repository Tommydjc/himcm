# CCD 三重胁迫分岔

在农药、瓦螨与倒春寒同时作用下，扫描外勤死亡率 \(\mu_F\)，用年末储蜜绝对阈值划分 Safe / Compensatory / Collapse。

```yaml
type: method
year: 2022
status: ingested
sources:
  - 2022_A_honeybee_dynamics/src/stress_test/ccd_bifurcation.py
  - 2022_A_honeybee_dynamics/results/ccd_bifurcation_sweep.csv
  - 2022_A_honeybee_dynamics/results/ccd_tipping_point.csv
```

## 分区（克）

Safe \(N\ge 5000\)；Compensatory \(3000\le N<5000\)；Collapse \(N<3000\)。

## 实测拐点（`ccd_tipping_point.csv`）

- 软拐点（离开 Safe）：\(\mu_F^{\ast,\mathrm{soft}}=0.11\)
- 硬拐点（越过 Collapse 线）：\(\mu_F^{\ast}=0.305\,\mathrm{day}^{-1}\)

扫描：\(\mu_F\in[0.08,0.40]\) step \(0.005\)（65 点）。勿把未校准示例值 0.285 写入正文。

## 失败模式

- 无胁迫扫 \(\mu_F\) 却声称「田间 CCD」。
- 用相对跌幅阈值替代绝对 5000/3000 线，导致 Compensatory 带为空却硬画三区。

## Related

- [拐点 source](../sources/ccd-tipping-point.md)
- [2022 问题](../problems/2022-a-honeybee-dynamics.md)

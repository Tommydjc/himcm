# 20 英亩扁桃园蜂箱密度 OR

竞争稀释日粮 + Michaelis–Menten 结实 + \$200/箱租金，扫描蜂箱数 \(K\) 得农场净收益与蜂群健康。

```yaml
type: method
year: 2022
status: ingested
sources:
  - 2022_A_honeybee_dynamics/src/pollination/orchard_20acres.py
  - 2022_A_honeybee_dynamics/results/orchard_pollination_optimization.csv
  - 2022_A_honeybee_dynamics/results/orchard_pollination_recommendation.csv
```

## 公式

\[
\mathrm{DailyFoodPerHive}(K)=\min(C_{\max},N_{\mathrm{orchard}}/K),\quad
\mathrm{YieldRatio}=\frac{V}{V+V_{50}},\quad
\Pi=R_{\mathrm{full}}\cdot\mathrm{YieldRatio}-200K.
\]

\(C_{\max}=F_{\mathrm{pkg}}\eta_N=3500\,\mathrm{g\,day}^{-1}\)（表内标定）。黄金带 \(K\in[20,40]\)（1–2 箱/英亩）。

## 实测推荐（`orchard_pollination_recommendation.csv`）

- 带内最优 \(K^\star=40\)，YieldRatio \(0.848\)，净收益 \$51{,}394，健康指数 \(0.975\)
- 全局最优 \(K=42\)（恰在稀释起点）

## 失败模式

- 手写 94.2% 结实率却无对应 CSV 行。
- 把 `pollination_hives.csv` 的 industry 经验箱数直接当本 OR 最优解。

## Related

- [果园扫描 source](../sources/orchard-pollination-optimization.md)
- [2022 问题](../problems/2022-a-honeybee-dynamics.md)

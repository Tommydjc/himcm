# 2018 B 双轨：Teaching vs Legacy

同一题包两条证据宇宙；**禁止混表**。这是 2018 最易翻车的语义坑。

```yaml
type: concept
year: 2018
status: ingested
sources:
  - 2018_B_smart_house_mpc/experiments/protocols.md
  - 2018_B_smart_house_mpc/results/bang_vs_mpc_summary.json
  - 2018_B_smart_house_mpc/results/competitor_benchmarks.csv
```

## Teaching（论文主表）

- Specs：`building_specs_teaching.json`，TOU v2（0.12/0.22/0.48）  
- 对照：Bang-Bang vs 预测矩阵 MPC，24h 合成盛夏日  
- 锚点：成本 −4.2%，峰电 −15.8%，启停 42→3；峰占比实测 43.2%

## Legacy（附录）

- Specs：旧 `building_specs.json`，TOU 0.08/0.14/0.32  
- 对照：五控制器 36h 多物理场周切片  
- 锚点：相对 Nest-like 能耗 −64.6%（`competitor_benchmarks.csv`）  
- Streamlit Tab「Legacy Week」只服务本轨

## 禁令

1. 主文写 Nest−64.6% 却用 Teaching 天气叙事。  
2. 把 Legacy 峰电价 \$0.32 与 Teaching \$0.48 混进同一段。  
3. 用 Harness Teaching 雷达图去「证明」Legacy 竞品表。

## Related

- [2018 问题](../problems/2018-b-smart-house.md)
- [合成数据陷阱](../comparisons/synthetic-data-trap.md)
- [两套宇宙](two-universes.md)

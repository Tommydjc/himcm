# 黑盒气候评测 Harness

对 Bang-Bang / MPC 做完全黑盒、多场景压力体检；输出标准化 M1–M4 与雷达图。

```yaml
type: method
year: 2018
status: ingested
sources:
  - 2018_B_smart_house_mpc/src/eval_harness/benchmark_harness.py
  - 2018_B_smart_house_mpc/results/harness_benchmark_results.csv
  - 2018_B_smart_house_mpc/results/harness_benchmark_summary.json
  - 2018_B_smart_house_mpc/paper_figures/fig_harness_radar_benchmark.png
  - 2018_B_smart_house_mpc/paper_figures/fig_temperature_tracking_comparison.png
```

## 场景库

| ID | 扰动 |
| --- | --- |
| A | 标准盛夏 23–38°C |
| B | 连续 3 日热浪 ~42°C |
| C | 2 h 内室外骤降 15°C |
| D | 传感器 σ=1.5°C |
| E | 17–19h 漏风 ×5 |

## 指标

- \(M_1\)：总电费 \$  
- \(M_2\)：\(\int(T_{\mathrm{in}}-23)^+\,dt\)（°C·h）  
- \(M_3\)：启停 + 抖动惩罚  
- \(M_4\)：相对情景 A 的成本比方差（鲁棒）

## 均值锚点（locked summary）

Bang：\(M_1=2.427\)，\(M_2=7.380\)，\(M_3=77.6\)，\(M_4=0.865\)。  
MPC：\(M_1=2.340\)，\(M_2=8.574\)，\(M_3=6.4\)，\(M_4=0.874\)。

注意：情景 D 下 Bang 的 \(M_2\) 可因过冷而偏低，但 \(M_3\) 飙到 133；MPC 启停 16。叙事应分维度，勿只报单一 KPI。

## 入口

```text
PYTHONPATH=src python -m eval_harness.benchmark_harness
```

## Related

- [2018 问题](../problems/2018-b-smart-house.md)
- [MPC 方法](2r2c-prediction-matrix-mpc.md)
- [source CSV](../sources/2018-harness-benchmark.md)

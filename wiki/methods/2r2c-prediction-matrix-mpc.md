# 2R2C + 预测矩阵滚动 MPC

教学轨核心：连续 2R2C → ZOH 离散 → \(\mathbf{X}=\mathcal{A}x_0+\mathcal{B}\mathbf{Q}+\mathcal{E}\mathbf{D}\) → 滚动 L-BFGS-B；植物仍用 1 min RK4。

```yaml
type: method
year: 2018
status: ingested
sources:
  - 2018_B_smart_house_mpc/src/thermal_physics/building_rc.py
  - 2018_B_smart_house_mpc/src/control_mpc/state_space_2r2c.py
  - 2018_B_smart_house_mpc/src/control_mpc/prediction_matrices.py
  - 2018_B_smart_house_mpc/src/control_mpc/mpc_controller.py
  - 2018_B_smart_house_mpc/results/bang_vs_mpc_summary.json
  - 2018_B_smart_house_mpc/paper_figures/fig_bang_vs_mpc_day.png
```

## 适用条件

- 单区热核、分钟级夏季日、三档 TOU（峰 \$0.48）。
- 决策 \(u\in[0,1]\)，\(q=-u\,Q_{\mathrm{cool}}\)。
- 需要可审计的线性预测（非黑盒神经网络）。

## 公式要点

连续：\(C_{\mathrm{air}}\dot T_{\mathrm{in}}\)、\(C_{\mathrm{wall}}\dot T_{\mathrm{wall}}\)（见 teaching specs）。  
COP：\(3.8-0.04(T_{\mathrm{out}}-25)\)。  
离散：Van Loan ZOH，\(T_s=900\,\mathrm{s}\)。  
MPC：\(H=24\)；峰段抑 \(u^2\)；预冷目标可经 `MPCComfortConfig` 注入。

## 代码入口

```text
PYTHONPATH=src python -m thermal_physics.building_rc
PYTHONPATH=src python -m control_mpc.prediction_matrices
PYTHONPATH=src python -m control_mpc.mpc_controller
```

## 锁定对照（同日）

| | Bang-Bang | MPC |
| --- | ---: | ---: |
| 成本 \$ | 1.682 | 1.611（−4.2%） |
| 峰电 kWh | 1.514 | 1.274（−15.8%） |
| 启停 | 42 | 3 |
| 预峰 / 峰 \(\bar u\) | — | 0.40 / 0.17 |

## 失败模式

- 把 Legacy Nest−64.6% 写进 Teaching 主表。
- 峰占比造假 ≥60%（实测 Bang 为 43.2%）。
- 用 Streamlit Plotly 顶替 300 DPI `paper_figures/`。
- 预测模型与 RK4 植物偏差不声明。

## Related

- [2018 问题](../problems/2018-b-smart-house.md)
- [双轨](../concepts/2018-b-dual-track.md)
- [Harness](black-box-climate-harness.md)
- [MAS](cognitive-thermostat-mas.md)

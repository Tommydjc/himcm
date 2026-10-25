# BDI 旅客 × SimPy 机场充电 DES

微观：Belief–Desire–Intention 旅客在 SOC 与登机松弛下选择快充/慢充/充电宝/弃排。  
宏观物理：SimPy 三资源池 + NHPP 到达；认知事件与服务事件解耦。

```yaml
type: method
year: 2019
status: ingested
sources:
  - 2019_A_charge_des_agent/src/bdi_agents/passenger_bdi.py
  - 2019_A_charge_des_agent/src/des_env/airport_sim.py
  - 2019_A_charge_des_agent/results/stage2_baseline_metrics.csv
  - 2019_A_charge_des_agent/paper_figures/fig_bdi_intention_flow_sankey.png
  - 2019_A_charge_des_agent/paper_figures/stage2_queue_baseline.png
```

## 适用条件

- 有限服务台、可观测排队、明确弃排（reneging）规则。
- 需同时报告等待、弃排、利用率与系统负荷（勿只报 queue_length）。

## 要点

- 资源：65 W fast kiosk、15 W locker、power-bank bank。  
- 到达：16 h NHPP，三波峰；情景 `normal` / `rush` / `storm`。  
- BDI：约 25% 电量焦虑先验；`T_board ~ U(45,120)`；误机硬约束。  
- 状态机：IDLE→SEEK→QUEUE→RENEGING→RENT/ABORT。  
- KPI：`mean_wait_time`、`reneging_rate`、`max_system_load`、利用率。

## 入口

```text
cd 2019_A_charge_des_agent
PYTHONPATH=src python -m scripts.run_stage2_simulation
PYTHONPATH=src python -m pytest tests/test_airport_sim.py tests/test_passenger_bdi.py -q
```

## 失败模式

- 过配置时 `queue_length≡0` 却仍画「队长图」——应改用 `system_load` / `in_service`。  
- 负排队人数或时钟非单调（SimPy 纪律）。  
- 把 planning 子任务高弃排点（如 003 的 0.637）写成推荐膝点而不加注。

## Related

- [2019 问题](../problems/2019-a-airport-charging.md)
- [Pareto 选型](pareto-kiosk-sizing.md)
- [反思智能体](langgraph-des-reflection.md)

# 2022 A 复现命令

在题包根目录跑通弹性 / CCD / 果园与测试。

```yaml
type: playbook
year: 2022
status: ingested
sources:
  - 2022_A_honeybee_dynamics/pytest.ini
  - 2022_A_honeybee_dynamics/src/sensitivity/autograd_sens.py
  - 2022_A_honeybee_dynamics/src/stress_test/ccd_bifurcation.py
  - 2022_A_honeybee_dynamics/src/pollination/orchard_20acres.py
```

## 步骤

```bash
cd 2022_A_honeybee_dynamics
pytest tests/ -q
PYTHONPATH=. python -m src.sensitivity.autograd_sens
PYTHONPATH=. python -m src.stress_test.ccd_bifurcation
PYTHONPATH=. python -m src.pollination.orchard_20acres
```

论文 PDF（若已装 tectonic）：

```bash
cd paper
export PATH="$HOME/.local/bin:$PATH"
./compile.sh
```

## 验收

- `pytest`：8 passed（autograd / CCD / orchard）。
- `results/autograd_elasticity_ranking.csv`、`ccd_tipping_point.csv`、`orchard_pollination_recommendation.csv` 存在且非空。
- 黄金带 \(K^\star=40\)、硬拐点 \(0.305\) 能在 CSV 逐字对上。
- 不手改 CSV。

## Related

- [研究闭环](../playbooks/2022-a-study-loop.md)
- [2022 问题](../problems/2022-a-honeybee-dynamics.md)

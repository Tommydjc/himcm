# 2023 A — Dandelion Prisms

HiMCM 2023 Problem A（*Dandelions: Friend? Foe? Both? Neither?*）题包。
一公顷 \(100\times 100\,\mathrm{m}\) 草坪上，从地块西缘外侧一株绒球期蒲公英出发，做三气候空间扩散，并给出入侵影响因子、割草帕累托与 HOA 信。

权威数字只认 `results/*.csv`。气候 CSV 是**情景强迫**，不是台站观测。

## 官方题面对照

| 要求 | 本包入口 |
| --- | --- |
| Req 1：1/2/3/6/12 月扩散；温带 / 干旱 / 热带 | `src/physics/` + `src/biology/` + `src/simulation/` |
| 官方 Req 2：入侵种 impact factor（蒲公英 + 两种对照） | `src/decision/impact.py` |
| 课程扩展：割草双目标帕累托 + HOA 信 | `src/decision/pareto.py`，`paper/hoa_letter.tex` |

## 一键复现

在题包根目录、已激活仓库 `venv` 后：

```bash
PYTHONPATH=. python scripts/download_real_climate.py
PYTHONPATH=. python -m src.scripts.generate_forcing
PYTHONPATH=. python -m src.scripts.run_all
PYTHONPATH=. python -m src.scripts.plot_figures
python -m unittest discover -s tests -v
```

`download_real_climate.py` 写 Open-Meteo 2022 日表（`*_weather.csv` / `*_soil.csv`），不覆盖情景 `*_hourly.csv` / `*_smi.csv`。当前仿真默认仍读情景文件。

## 模块边界

| 目录 | 只做什么 |
| --- | --- |
| `src/physics/` | WALD 核与 2D 羽流，不谈生活史 |
| `src/biology/` | Lefkovitch 与自疏，不谈空间卷积 |
| `src/simulation/` | 格网卷积引擎，不写控制目标 |
| `src/theory/` | Fisher–KPP 解析/数值验算 |
| `src/decision/` | 影响因子与割草帕累托 |
| `src/scripts/` | 出表出图流水线 |

Python 3.10+。依赖：NumPy、SciPy、pandas、Matplotlib。  
Agent 规则：本目录 `.cursorrules`（Simpson–McCue 双重印证 + PRISM 检索再改）。

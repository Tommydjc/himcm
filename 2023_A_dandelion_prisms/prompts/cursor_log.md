# Cursor AI Interaction Log（COMAP AI Disclosure）

课结束 ingest：把本包 `results/*.csv` 交给 Agent，按仓库 `AGENTS.md` 更新 Wiki，不要在本文件里手填实验表。

| 时间戳 | 任务目标 | 涉及文件 | 人工确认要点 |
|---|---|---|---|
| 2026-09-26 | 按六层边界搭建 2023 A 题包：WALD 羽流、Lefkovitch、1 ha 卷积、Fisher–KPP、影响因子、割草帕累托 | `src/physics/` `src/biology/` `src/simulation/` `src/theory/` `src/decision/` `src/scripts/` | 气候 CSV 是情景不是台站；未手改 `results/*.csv`；无 `np.random` 填表 |
| 2026-09-26 | 核归一化、点源质量账、自疏上限、KPP 波速测试 | `tests/test_*.py` | 11 项 unittest 通过；KPP 相对误差以 `kpp_check.csv` 为准 |
| 2026-09-26 | 全量出表出图 | `src/scripts/run_all.py` `src/scripts/plot_figures.py` | 数字只从模型写出；热带离岸风加西缘就近滞留，避免舍入鬼种群 |
| 2026-09-26 | 论文骨架与 HOA 信注入 CSV | `paper/himcm_paper.tex` `paper/hoa_letter.tex` | 表内数字来自 `monthly_metrics.csv` / `impact_factors.csv` / `bioeconomic_pareto.csv` / `kpp_check.csv` |
| 2026-09-26 | 写入 Wiki（draft ingest） | `wiki/problems/2023-a-dandelion-prisms.md` 及 methods/playbooks/sources | 不改模型、不改权威 CSV；status=`draft`（非 contest-ready） |
| 2026-09-26 | 按 Simpson–McCue 与 PRISM 两篇论文固化 2023 A `.cursorrules` | `2023_A_dandelion_prisms/.cursorrules`；`.cursor/rules/2023-a-*.mdc` | 未改 `results/*.csv`；根目录 2025 `.cursorrules` 未覆盖；glob 仅 `2023_A_dandelion_prisms/**` |
| 2026-09-26 | Open-Meteo Archive 下载 2022 三地逐日风温与 0–7 cm 土壤水 | `scripts/download_real_climate.py`；`data/raw/weather/*_weather.csv`；`data/raw/soil/*_soil.csv`；`data/raw/open_meteo_2022_manifest.json` | 无需注册；未手填观测；未覆盖情景 `*_hourly.csv`/`*_smi.csv`；未改 `results/*.csv`；365 行无缺测 |
| 2026-09-26 | 实现 SVR 混合扩散核：近场二元高斯 + 热上升 WALD，再乘 \(\Phi(T,\\theta)\) | `src/physics/dispersal_kernel.py`；`tests/test_dispersal_kernel.py` | 列名只认 Open-Meteo `wind_speed/wind_direction/temperature/soil_moisture`；`smi` 拒绝当作 VWC；衰减前离散质量和为 \(N\)；未改 `results/*.csv`；未改引擎默认 WALD 契约 |
| 2026-09-26 | 引擎周步改接混合核卷积与点源投放 | `src/simulation/engine.py`；`src/scripts/forcing_weeks.py`；`tests/test_engine_hybrid.py` | `K_disp` 替换 `build_plume_kernel`；`Phi` 仅当 `soil_moisture` 有 VWC；`smi` 仍只进 Lefkovitch；未手改 `results/*.csv` |
| 2026-09-26 | 日步物候 + 1 ha 吸收边界卷积引擎 | `src/biology/phenology.py`；`src/simulation/grid_engine.py`；`tests/test_phenology.py`；`tests/test_grid_engine.py` | (50,50) 一株成株；K_cell=15；25 日龄幼苗；`convolve2d(mode=same)`；365 步 <15s；与周步 `engine.py` 解耦；未改 `results/*.csv` |
| 2026-09-26 | 对齐 Stage-2 测试到现有函数式 API | `tests/test_stage2.py` | 去掉不存在的 `DandelionLifecycle`/`DandelionGridEngine` 与 pytest；改走 `PhenologyParams` + `run_hectare_days`；未改 `results/*.csv` |
| 2026-09-26 | 对齐 Stage-2 90 日演示脚本到日步引擎 | `tests/demo_stage2.py` | 去掉 `DandelionGridEngine` 与 `np.random`；`DailyForcing` 确定性春季风温；`on_day` 打印截面；未改 `results/*.csv` |
| 2026-09-26 | 日步 5 节点截面 + Fisher–KPP \(c^*=2\sqrt{rD}\) 对照出图 | `src/theory/fisher_kpp.py`；`src/scripts/run_monthly_analysis.py`；`tests/test_fisher_kpp.py` | \(D=M_2/(4\tau)=\frac12\sigma^2_{\mathrm{wind}}\)（\(\tau=1\) 日）；未覆盖周步 `kpp_check.csv`/`monthly_metrics.csv`；新表 `monthly_population_metrics.csv`；图从 CSV/NPY 读 |
| 2026-09-26 | PRISMS 四策略日步生物经济权衡 | `src/decision/tradeoff_model.py`；`src/simulation/grid_engine.py`；`tests/test_tradeoff_model.py` | \(B_{eco}=\alpha F_{spring}+\beta\bar B\)，\(C_{turf}=\gamma\mathrm{cover}^{1.3}\)；S4 花期+11 日单次修剪；写入 `bioeconomic_tradeoff.csv`；未覆盖周步 `bioeconomic_pareto.csv` |
| 2026-09-26 | 四策略空间/时序/损益对照出图 | `src/scripts/plot_tradeoff_effects.py`；`src/decision/tradeoff_model.py` | 截面 90/120/180/365；轨迹 CSV；图从 NPY/CSV 读；未覆盖周步 `bioeconomic_pareto.csv` |
| 2026-09-26 | 重写论文套件与 HOA 指南信 | `paper/himcm_paper.tex`；`paper/summary_sheet.tex`；`paper/hoa_community_statement.tex` | 数字只抄当前 CSV；拒用 2.8 m/214/38.6%/84.5%/68.2%；周步 cover 0.4511 与日步 100% 分列；未手改 `results/*.csv` |
| 2026-09-26 | PPO 自适应割草（课程扩展） | `src/rl/env_wrapper.py`；`src/rl/ppo_agent.py`；`src/rl/train_rl.py`；`src/rl/compare_policies.py` | 复用 `step_phenology` 非虚构引擎类；10×10 缓冲带加速；早春 Action 2/3 罚 −15；写 `rl_policy_evaluation.csv`；未覆盖周步帕累托表 |
| 2026-09-26 | 论文写入 PPO 第 4 节（真实表值） | `paper/himcm_paper.tex`；`paper/summary_sheet.tex`；`paper/hoa_community_statement.tex` | 甘特为全年 NO_OP；授粉 41.49 对 14 日 30.28；拒用 88.2%/6.8%；0.0064 标明来自周步帕累托 |

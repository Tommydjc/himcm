# Cursor AI Interaction Log（COMAP AI Disclosure）

## 2026-09-13 — Requirement 2 Split-Half 解析推导

- **核心 Prompt**：针对 2025 HiMCM Problem A Requirement 2（Figure 1 基础办公室），在给定几何与速度参数下，推导两名消防员独立对称扫荡（Split-Half）的路径拓扑、理论最短清空时间，以及走廊浓烟降速 40% 且高危房间双人交叉复核时的公式修正。
- **解决的关键问题**：
  1. 将 Figure 1 建成 1D 走廊悬挂图，给出 \(F_1,F_2\) 的责任分区与列优先路径序列。
  2. 建立加性单室服务时间 \(t_{\mathrm{svc}}\) 与 makespan 下界，得到 \(T_{\mathrm{clear}}^{\mathrm{SH}}=145\,\mathrm{s}\)。
  3. 引入 \(\alpha=0.4\) 的走廊降速与交叉复核项，分别给出中室高危与全室高危的闭式修正，并与 2-in-2-out 对照。
- **产出文件**：`results/req2_split_half_clearance.md`

## 2026-09-13 — 环境层 layouts / hazards 实现

- **核心 Prompt**：在 `src/environment/` 实现 `layouts.py`（NetworkX 办公室 / 日托 / 仓库拓扑 + `render_layout`）与 `hazards.py`（`HazardModel`，\(P_{spread}=1-(1-p_0)^k\)，浓烟速度为明火 1.5 倍并衰减速度与视野）。两文件互相解耦，完整 Type Hints。
- **解决的关键问题**：
  1. Figure 1 固化为 11 节点悬挂图（3 走廊节点 + 两侧 Exit + 6 室）。
  2. Req 3 日托三层楼梯连通与儿童 `vulnerable_weight`；仓库货架障碍与多通道出口。
  3. 火/烟动力学只依赖通用 `nx.Graph`，不反向导入布局模块。
- **产出文件**：`src/environment/layouts.py`，`src/environment/hazards.py`，`src/environment/__init__.py`；可视化 `results/figures/{office,daycare,warehouse}_layout.png`

## 2026-09-13 — ScoringPlanner 多智能体基线

- **核心 Prompt**：实现 `src/traditional_planner/scoring_planner.py`：空闲消防员按 Score(r) 选择未清空房间，确定性转移 move→sweep→tag，记录 \(T_{\mathrm{clear}}\) 与各室时间戳；`tests/test_planner.py` 验证 Office 双人无死锁清空。
- **解决的关键问题**：
  1. 将 \(w_{dist}, w_{hazard}, w_{priority}, w_{redundancy}\) 写成可计算评分，并用冗余项避免两人锁定同一房间。
  2. 有限状态机按边长/`v_hall`/`v_room` 推进，挂牌时刻写入 `room_clear_times`。
  3. 不可达或全员空闲且无可行目标时判定死锁，避免无限循环。
- **产出文件**：`src/traditional_planner/scoring_planner.py`，`src/traditional_planner/__init__.py`，`tests/test_planner.py`

## 2026-09-13 — Office 可视化演示脚本

- **核心 Prompt**：编写可独立运行的 `scripts/run_office_demo.py`：按 15 s/步打印双消防员位置，清空即停（上限 100 步），输出 Requirement 2 的总步数/秒/分钟与轨迹链，并生成 300 DPI 论文图 `paper_figures/office_sweep_trajectory.png`。
- **解决的关键问题**：
  1. 规划器增加 `PlannerConfig`、`tick()` 与节点轨迹记录，供演示逐步驱动。
  2. `sys.path` 容错导入 `src.*` / 顶层包。
  3. NetworkX+Matplotlib 叠加拓扑与两条消防员动线。
- **产出文件**：`scripts/run_office_demo.py`；规划器增量见 `src/traditional_planner/scoring_planner.py`

## 2026-09-13 — 禁忌表 + 动量抗震荡

- **核心 Prompt**：在 `scoring_planner.py` 引入长度 3 的房间禁忌表与动量奖励，消除走廊–房间边反向循环；叶子回走廊用渴望准则保留。
- **解决的关键问题**：
  1. \(T_a\) 只禁忌最近 3 间已清房间，不禁走廊割点。
  2. \(M(a,r)\) 奖励同枢纽兄弟房，形成列优先。
  3. 多最短路时禁止无必要的 `last_edge` 反向。
- **产出文件**：`src/traditional_planner/scoring_planner.py`，`tests/test_planner.py`

## 2026-09-13 — 蒙特卡洛扫荡实验脚本

- **核心 Prompt**：编写 `experiments/run_sweep_experiments.py`：Office/Daycare/Warehouse × 人数 × 无烟/动态浓烟，各 30 次随机起火蒙特卡洛，记录 \(t_{clear}\)、脆弱人群成功率、冗余度，写入 `results/experiment_summary.csv`。
- **解决的关键问题**：
  1. 实验层只调用 layouts / HazardModel / ScoringPlanner，不把规划逻辑写进 experiments。
  2. 无烟与 Req 4 烟火扩散两工况对照；起火点排除出口。
  3. 指标有明确代数定义（脆弱权重比、超额进入/|R|）。
- **产出文件**：`experiments/run_sweep_experiments.py`，`results/experiment_summary.csv`

## 2026-09-13 — 论文插图 plot_figures

- **核心 Prompt**：根据 `results/experiment_summary.csv` 编写 `src/utils/plot_figures.py`，生成人员–清空时间（95% CI + 边际拐点）、理想 vs 浓烟成功率箱线、冗余度折线，Times New Roman、300 DPI、Wong 色盲色。
- **解决的关键问题**：只读真实 CSV；拐点定义为 \(\arg\max(T(n)-T(n+1))\)；成功率用脆弱人群救出率（百分比）。
- **产出文件**：`src/utils/plot_figures.py`，`paper_figures/fig1_personnel_tradeoff.png`，`paper_figures/fig2_hazard_impact.png`，`paper_figures/fig3_redundancy_tradeoff.png`

## 2026-09-13 — HiMCM LaTeX 论文框架

- **核心 Prompt**：建立 `paper/himcm_paper.tex`（1 inch geometry，amsmath/booktabs/graphicx/hyperref），五段正文对齐 Requirement 1–5，插入 `paper_figures/` 插图与 `experiment_summary.csv` 的真实统计表。
- **解决的关键问题**：解析 \(145\,\mathrm{s}\) 与仿真 \(135\,\mathrm{s}\) 对照；无烟人员拐点与浓烟 \(\eta\) 置信区间均来自 420 次运行，不手填。
- **产出文件**：`paper/himcm_paper.tex`






# Cursor AI Interaction Log（COMAP AI Disclosure）

课结束 ingest：把本包 `results/*.csv` 交给 Agent，按仓库 `AGENTS.md` 更新 Wiki，不要在本文件里手填实验表。

| 时间戳 | 任务目标 | 涉及文件 | 人工确认要点 |
|---|---|---|---|
| 2026-09-19 | 实现 AHP 特征值求解器（精确主特征对 + CR） | `src/mcda/ahp.py`；`src/mcda/__init__.py`；`tests/test_ahp_cr.py` | 未写 `results/`；无 `np.random` 填表；无 `rl/`；不一致矩阵用 `warnings.warn` 且 `is_consistent=False` |
| 2026-09-19 | 规范化加权综合分 + 轻量 TOPSIS | `src/mcda/score.py`；`src/indicators/schema.py`；`tests/test_score.py` | 效益/成本 min-max+ε；未写 `results/`；无随机数；`schema.py` 仅补 `Indicator` 契约供 score 调用 |
| 2026-09-19 | SDE 特征管道：xlsx + 研究表 → processed | `src/indicators/loader.py`；`src/indicators/schema.py`（``validate_matrix``）；`data/raw/sde_factors_research.csv`；`tests/test_loader.py` | Appearances/Events 只来自官方 xlsx；IOC 五列来自带出处的 research csv，无 `np.random`；写出 `data/processed/sde_matrix.csv` 而非 `results/` |
| 2026-09-19 | 历史回测：锁定 AHP + 综合分 + 条形图 | `src/scripts/run_eval.py`；`data/raw/ahp_judgment_locked.csv`；`results/historical_ranking.csv`；`paper_figures/fig_historical_backtest.png` | 判断矩阵由锁定权重外积生成（CR=0），非随机；得分由 processed 矩阵计算 |
| 2026-09-19 | loader 同时导出布里斯班 2032 三候选 | `src/indicators/loader.py`；`src/indicators/schema.py`（``MATRIX_EXPORT_COLUMNS`` / ``HOST_POPULARITY``）；`data/processed/brisbane_candidates.csv`；`tests/test_loader.py` | 候选因子为团队锁定情景分，非 ``np.random``；列头与 ``sde_matrix.csv`` 一致；AHP 仍只用七准则 ``INDICATORS`` |
| 2026-09-19 | 2032 候选排序 + 六准则 ±20% OAT 与青年权重归零 | `src/scripts/run_sensitivity.py`；`results/brisbane_2032_ranking.csv`；`results/sensitivity_weight_shock.csv`；`paper_figures/fig_sensitivity_shock.png`；`tests/test_sensitivity.py` | 沿用锁定 AHP；扰动后重归一化；无随机数；HOST 不进综合分 |
| 2026-09-19 | 按 Task 1–5 搭 HiMCM 论文骨架并注入 CSV | `paper/himcm_paper.tex` | 表内数字来自三份 `results/*.csv`；未改 CSV；未覆盖根目录 2025 论文 |
| 2026-09-19 | Task 6 致 IOC 执委会政策建议信（约 1.5 页） | `paper/ioc_letter.tex` | 无特征根/矩阵公式；名次与敏感性只引用 `brisbane_2032_ranking.csv` 与 `sensitivity_weight_shock.csv` |
| 2026-09-19 | 小样本入席 Logistic/RF + LOOCV | `src/ml/classifier.py`；`results/ml_prediction_prob.csv`；`tests/test_classifier.py` | 训练仅 6 项回测；无 80/20；无随机填表；RF `random_state=0` 只为复现 |
| 2026-09-19 | AHP 主观权 vs Logistic/RF 特征归因 | `src/ml/importance.py`；`results/ahp_vs_ml_weights.csv`；`paper_figures/fig_ahp_vs_ml_weights.png`；`tests/test_importance.py` | `|β|` 与 permutation 后 L1 归一；余弦与 Spearman；RF 置换 `random_state=0` |
| 2026-09-19 | 无泄漏 ML 数据管道（惯性 + 折内 StandardScaler） | `src/ml/dataset.py`；`src/indicators/loader.py`（``historical_status``）；`tests/test_dataset.py` | 候选惯性强制 0；BSB/SBL 只算 xlsx 惯性、无编造 IOC 六列；LOOCV 禁止全局 scaler.fit |
| 2026-09-19 | 入席分类器改接 dataset + LOOCV 三项指标 + B=500 自助区间 | `src/ml/classifier.py`；`results/ml_prediction_2032.csv`；`tests/test_classifier.py` | Logistic C=0.8/liblinear、RF depth=3/80 树；无 80/20；自助法 seed=42 不是造项目 |
| 2026-09-19 | 六准则 AHP vs 标准化 logistic \|β\|（去截距/惯性） | `src/ml/importance.py`；`results/ahp_vs_ml_weights.csv`；`tests/test_importance.py` | Δw=w_ML-w_AHP；Spearman 双尾 p；AHP 经 `compute_ahp_weights` |
| 2026-09-19 | ML 出图 + 论文片段（实测 LOOCV，不写假高准确率） | `src/ml/plot_and_report.py`；`src/ml/__main__.py`；`paper/sections/sec4_ml_validation.tex`；`paper_figures/fig_ml_2032_probability.png`；`paper_figures/fig_ahp_vs_ml_dumbbell.png` | 数字来自 CSV；互证写部分一致而非强行第一名锁定 |
| 2026-09-19 | 2024 研究写入 Wiki（结课 ingest） | `wiki/problems/2024-a-olympic-sde.md` 及 methods/concepts/playbooks/sources；`wiki/log.md` | 不改模型、不改 `results/*.csv`；status=ingested |

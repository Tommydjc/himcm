# Cursor AI Interaction Log（COMAP AI Disclosure）

| 时间戳 | 任务目标 | 涉及文件 | 人工确认要点 |
|---|---|---|---|
| 2026-10-06 | 写入 2020 A 题包 Cursor 准则 | `.cursorrules`；仓库 `.cursor/rules/2020-a-summer-job.mdc` | 未覆盖根目录 2025 疏散准则；KMO/Bartlett + Varimax；分类只吃 3 因子；熵权；Softmax vs ReLU LOOCV |
| 2026-10-06 | 问卷宽表 + 8 岗位语料落库 | `scripts/prepare_dataset.py`；`data/raw/`；`data/job_descriptions/` | GitHub 须通过 50×15、1–10 校验才采用；否则 SYNTHETIC_FALLBACK + `DATA_SOURCE.txt`；不得把保底矩阵当真实问卷 |
| 2026-10-06 | EDA + KMO + Bartlett 适宜性 | `src/factor_analysis/suitability.py`；`paper_figures/fig_correlation_matrix.png` | 只读 `OriginalData.csv`；KMO≤0.70 告警不改表；Bartlett 用 χ²=-(N-1-(2p+5)/6)ln\|R\| |
| 2026-10-06 | Firecrawl 拉取 HiMCM2020 问卷镜像 | `scripts/fetch_himcm2020_firecrawl.py`；`data/raw/himcm2020_github/`；`student_job_factor_scores.csv` | 密钥只走环境变量；远端是 50×8×7 非 15 Likert；400×7 KMO 仍 <0.70；不把 7 维硬映射成 15 列 |
| 2026-10-06 | 题面职业 → BLS OOH / O*NET | `scripts/fetch_bls_onet_from_problem.py`；`data/raw/bls_ooh_quickfacts.csv`；`onet_work_context_problem_jobs.csv` | PDF 无 URL；按收银/救生/侍应/数据/行政/研究/远程映射；岗位侧相关，不覆盖学生 15 列 |
| 2026-10-06 | PC-EFA + Kaiser 正规化 Varimax + Thompson 得分 | `src/factor_analysis/efa_engine.py`；`results/factor_loadings.csv`；`results/student_factor_scores.csv`；`paper_figures/fig_scree_plot.png` | 只读 `OriginalData.csv`；竞赛抽取 m=3；本题 Kaiser λ>1 给出 m=6、m=3 累计方差 42.6%（未过 65%）；未改问卷表凑准则 |
| 2026-10-06 | LLMFactor：抽出因子命名 + 8 份招聘文本零样本打分 | `src/llm_factor/llm_factor_agent.py`；`results/factor_interpretation.md`；`results/job_factor_matrix.csv` | 命名锚定 `factor_loadings.csv` 每列 \|λ\| 前 4；未把题面「时薪/人力资本/负荷」三元组硬套到错位的 F1–F3；Ollama `qwen2.5:7b`，失败则 mock；岗位坐标与学生 Thompson 得分同轴 |
| 2026-10-06 | 信息熵权：学生 50×3 求 w，再给 8 岗打综合效用 | `src/entropy_weight/entropy_engine.py`；`results/entropy_weights.csv`；`results/job_entropy_utility.csv`；`paper_figures/fig_entropy_weights.png` | w 只由学生 Thompson 得分的 E_j/D_j 决定，无手写常数；F3 仅在效用里取负（成本极性）；ε=1e-4 平移；参考 GitHub entropy.py 但不在 7 维原表上加权 |
| 2026-10-06 | 三维因子空间 K-Means 人格画像 | `src/clustering/persona_kmeans.py`；`results/student_cluster_labels.csv`；`paper_figures/fig_persona_clusters_3d.png` | 竞赛 K=3；肘部曲率 K=2、轮廓最大 K=7、K=3 轮廓 0.26；簇标签按设计原型置换后仍用抽出轴解读；未把 F1 写成时薪 |
| 2026-10-06 | Softmax vs 2 层 ReLU：三维因子 LOOCV 推荐 | `src/classification/dual_recommender.py`；`results/model_comparison_metrics.csv`；`paper_figures/fig_softmax_weight_matrix_heatmap.png` | 只用 50×3 Thompson，不用 15 列；标签 work_choice 0..49 与得分 1..50 按行对齐；LOOCV Top-1 两模型均为 0.20，低于多数类基线 0.22；未调参去追高准确率 |
| 2026-10-06 | Streamlit 兼职导航 SummerJobMatch（Requirement 5） | `src/web_app/app.py`；`src/web_app/service.py` | 滑块用赛题 1–10 用语；推理映射到抽出 Thompson 空间后走全样本 Softmax；时薪从岗位 txt 解析；页面注明 LOOCV Top-1=0.20 |
| 2026-10-06 | 论文套件：主文 + 1 页摘要 + 学生指南 | `paper/himcm_paper.tex`；`paper/summary_sheet.tex`；`paper/high_school_job_guide.tex` | 数字只读 results/；拒绝模板 KMO 0.782、71.4%、Top-3 92%；15 列标明 SYNTHETIC\_FALLBACK |
| 2026-10-06 | Wiki ingest 2020–2025（只写 wiki/ 与根 STUDIES.md） | `wiki/problems/2020-a-summer-job.md` 等；`STUDIES.md` | 未改 `results/*.csv`；KMO 0.445 / Top-1 0.20 写入 Wiki |

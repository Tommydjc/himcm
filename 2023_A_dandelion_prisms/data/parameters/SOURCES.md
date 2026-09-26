# 生物物理参数出处

本目录是**契约**，不是实验结果。数值取自公开文献的数量级中位，供模型闭式使用。
仿真输出必须由代码算出，禁止手改 `results/*.csv`。

| 符号 | 字段 | 取值 | 依据 |
| --- | --- | --- | --- |
| \(H\) | `H_m` | 0.35 m | 花葶高度，Wisconsin Extension / 常见植株尺度 |
| \(v_t\) | `v_t_mps` | 0.32 m/s | *Taraxacum* 冠毛沉降速度约 0.2–0.4 m/s（Sheldon & Burrows 1973；Tackenberg 2003） |
| \(\kappa\) | `kappa_turb` | 0.40 | 近地层 \(\sigma_w/\bar U\) 量级 0.3–0.5 |
| \(n_{\mathrm{cap}}\) | `n_seed_per_capitulum` | 180 | 单头落籽约 150–200 |
| \(K\) | `K_max_per_m2` | 80 株/m² | 重度草坪侵染的数量级上限，不是观测普查 |
| WALD | `wald.py` | — | Katul et al. (2005) Inverse-Gaussian 长距扩散近似 |
| 生活史 | Lefkovitch | — | Caswell, *Matrix Population Models* |
| 自疏 | Yoda 型盖帽 | — | Yoda et al. (1963) 的密度上限形式，不是 \(-3/2\) 生物量回归 |

气候月均值见 `climate_monthly.json`：Köppen 启发的**情景**，不是 NOAA 下载。

混合核 `src/physics/dispersal_kernel.py` 使用分离涡环默认 \(v_t=0.28\,\mathrm{m/s}\)、\(H_0=0.25\,\mathrm{m}\)，与上表 WALD 引擎契约（0.32 / 0.35）并存，互不覆盖。\(\Phi\) 的 \(\theta\) 只认 Open-Meteo `soil_moisture`（m³/m³）。

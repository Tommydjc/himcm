# sde_factors_research.csv 出处说明

`GLOBAL_REACH` 取公开的国际单项联会会员协会数（约数，对应 2024 赛季前后公开页）。  
`GENDER_PARITY` 为 0–100 的百分比代理（巴黎 2024 性别均等纲领下多数奥运项目取 50；板球按男女职业版图差距取 42）。  
`YOUTH_APPEAL` / `INFRA_COST` / `BROADCAST_VAL` 为 1–9 Saaty **研究赋分**，不是 `np.random`，也不是官方 xlsx 列。

xlsx 只提供 `Appearances` 与 `Events_Count`。更换研究表后重新运行：

```bash
cd 2024_A_olympic_sde
PYTHONPATH=. python -m src.indicators.loader
```

# IOC 六块叙述与代码七叶

IOC 政策标题是六块；实现时拆成七个可算叶子：xlsx 提供 Appearances 与 Events_Count，其余五列来自研究表，不是官方 xlsx。

```yaml
type: concept
year: 2024
status: ingested
sources:
  - 2024_A_olympic_sde/src/indicators/schema.py
  - 2024_A_olympic_sde/data/raw/README_FACTORS.md
  - 2024_A_olympic_sde/data/raw/sde_factors_research.csv
  - 2024_A_olympic_sde/paper/himcm_paper.tex
```

## 对应关系

| IOC 叙述（论文） | 代码叶 |
| --- | --- |
| Popularity and Accessibility | Appearances（效益）、Events_Count（成本）、BROADCAST_VAL |
| Inclusivity | GLOBAL_REACH |
| Gender Equity | GENDER_PARITY |
| Relevance and Innovation | YOUTH_APPEAL |
| Sustainability | INFRA_COST |

`HOST_POPULARITY` 只出现在 processed 表，不进 `INDICATORS`、不进综合分。

xlsx 只保证 Appearances / Events_Count。`GLOBAL_REACH` 等为公开会员数约数或 Saaty 研究赋分（`README_FACTORS.md`）。`BROADCAST_VAL` 校验允许到 10。更换研究表后须重跑 loader，不得手改 processed 当官方。

## Related

- [AHP 特征值](../methods/ahp-eigenvalue.md)
- [2024 问题](../problems/2024-a-olympic-sde.md)
- [失败模式](../concepts/2024-a-failure-modes.md)

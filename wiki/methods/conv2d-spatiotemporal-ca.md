# 2D convolution spatial CA

用离散核在格网上做空间扩散/物候步进，代替纯 Python 双层循环。2023 周步引擎：`2023_A_dandelion_prisms/src/simulation/`。羽流核细节见 [WALD](wald-plume.md)、种群矩阵见 [Lefkovitch](lefkovitch-self-thinning.md)。

```yaml
type: method
year: 2023
status: ingested
sources:
  - 2023_A_dandelion_prisms/results/monthly_metrics.csv
  - 2023_A_dandelion_prisms/src/simulation
```

## 适用

公顷级格网、周步或日步、核可写成卷积。本包温带 12 月周步覆盖 **0.4511**（`monthly_metrics.csv`）。

## 失败模式

日步中心宇宙（365 日 cover 1.0）与周步西缘宇宙平均；把 Open-Meteo 下载直接覆盖权威周步 CSV。

## Related

- [2023 蒲公英](../problems/2023-a-dandelion.md)
- [WALD](wald-plume.md)
- [Fisher–KPP](fisher-kpp.md)

# Fisher–KPP 验算

用核二阶矩与 Lefkovitch 谱半径对照连续反应扩散的最小波速。

```yaml
type: method
year: 2023
status: draft
sources:
  - 2023_A_dandelion_prisms/src/theory/fisher_kpp.py
  - 2023_A_dandelion_prisms/results/kpp_check.csv
```

## 公式

\[
\partial_t u=D\nabla^2 u+ru(1-u/K),\qquad
D=\frac{M_2}{4\tau},\qquad
r=\frac{\ln\rho}{\tau},\qquad
c^*=2\sqrt{rD}.
\]

`kpp_check.csv`（常气候一周）：\(D=4.7959\,\mathrm{m}^2/\mathrm{wk}\)，\(r=0.6554\,\mathrm{wk}^{-1}\)，\(c^*=3.5458\)，一维数值波前 \(c_{\mathrm{num}}=3.3611\)，相对误差 0.0521。

## 适用条件

只验常系数 PDE 与矩匹配，不把 \(52c^*\) 写成温带 12 月锋面 46.5 m。

## Related

- [WALD 羽流](../methods/wald-plume.md)
- [2023 问题](../problems/2023-a-dandelion-prisms.md)

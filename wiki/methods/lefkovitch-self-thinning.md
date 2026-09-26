# Lefkovitch 与自疏

四阶段周步 \((S,L,R,A)\)，气候乘子 \(\varphi(T,\theta)\)，莲座+成株盖帽 \(K_{\mathrm{eff}}=K\theta^{a}\)。

```yaml
type: method
year: 2023
status: draft
sources:
  - 2023_A_dandelion_prisms/src/biology/lefkovitch.py
  - 2023_A_dandelion_prisms/src/biology/self_thinning.py
  - 2023_A_dandelion_prisms/data/parameters/biophysics.json
  - 2023_A_dandelion_prisms/tests/test_lefkovitch.py
  - 2023_A_dandelion_prisms/tests/test_self_thinning.py
```

## 公式

无产籽时线性转移使总量不增（测试锁定）。花期产籽

\[
F=A\,n_{\mathrm{cap}}\,n_{\mathrm{head}}\,\varphi\,\mathbf{1}_{\mathrm{bloom}}
\]

写入空间层，不在本模块闭环。\(K=80\,\mathrm{m}^{-2}\)，\(a=0.60\)。

## 适用条件

- 时间步必须与花期日历一致（周）。
- \(n_{\mathrm{plants}}=L+R+A\) 可以超过 \(K\)，因为幼苗不受盖帽。

## 失败模式

- 把 12 月温带 \(n_{\mathrm{plants}}\approx 1.45\times 10^6\) 写成开花成株（成株约 3444，见 `monthly_metrics.csv`）。
- 低温 \(\varphi=0\) 时仍应用花期产籽。

## Related

- [2023 问题](../problems/2023-a-dandelion-prisms.md)
- [月末指标](../sources/monthly-metrics.md)

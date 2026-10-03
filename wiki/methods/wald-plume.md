# WALD / 2D 羽流核

冠毛顺风用 Inverse-Gaussian（WALD），侧向用随下风距离变宽的高斯；离散核强制质量和为 1。

```yaml
type: method
year: 2023
status: ingested
sources:
  - 2023_A_dandelion_prisms/src/physics/wald.py
  - 2023_A_dandelion_prisms/src/physics/plume.py
  - 2023_A_dandelion_prisms/data/parameters/biophysics.json
  - 2023_A_dandelion_prisms/tests/test_physics_normalization.py
```

## 公式

\[
\mu=\frac{H\bar U}{v_t},\quad
\lambda=\frac{\mu}{\kappa^2},\quad
f(r)=\sqrt{\frac{\lambda}{2\pi r^3}}\exp\left(-\frac{\lambda(r-\mu)^2}{2\mu^2 r}\right).
\]

契约：\(H=0.35\,\mathrm{m}\)，\(v_t=0.32\,\mathrm{m/s}\)，\(\kappa=0.40\)。气象来向 \(\theta\) 的下风单位向量为 \((-\sin\theta,-\cos\theta)\)。

## 适用条件

- 干冠毛、近地层水平风；不做上升气流。
- 源在域外时，必须单独记窗外质量。
- 离岸风 + 西缘源：WALD 质量几乎全部离园，殖民靠 \(\varepsilon=0.05\) 就近滞留。

## 代码入口

`src.physics.plume.build_plume_kernel`，`deposit_from_point`。

## 失败模式

- \(\bar U\to 0\) 时 \(\mu\to 0\)：用 `u_floor_mps=0.50`。
- 连续密度直接采样不守恒：必须盖已归一化核。

## Related

- [2023 问题](../problems/2023-a-dandelion-prisms.md)
- [Fisher–KPP](../methods/fisher-kpp.md)

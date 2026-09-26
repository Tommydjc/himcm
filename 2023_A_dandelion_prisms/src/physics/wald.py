"""WALD（Inverse-Gaussian）顺风一维核。

Katul et al. (2005) 的长距扩散近似：对 \(r>0\)

.. math::

    f(r)=\\sqrt{\\frac{\\lambda}{2\\pi r^3}}
    \\exp\\left(-\\frac{\\lambda(r-\\mu)^2}{2\\mu^2 r}\\right),

其中均值与形状取维度自洽闭式

.. math::

    \\mu=\\frac{H\\bar U}{v_t},\\qquad
    \\kappa=\\frac{\\sigma_w}{\\bar U},\\qquad
    \\lambda=\\frac{\\mu}{\\kappa^2}.

此时变异系数 \(\\mathrm{CV}=\\kappa\)。本模块不旋转坐标、不做侧向扩散。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def wald_mean_shape(H_m: float, U_mps: float, v_t_mps: float, kappa: float) -> tuple[float, float]:
    """由释放高度、风速、沉降速度与湍流强度计算 \((\mu,\lambda)\)。

    Parameters
    ----------
    H_m :
        \(H\)（m）。
    U_mps :
        \(\bar U\)（m/s），须已施加地板。
    v_t_mps :
        \(v_t\)（m/s）。
    kappa :
        \(\kappa=\sigma_w/\bar U>0\)。

    Returns
    -------
    tuple[float, float]
        \((\mu,\lambda)\)，单位均为 m。
    """
    if H_m <= 0.0 or U_mps <= 0.0 or v_t_mps <= 0.0 or kappa <= 0.0:
        raise ValueError("H, U, v_t, kappa must be positive")
    mu = (H_m * U_mps) / v_t_mps
    lam = mu / (kappa * kappa)
    return float(mu), float(lam)


def wald_pdf(r_m: NDArray[np.float64], mu: float, lam: float) -> NDArray[np.float64]:
    """Inverse-Gaussian 密度 \(f(r)\)（1/m）。\(r\\le 0\) 处置 0。

    Parameters
    ----------
    r_m :
        顺风距离，形状任意（m）。
    mu, lam :
        WALD 参数（m）。
    """
    radius = np.asarray(r_m, dtype=np.float64)
    out = np.zeros_like(radius)
    pos = radius > 0.0
    if not np.any(pos):
        return out
    rp = radius[pos]
    pref = np.sqrt(lam / (2.0 * np.pi * rp**3))
    expo = -lam * (rp - mu) ** 2 / (2.0 * mu * mu * rp)
    out[pos] = pref * np.exp(np.clip(expo, -80.0, 80.0))
    return out

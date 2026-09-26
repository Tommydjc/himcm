"""密度上限自疏：对莲座+成株施加 \(K_{\\mathrm{eff}}\)。

.. math::

    K_{\\mathrm{eff}}=K\\,\\theta^{a},\\qquad
    N=R+A,\\qquad
    (R,A)\\leftarrow (R,A)\\,\\min(1,K_{\\mathrm{eff}}/N).

种子库与幼苗不受盖帽，以免把未建成株一并抹掉。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.params import Biophysics


def carrying_capacity(smi: float, params: Biophysics) -> float:
    """标量 \(K_{\\mathrm{eff}}\)（株/m²）。"""
    theta = min(max(float(smi), 0.0), 1.0)
    return float(params.K_max_per_m2 * (theta ** params.K_smi_exp))


def apply_self_thinning(
    rosette: NDArray[np.float64],
    adult: NDArray[np.float64],
    smi: float,
    params: Biophysics,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """逐格把 \(R+A\) 压到 \(K_{\\mathrm{eff}}\) 以下。"""
    k_eff = carrying_capacity(smi, params)
    density = rosette + adult
    scale = np.ones_like(density)
    over = density > k_eff
    scale[over] = k_eff / np.maximum(density[over], 1.0e-15)
    return rosette * scale, adult * scale

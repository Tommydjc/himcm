"""PyTorch 动态计算图上的可微蜂群全年仿真（DFM 五维）。

状态 ``[B, H, F, P, N]``；待分析参数是 ``nn.Parameter``；
365 步显式欧拉 / RK2 展开计算图，供 Autograd 灵敏度。

单位：蜂口为个体数；花粉 / 储蜜为克（g）。
"""

from __future__ import annotations

import time
from typing import Dict, Literal, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS: float = 1.0e-8
DAYS_PER_YEAR: int = 365


def _hill2(x: torch.Tensor, k: torch.Tensor | float, eps: float = EPS) -> torch.Tensor:
    r"""Holling-III：\(x^2/(K^2+x^2+\varepsilon)\)，全程可微。"""

    x2 = x * x
    kk = k * k if torch.is_tensor(k) else float(k) * float(k)
    return x2 / (kk + x2 + eps)


def _softplus_beta(x: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    r"""稳定 Softplus：\(\beta^{-1}\log(1+e^{\beta x})\)。"""

    return F.softplus(x * beta, threshold=20.0) / beta


def _project_positive(
    y: torch.Tensor,
    beta: float = 20.0,
    floor: float = 1.0e-4,
) -> torch.Tensor:
    """光滑非负投影，避免 ``clamp`` 在边界把梯度掐死为 0。"""

    return _softplus_beta(y - floor, beta=beta) + floor


class DifferentiableColonySimulator(nn.Module):
    r"""可微蜂群全年仿真器（显式步进展开）。

    叶子参数（``requires_grad=True``）
        - ``laying_rate_max`` → DFM \(L_{\max}\)（eggs/day）
        - ``forager_mortality`` → \(d_{F,\max}\)（day\(^{-1}\)）
        - ``pollen_intake_rate`` → \(\eta_P\)（g/(forager·day)）
        - ``nectar_intake_rate`` → \(\eta_N\)（g/(forager·day)）
        - ``social_inhibition_k`` → \(\sigma_F\)（day\(^{-1}\)）

    Args:
        days: 前向天数，默认 365。
        dt: 步长（日），默认 1.0。
        method: ``"euler"`` 或 ``"rk2"``。
        dtype: 张量精度；灵敏度建议 ``float64``。
        device: 计算设备；默认 CPU。
    """

    def __init__(
        self,
        days: int = DAYS_PER_YEAR,
        dt: float = 1.0,
        method: Literal["euler", "rk2"] = "euler",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        laying_rate_max: float = 1600.0,
        forager_mortality: float = 0.14,
        pollen_intake_rate: float = 0.10,
        nectar_intake_rate: float = 0.35,
        social_inhibition_k: float = 10.0,
    ) -> None:
        super().__init__()
        if method not in ("euler", "rk2"):
            raise ValueError(f"method must be 'euler' or 'rk2', got {method!r}")
        self.days = int(days)
        self.dt = float(dt)
        self.method = method
        self.dtype = dtype
        dev = torch.device(device) if device is not None else torch.device("cpu")

        self.laying_rate_max = nn.Parameter(
            torch.tensor(laying_rate_max, dtype=dtype, device=dev)
        )
        self.forager_mortality = nn.Parameter(
            torch.tensor(forager_mortality, dtype=dtype, device=dev)
        )
        self.pollen_intake_rate = nn.Parameter(
            torch.tensor(pollen_intake_rate, dtype=dtype, device=dev)
        )
        self.nectar_intake_rate = nn.Parameter(
            torch.tensor(nectar_intake_rate, dtype=dtype, device=dev)
        )
        self.social_inhibition_k = nn.Parameter(
            torch.tensor(social_inhibition_k, dtype=dtype, device=dev)
        )

        def buf(name: str, value: float) -> None:
            self.register_buffer(name, torch.tensor(value, dtype=dtype, device=dev))

        buf("gamma_B", 1.0 / 21.0)
        buf("d_B", 0.01)
        buf("d_H", 0.005)
        buf("d_F_min", 0.007)
        buf("alpha_0", 0.20)
        buf("alpha_starve", 0.12)
        buf("K_P", 2000.0)
        buf("K_N", 4000.0)
        buf("K_N_flight", 800.0)
        buf("k_care", 0.5)
        buf("K_deplete", 500.0)
        buf("c_P_B", 0.00020)
        buf("c_P_H", 0.00004)
        buf("c_P_F", 0.00003)
        buf("c_N_B", 0.00010)
        buf("c_N_H", 0.00008)
        buf("c_N_F", 0.00045)
        buf("t_peak", 172.0)
        buf("theta_season", 2.0)
        buf("gauss_sigma", 45.0)
        buf("softplus_beta", 10.0)
        buf("project_beta", 20.0)
        buf("mu_env_amp", 0.0)
        buf("eps", EPS)
        # 三重胁迫（默认关闭；由 configure_stress 写入）
        buf("pesticide_index", 0.0)  # A: μ_F' = μ_F (1+P)
        buf("varroa_load", 0.0)  # B: 羽化衰减强度
        buf("varroa_hive_mort_mult", 1.0)  # B: 内勤死亡乘子（腰斩寿命→2）
        buf("frost_start", -1.0)  # C: 倒春寒起止日；<0 表示关闭
        buf("frost_end", -1.0)

        self.register_buffer(
            "y0_default",
            torch.tensor(
                [14000.0, 15000.0, 5000.0, 15000.0, 20000.0],
                dtype=dtype,
                device=dev,
            ),
        )

    def configure_stress(
        self,
        pesticide_index: float = 0.0,
        varroa_load: float = 0.0,
        varroa_hive_mort_mult: float = 1.0,
        frost_start: float = -1.0,
        frost_end: float = -1.0,
    ) -> None:
        r"""写入三重外界胁迫强度（不改叶子参数身份）。

        Args:
            pesticide_index:
                胁迫 A。有效外勤死亡 \(\mu_F'=\mu_F(1+P)\)。
            varroa_load:
                胁迫 B。羽化率 \(\gamma_B'=\gamma_B e^{-v}\)。
            varroa_hive_mort_mult:
                胁迫 B。内勤死亡乘子；寿命腰斩时取 ``2.0``。
            frost_start, frost_end:
                胁迫 C。倒春寒窗内觅食通量置零；``start<0`` 关闭。
        """

        self.pesticide_index.fill_(float(pesticide_index))
        self.varroa_load.fill_(float(varroa_load))
        self.varroa_hive_mort_mult.fill_(float(varroa_hive_mort_mult))
        self.frost_start.fill_(float(frost_start))
        self.frost_end.fill_(float(frost_end))

    def leaf_parameter_dict(self) -> Dict[str, nn.Parameter]:
        """返回五个待灵敏度分析的叶子参数。"""

        return {
            "laying_rate_max": self.laying_rate_max,
            "forager_mortality": self.forager_mortality,
            "pollen_intake_rate": self.pollen_intake_rate,
            "nectar_intake_rate": self.nectar_intake_rate,
            "social_inhibition_k": self.social_inhibition_k,
        }

    def seasonal_omega(self, t: torch.Tensor) -> torch.Tensor:
        r"""物候波 \(\Omega(t)=\bigl[(1+\cos(2\pi(t-t_p)/365))/2\bigr]^\theta\)。"""

        phase = 0.5 * (1.0 + torch.cos(2.0 * torch.pi * (t - self.t_peak) / 365.0))
        return torch.exp(self.theta_season * torch.log(phase + self.eps))

    def laying_gaussian(self, t: torch.Tensor) -> torch.Tensor:
        r"""夏至产卵高斯峰 \(G(t)=\exp\bigl(-\frac12((t-t_p)/\sigma)^2\bigr)\)。"""

        z = (t - self.t_peak) / (self.gauss_sigma + self.eps)
        return torch.exp(-0.5 * z * z)

    def temperature_c(self, t: torch.Tensor) -> torch.Tensor:
        r"""光滑气温日历 \(T(t)=11+13\sin(2\pi(t-110)/365)\)（℃）。"""

        return 11.0 + 13.0 * torch.sin(2.0 * torch.pi * (t - 110.0) / 365.0)

    def _frost_mask(self, t: torch.Tensor) -> torch.Tensor:
        """倒春寒窗内为 1，否则 0（光滑阶跃用乘积，窗关闭时恒 0）。"""

        active = (self.frost_start >= 0.0).to(dtype=t.dtype)
        inside = ((t >= self.frost_start) & (t <= self.frost_end)).to(dtype=t.dtype)
        return active * inside

    def rhs(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""DFM 右端 \(f(t,y)\)，形状 ``(5,)``；含可选三重胁迫。"""

        y_pos = _project_positive(y, beta=float(self.project_beta.item()))
        B, H, F_bee, P, N = y_pos[0], y_pos[1], y_pos[2], y_pos[3], y_pos[4]
        eps = self.eps

        omega = self.seasonal_omega(t)
        gauss = self.laying_gaussian(t)
        season = omega * gauss
        frost = self._frost_mask(t)
        # 胁迫 C：霜冻日觅食活动 = 0，纯消耗库存
        forage_act = (1.0 - frost) * omega
        kappa_P = forage_act
        kappa_N = forage_act
        mu_env = self.mu_env_amp * omega

        L_max = self.laying_rate_max
        d_F_max = self.forager_mortality
        eta_P = self.pollen_intake_rate
        eta_N = self.nectar_intake_rate
        sigma_F = self.social_inhibition_k

        # 胁迫 B：瓦螨 → 羽化衰减、内勤寿命下降
        gamma_B = self.gamma_B * torch.exp(-self.varroa_load)
        d_H = self.d_H * self.varroa_hive_mort_mult

        Lambda = (
            L_max
            * season
            * _hill2(P, self.K_P, float(eps))
            * _hill2(N, self.K_N, float(eps))
        )
        care_den = (self.k_care * B) ** 2 + H * H + eps
        f_care = (H * H) / care_den

        ratio_F = F_bee / (H + F_bee + eps)
        pollen_sat = P / (self.K_P + P + eps)
        recruit_arg = (
            self.alpha_0
            - sigma_F * ratio_F
            + self.alpha_starve * (1.0 - pollen_sat)
        )
        R_dfm = H * _softplus_beta(recruit_arg, beta=float(self.softplus_beta.item()))

        d_F = self.d_F_min + (d_F_max - self.d_F_min) * omega + mu_env
        # 胁迫 A：新烟碱迷巢 → 有效死亡率放大
        d_F = d_F * (1.0 + self.pesticide_index)

        I_P = eta_P * kappa_P * F_bee * _hill2(N, self.K_N_flight, float(eps))
        C_P = (self.c_P_B * B + self.c_P_H * H + self.c_P_F * F_bee) * (
            P / (self.K_deplete + P + eps)
        )
        I_N = eta_N * kappa_N * F_bee
        C_N = (self.c_N_B * B + self.c_N_H * H + self.c_N_F * F_bee) * (
            N / (self.K_deplete + N + eps)
        )

        dB = Lambda * f_care - (gamma_B + self.d_B) * B
        dH = gamma_B * B - R_dfm - d_H * H
        dF = R_dfm - d_F * F_bee
        dP = I_P - C_P
        dN = I_N - C_N
        return torch.stack([dB, dH, dF, dP, dN])

    def recruit_rate(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""瞬时内勤→外勤转化率 \(r=R_{\mathrm{DFM}}/(H+\varepsilon)\)（day\(^{-1}\)）。"""

        y_pos = _project_positive(y, beta=float(self.project_beta.item()))
        H = y_pos[1]
        F_bee = y_pos[2]
        P = y_pos[3]
        eps = self.eps
        ratio_F = F_bee / (H + F_bee + eps)
        pollen_sat = P / (self.K_P + P + eps)
        recruit_arg = (
            self.alpha_0
            - self.social_inhibition_k * ratio_F
            + self.alpha_starve * (1.0 - pollen_sat)
        )
        return _softplus_beta(recruit_arg, beta=float(self.softplus_beta.item()))

    def _step(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """单步欧拉或 RK2，再做光滑非负投影。"""

        dt = self.dt
        if self.method == "euler":
            y_next = y + dt * self.rhs(t, y)
        else:
            k1 = self.rhs(t, y)
            y_mid = y + dt * k1
            k2 = self.rhs(t + dt, y_mid)
            y_next = y + 0.5 * dt * (k1 + k2)
        return _project_positive(y_next, beta=float(self.project_beta.item()))

    def forward(
        self,
        y0: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """展开 ``days`` 步；返回 ``B,H,F,P,N`` 轨迹（各 ``(days,)``）。"""

        y = self.y0_default.clone() if y0 is None else y0.to(dtype=self.dtype)
        if y.shape != (5,):
            raise ValueError(f"y0 must have shape (5,), got {tuple(y.shape)}")
        y = _project_positive(y, beta=float(self.project_beta.item()))

        hist: list[torch.Tensor] = []
        temps: list[torch.Tensor] = []
        omegas: list[torch.Tensor] = []
        recruits: list[torch.Tensor] = []
        for k in range(self.days):
            t = torch.tensor(float(k + 1), dtype=self.dtype, device=y.device)
            hist.append(y)
            temps.append(self.temperature_c(t))
            omegas.append(self.seasonal_omega(t))
            recruits.append(self.recruit_rate(t, y))
            y = self._step(t, y)

        traj = torch.stack(hist, dim=0)
        adults = traj[:, 1] + traj[:, 2]
        return {
            "B": traj[:, 0],
            "H": traj[:, 1],
            "F": traj[:, 2],
            "P": traj[:, 3],
            "N": traj[:, 4],
            "adults": adults,
            "temp_c": torch.stack(temps),
            "omega": torch.stack(omegas),
            "recruit_rate": torch.stack(recruits),
        }

    def scalar_loss(
        self,
        trajectories: Mapping[str, torch.Tensor],
        target_end_adults: float = 15000.0,
    ) -> torch.Tensor:
        """年末成蜂相对目标的平方误差（自检用）。"""

        n_end = trajectories["adults"][-1]
        target = torch.tensor(target_end_adults, dtype=n_end.dtype, device=n_end.device)
        return (n_end - target) ** 2


def run_self_check(time_budget_s: float = 0.5) -> None:
    """快速自检：365 步前向 < budget，且 ``loss.backward()`` 成功。"""

    sim = DifferentiableColonySimulator(days=365, dt=1.0, method="euler")
    t0 = time.perf_counter()
    out = sim.forward()
    t_fwd = time.perf_counter() - t0
    loss = sim.scalar_loss(out)
    t1 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t1
    grads = {
        name: (None if p.grad is None else float(p.grad.detach()))
        for name, p in sim.leaf_parameter_dict().items()
    }
    print(
        f"[torch_sim self-check] forward_s={t_fwd:.4f} backward_s={t_bwd:.4f} "
        f"loss={float(loss.detach()):.4g} end_adults={float(out['adults'][-1].detach()):.2f}"
    )
    print(f"[torch_sim self-check] grads={grads}")
    if t_fwd >= time_budget_s:
        raise RuntimeError(f"forward took {t_fwd:.4f}s >= budget {time_budget_s}s")
    if any(g is None for g in grads.values()):
        raise RuntimeError("one or more leaf parameter grads are None")
    print("[torch_sim self-check] OK")


if __name__ == "__main__":
    run_self_check()

# Requirement 2：基础办公室场景的解析推导

**问题**：2025 HiMCM Problem A  
**场景**：Figure 1 基础办公室（双侧 6 室 + 中央走廊 + 东西出口）  
**策略**：两名消防员独立对称扫荡（Split-Half）

---

## 0. 图论表示与符号

将楼层建成沿走廊的 **1 维骨干 + 房间悬挂** 图。西出口 \(E_W\) 取 \(x=0\)，东出口 \(E_E\) 取 \(x=L=30\,\mathrm{m}\)。三开间各 \(10\,\mathrm{m}\)，与 Figure 1 门位一致：西列门靠近开间东沿、东列门靠近开间西沿、中列门在开间中部，故

\[
x_{N_1}=x_{S_1}=10,\quad x_{N_2}=x_{S_2}=15,\quad x_{N_3}=x_{S_3}=20.
\]

| 符号 | 含义 | 取值 |
|---|---|---|
| \(N_1,N_2,N_3\) | 北侧西 / 中 / 东办公室 | — |
| \(S_1,S_2,S_3\) | 南侧西 / 中 / 东办公室 | — |
| \(L\) | 走廊长度 | \(30\,\mathrm{m}\) |
| \(D\) | 房间进深 | \(8\,\mathrm{m}\) |
| \(v_h\) | 走廊移动速度 | \(1.5\,\mathrm{m/s}\) |
| \(v_r\) | 室内搜寻速度 | \(0.8\,\mathrm{m/s}\) |
| \(t_{\mathrm{sw}}\) | 视觉扫荡确认耗时 | \(20\,\mathrm{s}\) |
| \(t_{\mathrm{tag}}\) | 门把手挂“已清空”标卡 | \(5\,\mathrm{s}\) |
| \(W_h\) | 走廊净宽（横向穿越） | 未给定；1D 模型取 \(0\) |

**单室服务时间**（进门 \(\to\) 抵进深墙 \(\to\) 视觉确认 \(\to\) 返回门口 \(\to\) 挂牌）：

\[
t_{\mathrm{svc}}
= \frac{2D}{v_r} + t_{\mathrm{sw}} + t_{\mathrm{tag}}
= \frac{2\cdot 8}{0.8} + 20 + 5
= 45\,\mathrm{s}.
\]

对向门在 1D 模型中共横坐标；若计横向穿越，穿越时间为 \(W_h/v_h\)。下文默认 \(W_h=0\)，并保留含 \(W_h\) 的一般式。

**清空时刻** \(T_{\mathrm{clear}}\)：最后一间房挂牌完成的时刻（不等待两人回到出口）。若赛题要求“搜完并撤出”，在末项加 \(\mathrm{dist}(\text{末门},E)/v_h\)。

> **建模注记**：\(t_{\mathrm{sw}}\) 与 \(2D/v_r\) 可能部分重叠。若将 \(t_{\mathrm{sw}}\) 视为覆盖 \(10\times 8\) 的全部搜索，则应改用 \(t_{\mathrm{svc}}=2D/v_r+t_{\mathrm{tag}}\) 或 \(t_{\mathrm{svc}}=t_{\mathrm{sw}}+t_{\mathrm{tag}}\)。按题目所列参数，本文采用 **加性全模型**。

---

## 1. Split-Half 路径拓扑序列

独立对称扫荡定义为：按走廊中线 \(x=L/2=15\) **地理对分**，每人 3 间、互不进入对方责任区。这是 6 室、两端同时进入时的负载均衡剖分：西列两间归 \(F_1\)，东列两间归 \(F_2\)，两间中室各分一间。

### 1.1 责任分区

\[
\mathcal{R}_1=\{N_1,S_1,N_2\},\qquad
\mathcal{R}_2=\{N_3,S_3,S_2\}.
\]

中室南北对调 \(\{S_2\}\leftrightarrow\{N_2\}\) 时，完成时间不变。

### 1.2 最短走廊覆盖序（列优先、再向内）

对向门同 \(x\)，先清完一列再走 \(5\,\mathrm{m}\) 到中门，走廊弧长最短。

\[
\begin{aligned}
F_1:\quad
& E_W \;\to\; N_1 \;\to\; S_1 \;\to\; N_2 \\
F_2:\quad
& E_E \;\to\; N_3 \;\to\; S_3 \;\to\; S_2
\end{aligned}
\]

等价蛇形（时间相同）：

\[
F_1:\ E_W\to S_1\to N_1\to N_2,\qquad
F_2:\ E_E\to S_3\to N_3\to S_2.
\]

### 1.3 不应采用的剖分

- **南北对分**（\(F_1\) 清全部北室）：走廊需走到 \(x=20\)，弧长 \(20\,\mathrm{m}>15\,\mathrm{m}\)。
- **\(2+4\) 不均分**：makespan 由 4 室一侧决定，劣于 \(3+3\)。

### 1.4 走廊轨迹与弧长

每人走廊轨迹是简单路（非闭回路）：

\[
\pi_1:\ 0\to 10\to 10\to 15,\qquad
\pi_2:\ 30\to 20\to 20\to 15.
\]

弧长均为

\[
d_{\mathrm{hall}} = 15 + W_h \quad (\mathrm{m}).
\]

---

## 2. 理论最短清空时间

### 2.1 个体完成时间

\[
T^{(i)}
= \frac{d_{\mathrm{hall}}}{v_h} + |\mathcal{R}_i|\, t_{\mathrm{svc}}
= \frac{15+W_h}{v_h} + 3\, t_{\mathrm{svc}},\quad i=1,2.
\]

### 2.2 Makespan

两人并行、无共享房间，

\[
T_{\mathrm{clear}}^{\mathrm{SH}}
= \max\bigl(T^{(1)},T^{(2)}\bigr)
= \frac{15+W_h}{v_h} + 3\left(\frac{2D}{v_r}+t_{\mathrm{sw}}+t_{\mathrm{tag}}\right).
\]

\(W_h=0\) 时：

\[
T_{\mathrm{clear}}^{\mathrm{SH}}
= \frac{15}{1.5} + 3\times 45
= 10 + 135
= 145\,\mathrm{s}.
\]

### 2.3 下界论证（何以称为理论最短）

将任务看成 **带服务时间的 2-agent 路径覆盖**。任意可行方案满足

\[
T_{\mathrm{clear}}
\ge
\max_{i=1,2}
\left\{
\frac{\mathrm{dist}(E^{(i)},\mathrm{doors}(\mathcal{R}_i))}{v_h}
+ |\mathcal{R}_i|\, t_{\mathrm{svc}}
\right\}
\ge
3\, t_{\mathrm{svc}} + \frac{d^\star}{v_h}.
\]

理由：

1. \(\sum_i |\mathcal{R}_i|=6\) 且房间服务不可由同一消防员并行，故 \(\max_i|\mathcal{R}_i|\ge 3\)，房间项 \(\ge 3 t_{\mathrm{svc}}=135\,\mathrm{s}\)。
2. \(F_1\) 必须覆盖门集 \(\{10,15\}\)（西列 + 一间中室），从 \(x=0\) 出发的最短路径长度 \(\ge 15\,\mathrm{m}\)；\(F_2\) 对称。故 \(d^\star\ge 15\)。

Split-Half 同时取到这两个下界。因此，在 **“单人清室、无对向穿越、清空不要求返回出口”** 的模型里，\(145\,\mathrm{s}\) 即理论最短时间。

一般式（两端进入、\(n\) 间均分）：

\[
T_{\mathrm{clear}}^{\min}
= \frac{L/2+W_h}{v_h} + \left\lceil\frac{n}{2}\right\rceil t_{\mathrm{svc}}
\qquad (n=6\Rightarrow\lceil n/2\rceil=3).
\]

若必须返回最近出口：对称方案的末门已在 \(x=15\)，再加 \(15/v_h=10\,\mathrm{s}\)，得

\[
T_{\mathrm{egress}}=155\,\mathrm{s}.
\]

**门位稳健性**：若把门改到开间中点 \(\{5,15,25\}\)，西列 + 中室的最短走廊仍是 \(15\,\mathrm{m}\)，\(T_{\mathrm{clear}}^{\mathrm{SH}}\) 不变。

---

## 3. 浓烟降速与高危交叉复核的公式修正

### 3.1 走廊降速

仅走廊受烟（室内 \(v_r\) 不变，除非另设）：

\[
\alpha=0.40,\qquad
v_h'=(1-\alpha)v_h=0.9\,\mathrm{m/s}.
\]

主搜阶段走廊项变为 \((15+W_h)/v_h'\)。

### 3.2 交叉复核

记高危集 \(\mathcal{R}_H\subseteq\{N_1,N_2,N_3,S_1,S_2,S_3\}\)，复核强度 \(\gamma\in(0,1]\)（全文再搜 \(\gamma=1\)；抽查可 \(\gamma<1\)）。

**交叉** 指：主搜责任人不复核自己的高危房，由另一人复核。

主搜结束后 \(F_1\) 在 \(N_2\)（\(x=15\)），\(F_2\) 在 \(S_2\)（\(x=15\)）。令

\[
\mathcal{R}_H^{(1)}=\mathcal{R}_H\cap\mathcal{R}_1,\qquad
\mathcal{R}_H^{(2)}=\mathcal{R}_H\cap\mathcal{R}_2.
\]

则 \(F_2\) 复核 \(\mathcal{R}_H^{(1)}\)，\(F_1\) 复核 \(\mathcal{R}_H^{(2)}\)。

**修正公式（主搜 + 交叉复核，makespan）**

\[
T_{\mathrm{clear}}^{\mathrm{RV}}
=
\max_{i=1,2}
\Biggl[
\frac{15+W_h}{v_h'} + 3\, t_{\mathrm{svc}}
+
\frac{\ell_i(\mathcal{R}_H)}{v_h'} + \gamma\,|\mathcal{R}_H^{(3-i)}|\, t_{\mathrm{svc}}
\Biggr].
\]

其中 \(\ell_i(\mathcal{R}_H)\) 是从该消防员主搜终点出发、覆盖对方高危门集的最短路长。

### 3.3 典型高危集

#### （A）仅中室高危 \(\mathcal{R}_H=\{N_2,S_2\}\)

两人已在 \(x=15\)，交叉只需横穿走廊：\(\ell_i=W_h\)，各复核 1 间。

\[
T_{\mathrm{clear}}^{\mathrm{RV}}
= \frac{15+2W_h}{v_h'} + (3+\gamma)\, t_{\mathrm{svc}}.
\]

\(W_h=0,\ \gamma=1\)：

\[
\frac{15}{0.9} + 4\times 45 = 16.67 + 180 = 196.67\,\mathrm{s}.
\]

相对无烟无复核的 \(145\,\mathrm{s}\)：走廊 \(+6.67\,\mathrm{s}\)，复核 \(+45\,\mathrm{s}\)。

#### （B）全部 6 室高危

\(F_1\) 主搜完在 \(N_2\)，需覆盖 \(\{N_3,S_3,S_2\}\)：\(15\to 20\) 再对向门，\(\ell_1=5+W_h\)；\(F_2\) 对称。各加 3 间复核。

\[
T_{\mathrm{clear}}^{\mathrm{RV}}
= \frac{15+W_h}{v_h'} + 3 t_{\mathrm{svc}}
+ \frac{5+W_h}{v_h'} + 3\gamma\, t_{\mathrm{svc}}
= \frac{20+2W_h}{v_h'} + 3(1+\gamma)\, t_{\mathrm{svc}}.
\]

\(\gamma=1,W_h=0\)：

\[
\frac{20}{0.9} + 270 \approx 22.22 + 270 = 292.22\,\mathrm{s}.
\]

### 3.4 对照：双人同室（2-in-2-out），非交叉

高危房必须两人同时在场时，高危服务不能分给两条时间轴。需先约在第一间高危房：

\[
T_{\mathrm{clear}}^{\mathrm{buddy}}
= T_{\mathrm{low\text{-}risk}}^{\parallel}
+ T_{\mathrm{rendezvous}}
+ |\mathcal{R}_H|\, t_{\mathrm{svc}}
+ T_{\mathrm{tour}}(\mathcal{R}_H; v_h').
\]

对 \(\mathcal{R}_H=\{N_2,S_2\}\)，低危（西列 / 东列）已并行清完时会合等待为 0，再串行 2 间：

\[
T = \frac{15+W_h}{v_h'} + 4 t_{\mathrm{svc}}.
\]

与（A）且 \(\gamma=1\) 数值相同，但机制不同：buddy 是会合后 **串行** 清 2 间；交叉是各做 1 间对方的房。高危多于 2 间时，buddy 将 \(|\mathcal{R}_H|\,t_{\mathrm{svc}}\) 全部串行，交叉仍可两人同时复核不同房间，**交叉通常严格更短**。

---

## 4. 数值汇总（解析基线）

| 情景 | 公式 | \(W_h=0\) 数值 |
|---|---|---|
| Split-Half，基准 | \(\dfrac{15}{v_h}+3t_{\mathrm{svc}}\) | \(145\,\mathrm{s}\) |
| Split-Half，返回出口 | \(\dfrac{30}{v_h}+3t_{\mathrm{svc}}\) | \(155\,\mathrm{s}\) |
| 仅浓烟 \(\alpha=0.4\) | \(\dfrac{15}{(1-\alpha)v_h}+3t_{\mathrm{svc}}\) | \(166.67\,\mathrm{s}\) |
| 浓烟 + 中室交叉复核 \(\gamma=1\) | \(\dfrac{15}{v_h'}+4t_{\mathrm{svc}}\) | \(196.67\,\mathrm{s}\) |
| 浓烟 + 全室交叉复核 \(\gamma=1\) | \(\dfrac{20}{v_h'}+6t_{\mathrm{svc}}\) | \(292.22\,\mathrm{s}\) |

上述闭式结果可作为传统规划器 / RL 实验的 **analytical baseline**。

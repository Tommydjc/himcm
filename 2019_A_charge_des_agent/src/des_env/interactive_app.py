"""
Streamlit sandbox: interactive SimPy + BDI airport charging DES.

Launch from pack root:
  PYTHONPATH=src streamlit run src/des_env/interactive_app.py --server.port 8506
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PACK_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PACK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from des_env.airport_sim import SCENARIO_PRESETS, run_airport_simulation
except ImportError as exc:
    st.error(f"Cannot import airport_sim: {exc}")
    st.stop()


SCENARIO_LABELS: Dict[str, str] = {
    "normal": "平时标准客流 (Normal Day)",
    "morning_rush": "早高峰集中离港 (Morning Rush)",
    "thunderstorm": "极端雷暴大面积延误 (Thunderstorm Delay)",
}
LABEL_TO_KEY = {v: k for k, v in SCENARIO_LABELS.items()}


def _run_sim(
    c_fast: int,
    c_slow: int,
    c_bank: int,
    scenario_key: str,
    seed: int,
) -> Tuple[Dict, pd.DataFrame]:
    """Run DES and return metrics dict + timeline DataFrame."""
    metrics = run_airport_simulation(
        c_fast=c_fast,
        c_slow=c_slow,
        c_bank=c_bank,
        scenario=scenario_key,
        seed=seed,
    )
    timeline = pd.DataFrame(metrics.get("queue_timeline", []))
    if not timeline.empty:
        timeline["time_hours"] = timeline["time_minutes"] / 60.0 + 6.0
    return metrics, timeline


st.set_page_config(
    page_title="Airport Charging Sim - 2019 HiMCM",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("机场航站楼移动充电基础设施：动态离散事件仿真沙盒")
st.markdown(
    "SimPy DES + BDI passenger model (arXiv:2602.13878). "
    "调节左侧硬件与航班情景，推演 16 小时（06:00–22:00）拥堵与分流。"
)

st.sidebar.header("硬件基础设施配置")
c_fast = st.sidebar.slider(
    "65W 快充立柱数量 (台)",
    min_value=1,
    max_value=40,
    value=12,
    step=1,
    help="单台成本约 $2,000，平均充电约 25 分钟",
)
c_slow = st.sidebar.slider(
    "15W 慢充安全储物柜数量 (台)",
    min_value=0,
    max_value=30,
    value=15,
    step=1,
    help="单台成本约 $1,200，平均充电约 55 分钟",
)
c_bank = st.sidebar.slider(
    "共享充电宝机柜组数 (组)",
    min_value=0,
    max_value=8,
    value=2,
    step=1,
    help="单组 48 块，建设成本约 $3,500",
)
seed = st.sidebar.number_input("随机种子", min_value=0, max_value=10_000, value=42, step=1)

st.sidebar.header("航班运行情景")
scenario_label = st.sidebar.radio(
    "选择客流工况",
    list(SCENARIO_LABELS.values()),
)
scenario_key = LABEL_TO_KEY[scenario_label]
preset = SCENARIO_PRESETS[scenario_key]
st.sidebar.caption(
    f"λ₀={preset['lambda_base']:.2f}/min，峰值倍率={preset['peak_multiplier']:.0f}×，"
    f"定价系数={preset['pricing_mode']:.1f}"
)

total_capex = c_fast * 2000 + c_slow * 1200 + c_bank * 3500
total_area = c_fast * 0.8 + c_slow * 2.0 + c_bank * 0.5
budget = 80_000.0
area_cap = 40.0
st.sidebar.markdown("---")
st.sidebar.metric(
    "基础设施建设总投资 (CapEx)",
    f"${total_capex:,.0f}",
    delta=f"${budget - total_capex:,.0f} 预算结余"
    if total_capex <= budget
    else "超预算",
    delta_color="normal" if total_capex <= budget else "inverse",
)
st.sidebar.metric(
    "航站楼占地面积",
    f"{total_area:.1f} m²",
    delta=f"{area_cap - total_area:.1f} m² 剩余"
    if total_area <= area_cap
    else "超面积",
    delta_color="normal" if total_area <= area_cap else "inverse",
)

run_btn = st.sidebar.button("运行仿真", type="primary", use_container_width=True)

if "last_metrics" not in st.session_state or run_btn:
    with st.spinner("SimPy 正在步进 16 小时旅客事件..."):
        metrics, timeline = _run_sim(c_fast, c_slow, c_bank, scenario_key, int(seed))
    st.session_state["last_metrics"] = metrics
    st.session_state["last_timeline"] = timeline
    st.session_state["last_cfg"] = {
        "c_fast": c_fast,
        "c_slow": c_slow,
        "c_bank": c_bank,
        "scenario": scenario_key,
        "capex": total_capex,
        "area": total_area,
    }

metrics = st.session_state["last_metrics"]
timeline = st.session_state["last_timeline"]
cfg = st.session_state["last_cfg"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "全天电量焦虑旅客",
        f"{metrics.get('anxious_passengers', 0):,} 人",
        "电量 < 15%",
    )
with col2:
    mean_wait = float(metrics.get("mean_wait_time", 0.0))
    st.metric(
        "旅客平均排队等待",
        f"{mean_wait:.1f} 分钟",
        delta="优良 (<5min)" if mean_wait <= 5 else "拥堵",
        delta_color="inverse" if mean_wait > 5 else "normal",
    )
with col3:
    reneging_rate = float(metrics.get("reneging_rate", 0.0)) * 100.0
    st.metric(
        "弃排离队率",
        f"{reneging_rate:.1f}%",
        delta="安全 (<5%)" if reneging_rate <= 5 else "误机风险",
        delta_color="inverse" if reneging_rate > 5 else "normal",
    )
with col4:
    max_load = int(metrics.get("max_system_load", metrics.get("max_queue_len", 0)))
    st.metric(
        "峰值系统负荷 (等待+在服)",
        f"{max_load} 人",
        delta="高峰过载" if max_load > 25 else "平稳",
        delta_color="inverse" if max_load > 25 else "normal",
    )

st.caption(
    f"配置快照: C_fast={cfg['c_fast']}, C_slow={cfg['c_slow']}, C_bank={cfg['c_bank']}, "
    f"scenario={cfg['scenario']}, 到达={metrics.get('total_passengers', 0)}, "
    f"纯等待峰值={metrics.get('max_queue_len', 0)}"
)

st.markdown("### 16 小时动态波形与服务分流")
tab1, tab2 = st.tabs(["拥堵与到达强度", "旅客补电方式分流"])

with tab1:
    if timeline is not None and not timeline.empty:
        fig_line = go.Figure()
        fig_line.add_trace(
            go.Scatter(
                x=timeline["time_hours"],
                y=timeline["system_load"],
                mode="lines",
                name="系统负荷 (等待+在服)",
                line=dict(color="#2980b9", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(41, 128, 185, 0.15)",
            )
        )
        fig_line.add_trace(
            go.Scatter(
                x=timeline["time_hours"],
                y=timeline["in_service"],
                mode="lines",
                name="在服人数",
                line=dict(color="#27ae60", width=1.8),
            )
        )
        fig_line.add_trace(
            go.Scatter(
                x=timeline["time_hours"],
                y=timeline["queue_length"],
                mode="lines",
                name="纯等待队长",
                line=dict(color="#e74c3c", width=1.5, dash="dash"),
            )
        )
        fig_line.add_trace(
            go.Scatter(
                x=timeline["time_hours"],
                y=timeline["lambda_t"],
                mode="lines",
                name="到达强度 λ(t)",
                yaxis="y2",
                line=dict(color="#8e44ad", width=1.5),
            )
        )
        fig_line.add_hline(
            y=15,
            line_dash="dot",
            line_color="orange",
            annotation_text="预警线 (15人)",
        )
        fig_line.update_layout(
            title="航站楼充电拥堵与 NHPP 波峰 (06:00–22:00)",
            xaxis_title="时刻 (24h)",
            yaxis_title="人数",
            yaxis2=dict(
                title="到达率 (人/分钟)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            hovermode="x unified",
            height=460,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.info(
            "说明：基线硬件较大时「纯等待队长」可为 0；"
            "请看「系统负荷」与「在服人数」反映真实占用。仅当容量不足时红色虚线才会抬升。"
        )
    else:
        st.info("暂无时序数据，请点击左侧「运行仿真」。")

with tab2:
    labels = ["65W 快充立柱", "15W 慢充储物柜", "移动共享充电宝", "放弃排队 (弃排)"]
    values = [
        metrics.get("served_fast", 0),
        metrics.get("served_slow", 0),
        metrics.get("served_powerbank", 0),
        metrics.get("reneged_passengers", 0),
    ]
    colors = ["#3498db", "#9b59b6", "#2ecc71", "#e74c3c"]
    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker_colors=colors,
            )
        ]
    )
    fig_donut.update_layout(
        title="全天电量焦虑旅客最终分流",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

st.markdown("### 运筹诊断")
if mean_wait > 5.0 or reneging_rate > 5.0:
    st.warning(
        f"**系统过载。** 平均等待 {mean_wait:.1f} min，弃排率 {reneging_rate:.1f}%。 "
        "可减少慢充柜、增加快充桩或充电宝组数，再点「运行仿真」。"
    )
else:
    budget_ok = cfg["capex"] <= budget
    st.success(
        f"**当前配置运行平稳。** 平均等待 {mean_wait:.1f} min，弃排率 {reneging_rate:.1f}%， "
        f"CapEx ${cfg['capex']:,.0f}"
        + ("（预算内）" if budget_ok else "（已超 $80k 预算）")
        + "。"
    )

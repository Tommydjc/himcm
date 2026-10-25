"""
Streamlit sandbox: Teaching Day + Legacy Week + Harness (2018 HiMCM B).

Launch from pack root:
  MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src streamlit run src/ui/interactive_app.py \\
      --server.port 8507 --server.fileWatcherType none
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PACK_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PACK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

COMPARE_JSON = PACK_ROOT / "results" / "bang_vs_mpc_summary.json"
BANG_CSV = PACK_ROOT / "results" / "baseline_bang_bang_dynamics.csv"
MPC_CSV = PACK_ROOT / "results" / "mpc_optimal_dynamics.csv"
FIG_TEACHING = PACK_ROOT / "paper_figures" / "fig_bang_vs_mpc_day.png"
HARNESS_CSV = PACK_ROOT / "results" / "stress_harness.csv"
FIG_RADAR = PACK_ROOT / "paper_figures" / "fig_harness_radar.png"
LOG_DIR = PACK_ROOT / "logs"

try:
    from baseline_thermostat.competitors import run_controller_week
    from comfort_tariffs.fanger_pmv import discomfort_degree_hours
    from comfort_tariffs.tariff_calculator import bill_from_power_W
    from control_mpc.arrival_predictor import build_occupancy_scenario
    from control_mpc.mpc_solver import simulate_mpc_week
    from eval_harness.run_logger import new_session
except ImportError as exc:
    st.error(f"Cannot import AeroTherm modules: {exc}")
    st.stop()


CONTROLLER_LABELS: Dict[str, str] = {
    "programmable": "传统可编程双位恒温器 (Programmable)",
    "remote_app": "一代远程 App 控温 (Remote App)",
    "nest_like": "二代学习型 Nest 拟合 (Nest-like)",
    "ecobee_like": "多传感器 Ecobee 拟合 (Ecobee-like)",
    "aerotherm_mpc": "AeroTherm-MPC（本系统）",
}
LABEL_TO_KEY = {v: k for k, v in CONTROLLER_LABELS.items()}

SCENARIO_LABELS: Dict[str, str] = {
    "nominal": "夏热基准周切片 (Nominal)",
    "heatwave": "热浪冲击 +4°C (Heatwave)",
    "tariff_surge": "电价暴涨 ×1.8 (Tariff Surge)",
}
SCENARIO_TO_KEY = {v: k for k, v in SCENARIO_LABELS.items()}


# ---------------------------------------------------------------------------
# Shared loaders / legacy helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_weather_legacy() -> pd.DataFrame:
    """Load TMY3-like summer weather CSV (legacy week track)."""
    path = PACK_ROOT / "data" / "raw" / "weather_tmy3_summer.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_locked_teaching() -> Dict[str, Any]:
    """
    Load locked Teaching-track artifacts.

    Returns empty dicts/frames if a file is missing (UI shows a warning).
    """
    out: Dict[str, Any] = {
        "compare": {},
        "bang": pd.DataFrame(),
        "mpc": pd.DataFrame(),
    }
    if COMPARE_JSON.is_file():
        out["compare"] = json.loads(COMPARE_JSON.read_text(encoding="utf-8"))
    if BANG_CSV.is_file():
        out["bang"] = pd.read_csv(BANG_CSV)
    if MPC_CSV.is_file():
        out["mpc"] = pd.read_csv(MPC_CSV)
    return out


@st.cache_data(show_spinner=False)
def load_harness() -> pd.DataFrame:
    """Load stress harness CSV if present."""
    if not HARNESS_CSV.is_file():
        return pd.DataFrame()
    return pd.read_csv(HARNESS_CSV)


def prepare_weather(
    df: pd.DataFrame,
    hours: int,
    scenario_key: str,
) -> Tuple[List[Dict[str, Any]], float]:
    """Slice weather and apply scenario shocks; return rows + surge multiplier."""
    rows = df.head(hours).to_dict(orient="records")
    surge = 1.0
    if scenario_key == "heatwave":
        for r in rows:
            r["T_out_C"] = float(r["T_out_C"]) + 4.0
    elif scenario_key == "tariff_surge":
        surge = 1.8
    return rows, surge


def run_once(
    controller_key: str,
    weather_rows: List[Dict[str, Any]],
    *,
    seed: int,
    surge: float,
    mpc_horizon: int,
) -> Dict[str, Any]:
    """Simulate one legacy controller; return trajectory DataFrame + KPI dict."""
    n = len(weather_rows)
    scenario = build_occupancy_scenario(n, seed=seed)
    occ_mean = scenario["occupancy"].mean(axis=1).tolist()

    if controller_key == "aerotherm_mpc":
        out = simulate_mpc_week(
            weather_rows,
            scenario,
            horizon=mpc_horizon,
            surge=surge,
        )
        traj = out["trajectory"]
    else:
        out = run_controller_week(controller_key, weather_rows, occ_mean)
        traj = out["trajectory"]

    df = pd.DataFrame(traj)
    bill = bill_from_power_W(
        df["P_elec_W"].to_numpy(),
        df["hour_of_day"].astype(int).to_numpy(),
        surge_multiplier=surge,
    )
    ddH = discomfort_degree_hours(
        df["T_in"].to_numpy(),
        23.0,
        df["occupied"].to_numpy(),
    )
    kpis = {
        "energy_kwh": bill["energy_kwh"],
        "cost_usd": bill["cost_usd"],
        "peak_energy_kwh": bill["peak_energy_kwh"],
        "discomfort_degree_hours": ddH,
        "rh_over60_hours": float((df["RH"] > 60.0).sum()),
        "allergen_over_hours": float(
            ((df["C_pol"] > 35.0) * df["occupied"].clip(lower=0)).sum()
        ),
        "mean_T_in": float(df["T_in"].mean()),
        "mean_RH": float(df["RH"].mean()),
        "peak_C_pol": float(df["C_pol"].max()),
    }
    return {"trajectory": df, "kpis": kpis, "weather": weather_rows}


def plot_legacy_dynamics(
    df: pd.DataFrame,
    weather_rows: List[Dict[str, Any]],
) -> go.Figure:
    """Three-row Plotly: T, RH, pollutant + HVAC power."""
    wdf = pd.DataFrame(weather_rows)
    t = df["hour_index"].to_numpy()
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Temperature (°C)",
            "Relative humidity (%) & HVAC power (W)",
            "Indoor pollutant C_pol (µg/m³)",
        ),
        specs=[[{}], [{"secondary_y": True}], [{}]],
    )
    fig.add_trace(
        go.Scatter(x=t, y=wdf["T_out_C"], name="T_out", line=dict(color="#e67e22")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=df["T_in"], name="T_in", line=dict(color="#2980b9", width=2)),
        row=1,
        col=1,
    )
    for day in range(int(t.max()) // 24 + 1):
        fig.add_vrect(
            x0=day * 24 + 14,
            x1=day * 24 + 19,
            fillcolor="#e74c3c",
            opacity=0.08,
            line_width=0,
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(x=t, y=df["RH"], name="RH", line=dict(color="#16a085")),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_hline(y=60, line_dash="dash", line_color="#c0392b", row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=df["P_HVAC_W"],
            name="P_HVAC",
            line=dict(color="#8e44ad", width=1.5),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=t, y=df["C_pol"], name="C_pol", line=dict(color="#2c3e50")),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="RH %", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="W", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="µg/m³", row=3, col=1)
    fig.update_xaxes(title_text="Hour index", row=3, col=1)
    fig.update_layout(height=720, legend=dict(orientation="h", y=1.08), margin=dict(t=60))
    return fig


def plot_teaching_day(bang: pd.DataFrame, mpc: pd.DataFrame) -> go.Figure:
    """Interactive Bang vs MPC temperature + power for teaching day."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Indoor / outdoor temperature", "Site electric power"),
    )
    t = bang["hour_frac"].to_numpy()
    fig.add_vrect(x0=14, x1=19, fillcolor="#e74c3c", opacity=0.12, line_width=0, row=1, col=1)
    fig.add_vrect(x0=14, x1=19, fillcolor="#e74c3c", opacity=0.12, line_width=0, row=2, col=1)
    fig.add_trace(
        go.Scatter(x=t, y=bang["T_out_C"], name="T_out", line=dict(color="#e67e22")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=bang["T_in_C"], name="Bang T_in", line=dict(color="#7f8c8d")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=mpc["T_in_C"], name="MPC T_in", line=dict(color="#2980b9", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=bang["P_elec_W"] / 1000.0,
            name="Bang kW",
            line=dict(color="#7f8c8d"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=mpc["P_elec_W"] / 1000.0,
            name="MPC kW",
            line=dict(color="#c0392b", width=2),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=2, col=1)
    fig.update_xaxes(title_text="Hour of day", row=2, col=1)
    fig.update_layout(height=560, legend=dict(orientation="h", y=1.12), margin=dict(t=50))
    return fig


def pct_drift(live: float, locked: float) -> Optional[float]:
    """Relative drift (%) of live vs locked; None if locked≈0."""
    if abs(locked) < 1e-12:
        return None
    return float(100.0 * (live - locked) / locked)


def latest_jsonl_path() -> Optional[Path]:
    """Most recently modified run_*.jsonl under logs/."""
    if not LOG_DIR.is_dir():
        return None
    files = sorted(LOG_DIR.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AeroTherm-MPC Sandbox - 2018 HiMCM B",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AeroTherm-MPC · 2018 HiMCM B 动态沙盒")
st.caption(
    "Teaching Day（论文主表）与 Legacy Week（附录周切片）严格分轨。"
    "论文插图只用 paper_figures/ 300 DPI；本页 Plotly 仅供交互探索。"
)

tab_teach, tab_legacy, tab_harness = st.tabs(
    ["Teaching Day（Bang vs MPC）", "Legacy Week（五控制器）", "Stress Harness"]
)


# ---------------------------------------------------------------------------
# Tab A — Teaching Day (read locked results; optional recompute)
# ---------------------------------------------------------------------------

with tab_teach:
    st.subheader("教学黄金日 · Bang-Bang vs 预测矩阵 MPC")
    st.markdown(
        "默认读取锁定文件 `bang_vs_mpc_summary.json` 与两条 dynamics CSV。"
        "TOU v2：谷 $0.12 / 平 $0.22 / **峰 $0.48**（14:00–19:00）。"
    )

    locked = load_locked_teaching()
    cmp = locked["compare"]
    bang_df = locked["bang"]
    mpc_df = locked["mpc"]

    if not cmp or bang_df.empty or mpc_df.empty:
        st.warning(
            "缺少 Teaching 锁定产物。请先运行：\n"
            "`PYTHONPATH=src python -m control_mpc.mpc_controller`"
        )
    else:
        bang_m = cmp["bang"]
        mpc_m = cmp["mpc"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            "电费 Bang → MPC",
            f"${mpc_m['cost_usd']:.3f}",
            delta=f"{-cmp['cost_savings_pct']:.1f}% vs Bang",
            delta_color="inverse",
        )
        c2.metric(
            "峰段电量↓",
            f"{mpc_m['peak_energy_kwh']:.3f} kWh",
            delta=f"{-cmp['peak_energy_savings_pct']:.1f}%",
            delta_color="inverse",
        )
        c3.metric(
            "峰段电费占比",
            f"{100 * mpc_m['peak_cost_share']:.1f}%",
            delta=f"Bang {100 * bang_m['peak_cost_share']:.1f}%",
        )
        c4.metric(
            "压缩机启动",
            f"{mpc_m['short_cycle_starts']}",
            delta=f"{-cmp['short_cycle_reduction']} vs Bang",
            delta_color="inverse",
        )
        c5.metric(
            "预峰 / 峰 ū",
            f"{mpc_m.get('mean_u_prepeak', 0):.2f} / {mpc_m.get('mean_u_peak', 0):.2f}",
        )

        st.plotly_chart(plot_teaching_day(bang_df, mpc_df), use_container_width=True)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("**锁定 KPI 表**（`bang_vs_mpc_summary.json`）")
            table = pd.DataFrame(
                [
                    {
                        "controller": "Bang-Bang",
                        "energy_kWh": bang_m["energy_kwh"],
                        "cost_USD": bang_m["cost_usd"],
                        "peak_kWh": bang_m["peak_energy_kwh"],
                        "peak_share": bang_m["peak_cost_share"],
                        "starts": bang_m["short_cycle_starts"],
                    },
                    {
                        "controller": "AeroTherm-MPC",
                        "energy_kWh": mpc_m["energy_kwh"],
                        "cost_USD": mpc_m["cost_usd"],
                        "peak_kWh": mpc_m["peak_energy_kwh"],
                        "peak_share": mpc_m["peak_cost_share"],
                        "starts": mpc_m["short_cycle_starts"],
                    },
                ]
            )
            st.dataframe(
                table.style.format(
                    {
                        "energy_kWh": "{:.3f}",
                        "cost_USD": "{:.3f}",
                        "peak_kWh": "{:.3f}",
                        "peak_share": "{:.1%}",
                    }
                ),
                use_container_width=True,
            )
        with col_b:
            if FIG_TEACHING.is_file():
                st.markdown("**论文插图** `fig_bang_vs_mpc_day.png`（只读）")
                st.image(str(FIG_TEACHING), use_container_width=True)
            else:
                st.info("尚未生成 fig_bang_vs_mpc_day.png")

        st.markdown("---")
        st.markdown("**可选重算**（写入新 JSONL；默认不覆盖锁定 CSV）")
        recompute = st.checkbox("启用重算控件", value=False, key="teach_recompute")
        if recompute:
            do_run = st.button("重算 Teaching Day（约数秒）", type="primary")
            if do_run:
                from baseline_thermostat.bang_bang import run_bang_bang_day, write_weather_csv
                from control_mpc.mpc_controller import compare_to_bang, simulate_mpc_day

                log = new_session(component="streamlit_teaching", track="teaching")
                with st.spinner("Bang-Bang + rolling MPC…"):
                    write_weather_csv()
                    bang_live = run_bang_bang_day()
                    log.bang_day(bang_live["metrics"])
                    mpc_live = simulate_mpc_day(logger=log)
                    live_cmp = compare_to_bang(mpc_live["metrics"], bang_live["metrics"])
                    log.compare_batch(live_cmp)
                    log.export(paths={"jsonl": str(log.path)}, note="streamlit_live_only")
                st.session_state["teach_live"] = {
                    "bang": bang_live["metrics"],
                    "mpc": mpc_live["metrics"],
                    "cmp": live_cmp,
                    "jsonl": str(log.path),
                }
                st.success(f"重算完成（未写 results/）。JSONL：`{log.path.name}`")

            live = st.session_state.get("teach_live")
            if live:
                st.markdown("相对锁定 CSV 的偏差（正 = 重算更高）")
                drift_rows = []
                for key, label in [
                    ("cost_usd", "cost_usd"),
                    ("peak_energy_kwh", "peak_energy_kwh"),
                    ("short_cycle_starts", "starts"),
                ]:
                    d_bang = pct_drift(live["bang"][key], bang_m[key])
                    d_mpc = pct_drift(live["mpc"][key], mpc_m[key])
                    drift_rows.append(
                        {
                            "metric": label,
                            "Bang drift %": d_bang,
                            "MPC drift %": d_mpc,
                        }
                    )
                st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)
                jl = Path(live["jsonl"])
                if jl.is_file():
                    st.download_button(
                        "下载本次 JSONL",
                        data=jl.read_bytes(),
                        file_name=jl.name,
                        mime="application/x-ndjson",
                    )

        jl_latest = latest_jsonl_path()
        if jl_latest is not None:
            st.caption(f"最近一次日志：`logs/{jl_latest.name}`")
            st.download_button(
                "下载最近 JSONL",
                data=jl_latest.read_bytes(),
                file_name=jl_latest.name,
                mime="application/x-ndjson",
                key="dl_latest_jsonl",
            )


# ---------------------------------------------------------------------------
# Tab B — Legacy Week
# ---------------------------------------------------------------------------

with tab_legacy:
    st.subheader("附录轨道 · 多物理场周切片五系统对比")
    st.caption(
        "TOU legacy：谷 $0.08 / 平 $0.14 / 峰 $0.32。"
        "与 Teaching Day **禁止混表**；Nest −64.6% 仅属本轨。"
    )

    weather_df = load_weather_legacy()
    col_l, col_r = st.columns([1, 3])
    with col_l:
        ctrl_label = st.selectbox(
            "控制器",
            list(CONTROLLER_LABELS.values()),
            index=4,
            key="legacy_ctrl",
        )
        controller_key = LABEL_TO_KEY[ctrl_label]
        scen_label = st.radio(
            "扰动情景",
            list(SCENARIO_LABELS.values()),
            key="legacy_scen",
        )
        scenario_key = SCENARIO_TO_KEY[scen_label]
        hours = st.slider(
            "仿真时长 (小时)",
            min_value=12,
            max_value=min(96, len(weather_df)),
            value=36,
            step=6,
            key="legacy_hours",
        )
        seed = st.number_input(
            "随机种子", min_value=0, max_value=10_000, value=2018, step=1, key="legacy_seed"
        )
        mpc_horizon = st.slider(
            "MPC 时域 H",
            min_value=2,
            max_value=6,
            value=3,
            step=1,
            key="legacy_H",
        )
        run_btn = st.button("运行仿真", type="primary", use_container_width=True, key="legacy_run")
        compare_btn = st.button("一键五系统对比", use_container_width=True, key="legacy_cmp")

    weather_rows, surge = prepare_weather(weather_df, int(hours), scenario_key)

    if run_btn or "legacy_result" not in st.session_state:
        with st.spinner("正在积分 legacy 2R2C + 控制器…"):
            log = new_session(
                component="streamlit_legacy",
                track="legacy",
                controller=controller_key,
                scenario=scenario_key,
                hours=int(hours),
                seed=int(seed),
            )
            result = run_once(
                controller_key,
                weather_rows,
                seed=int(seed),
                surge=surge,
                mpc_horizon=int(mpc_horizon),
            )
            log.emit("run_once_legacy", kpis=result["kpis"])
            st.session_state["legacy_result"] = result
            st.session_state["legacy_controller"] = controller_key
            st.session_state["legacy_scenario"] = scenario_key
            st.session_state["legacy_jsonl"] = str(log.path)

    result = st.session_state["legacy_result"]
    kpis = result["kpis"]
    df = result["trajectory"]

    with col_r:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("用电量 (kWh)", f"{kpis['energy_kwh']:.2f}")
        m2.metric("电费 (USD)", f"${kpis['cost_usd']:.2f}")
        m3.metric("峰段电量 (kWh)", f"{kpis['peak_energy_kwh']:.2f}")
        m4.metric("不舒适度小时", f"{kpis['discomfort_degree_hours']:.1f}")
        m5.metric("RH>60% 小时", f"{kpis['rh_over60_hours']:.0f}")
        st.caption(
            f"控制器=`{st.session_state.get('legacy_controller')}` · "
            f"工况=`{st.session_state.get('legacy_scenario')}` · surge×{surge:.1f}"
        )
        st.plotly_chart(
            plot_legacy_dynamics(df, result["weather"]),
            use_container_width=True,
        )

    if compare_btn:
        with st.spinner("五系统同台对比中…"):
            rows = []
            for key, label in CONTROLLER_LABELS.items():
                r = run_once(
                    key,
                    weather_rows,
                    seed=int(seed),
                    surge=surge,
                    mpc_horizon=int(mpc_horizon),
                )
                rows.append({"controller": label, **r["kpis"]})
            cmp_df = pd.DataFrame(rows)
            st.subheader("五系统 KPI 对比（当前工况 · Legacy）")
            st.dataframe(cmp_df, use_container_width=True)
            fig_bar = go.Figure(
                data=[
                    go.Bar(name="Energy kWh", x=cmp_df["controller"], y=cmp_df["energy_kwh"]),
                    go.Bar(name="Cost USD", x=cmp_df["controller"], y=cmp_df["cost_usd"]),
                ]
            )
            fig_bar.update_layout(
                barmode="group",
                title="能耗与电费对比（Legacy）",
                height=420,
                xaxis_tickangle=-20,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("轨迹预览"):
        st.dataframe(df.head(48), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab C — Stress Harness (read-only)
# ---------------------------------------------------------------------------

with tab_harness:
    st.subheader("压力工况雷达（只读）")
    harness = load_harness()
    if harness.empty:
        st.warning("缺少 `results/stress_harness.csv`（Legacy pipeline 产物）。")
    else:
        st.dataframe(harness, use_container_width=True)
        # Simple radar for nominal_mpc vs nominal_nest if present.
        metrics = [
            "energy_kwh",
            "cost_usd",
            "discomfort_degree_hours",
            "rh_over60_hours",
            "mean_abs_pmv",
        ]
        cases = ["nominal_nest", "nominal_mpc", "heatwave_mpc", "tariff_surge_mpc"]
        present = [c for c in cases if c in set(harness["case"].astype(str))]
        if present and all(m in harness.columns for m in metrics):
            fig = go.Figure()
            for case in present:
                row = harness.loc[harness["case"] == case].iloc[0]
                vals = [float(row[m]) for m in metrics]
                # Normalize by column max for radar shape only.
                maxima = [float(harness[m].max()) or 1.0 for m in metrics]
                norm = [v / mx for v, mx in zip(vals, maxima)]
                fig.add_trace(
                    go.Scatterpolar(
                        r=norm + [norm[0]],
                        theta=metrics + [metrics[0]],
                        name=case,
                        fill="toself",
                    )
                )
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Normalized harness KPIs (lower better; UI only)",
                height=480,
            )
            st.plotly_chart(fig, use_container_width=True)
        if FIG_RADAR.is_file():
            st.markdown("**论文雷达图** `fig_harness_radar.png`")
            st.image(str(FIG_RADAR), use_container_width=True)

st.markdown(
    "证据：`results/bang_vs_mpc_summary.json` · "
    "`results/competitor_benchmarks.csv` · `experiments/protocols.md` · `logs/`"
)

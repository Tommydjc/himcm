"""SummerJobMatch: Streamlit navigator for HiMCM 2020 A Requirement 5.

Run from the pack root:

    streamlit run src/web_app/app.py

Sliders use the contest 1-10 labels (yield / career / strain tolerance).
Inference maps them into extracted Thompson space and applies the Stage-6
linear Softmax (same W as ``dual_recommender.fit_linear_softmax``).
"""

from __future__ import annotations

import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.web_app.service import NavigatorBundle, rank_jobs, radar_series, load_navigator_bundle

PAGE_TITLE = "🎓 SummerJobMatch: AI-Powered High School Job Navigator"


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1100px; padding-top: 1.2rem; }
        div[data-testid="stMetric"] { background: #f7f9fc; border-radius: 12px; padding: 0.4rem 0.8rem; }
        .sj-card { border-radius: 16px; padding: 1rem 1.15rem; margin-bottom: 0.8rem; }
        .sj-gold { background: linear-gradient(135deg, #fff8e7 0%, #ffe8a3 100%); border: 1px solid #f0c14b; }
        .sj-silver { background: linear-gradient(135deg, #f4f6f8 0%, #dfe6ed 100%); border: 1px solid #aeb6bf; }
        .sj-warn { background: linear-gradient(135deg, #fdecea 0%, #f5b7b1 100%); border: 1px solid #c0392b; }
        .sj-kicker { font-size: 0.8rem; letter-spacing: 0.04em; color: #5d6d7e; text-transform: uppercase; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Fitting the 3-factor Softmax on 50 students…")
def _bundle() -> NavigatorBundle:
    """Process-wide cache: one full-sample Softmax + job table."""
    return load_navigator_bundle()


def _init_state() -> None:
    st.session_state.setdefault("s1", 5)
    st.session_state.setdefault("s2", 5)
    st.session_state.setdefault("s3", 5)


def _preset_pragmatist() -> None:
    st.session_state.s1 = 9
    st.session_state.s3 = 8


def _preset_career() -> None:
    st.session_state.s2 = 10


def _preset_leisure() -> None:
    st.session_state.s3 = 1


def _radar_figure(
    bundle: NavigatorBundle,
    s1: int,
    s2: int,
    s3: int,
    best_factors: np.ndarray,
    best_title: str,
) -> plt.Figure:
    labels = [
        "Financial Yield\n(UI) / extracted F1",
        "Career & Skill\n(UI) / extracted F2",
        "Strain tolerance\n(UI) / extracted F3",
    ]
    user, job = radar_series(bundle, float(s1), float(s2), float(s3), best_factors)
    angles = np.linspace(0.0, 2.0 * np.pi, 3, endpoint=False)
    user_c = np.concatenate([user, user[:1]])
    job_c = np.concatenate([job, job[:1]])
    ang_c = np.concatenate([angles, angles[:1]])
    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw={"polar": True})
    ax.plot(ang_c, user_c, color="#1f4e79", linewidth=2.0, label="You (sliders)")
    ax.fill(ang_c, user_c, color="#1f4e79", alpha=0.18)
    ax.plot(ang_c, job_c, color="#c0392b", linewidth=2.0, label=f"Top job: {best_title}")
    ax.fill(ang_c, job_c, color="#c0392b", alpha=0.12)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(1.0, 10.0)
    ax.set_yticks([1, 4, 7, 10])
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8)
    ax.set_title("Profile overlay (1–10 radar)", pad=16)
    fig.tight_layout()
    return fig


def _card_html(kind: str, kicker: str, card) -> str:
    css = {"gold": "sj-gold", "silver": "sj-silver", "warn": "sj-warn"}[kind]
    return (
        f'<div class="sj-card {css}">'
        f'<div class="sj-kicker">{kicker}</div>'
        f"<h3 style='margin:0.2rem 0 0.4rem 0;'>{card.title_en}</h3>"
        f"<p><b>Match:</b> {card.match_pct:.1f}% &nbsp;·&nbsp; "
        f"<b>Pay (from ad):</b> {card.wage_range}</p>"
        f"<p>{card.reason}</p>"
        f"</div>"
    )


def main() -> None:
    """Streamlit entry: presets, three sliders, Softmax cards, radar."""
    st.set_page_config(page_title="SummerJobMatch", page_icon="🎓", layout="wide")
    _inject_css()
    _init_state()
    bundle = _bundle()

    st.title(PAGE_TITLE)
    st.caption(
        "Requirement 5 navigator. Sliders speak contest language "
        "(yield / career / strain). The model scores **extracted** "
        f"F1={bundle.extracted_names[0]}, F2={bundle.extracted_names[1]}, "
        f"F3={bundle.extracted_names[2]}. Slider 1–10 is min–max scaled onto "
        "the 50 students' Thompson scores, then passed to the linear Softmax."
    )

    st.subheader("Quick Persona Preset")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.button("💰 我极度缺钱 (The Pragmatist)", on_click=_preset_pragmatist, use_container_width=True)
        st.caption("Sets factor 1 → 9, factor 3 → 8")
    with c2:
        st.button("📚 我要充实申请文书 (The Career-Builder)", on_click=_preset_career, use_container_width=True)
        st.caption("Sets factor 2 → 10")
    with c3:
        st.button("☕ 我想轻松度过夏天 (The Leisure-Seeker)", on_click=_preset_leisure, use_container_width=True)
        st.caption("Sets factor 3 → 1")

    st.subheader("Your 3-factor profile")
    s1 = st.slider("💵 即时经济回报期望 (Financial Yield)", 1, 10, key="s1")
    s2 = st.slider("🚀 人力资本与简历积累 (Career & Skill Growth)", 1, 10, key="s2")
    s3 = st.slider("🧘 身心负荷耐受度 (Physical Strain Tolerance)", 1, 10, key="s3")

    ranked = rank_jobs(bundle, float(s1), float(s2), float(s3))
    best, second, avoid = ranked[0], ranked[1], ranked[-1]

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown(_card_html("gold", "🥇 Best match", best), unsafe_allow_html=True)
        st.markdown(_card_html("silver", "🥈 Runner-up", second), unsafe_allow_html=True)
        st.markdown(_card_html("warn", "⚠️ Strong mismatch — think twice", avoid), unsafe_allow_html=True)
        st.caption(
            f"Softmax P(best)={best.proba:.3f}, P(runner-up)={second.proba:.3f}, "
            f"P(avoid)={avoid.proba:.3f}. LOOCV Top-1 of this model on 50 labeled "
            "students was 0.20 (below the majority baseline); treat ranks as a "
            "preference overlay, not a guaranteed placement."
        )
    with right:
        fig = _radar_figure(bundle, s1, s2, s3, best.factors, best.title_en)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.expander("All eight jobs (Softmax order)"):
        rows = [
            {
                "rank": i,
                "job": card.title_en,
                "match_%": round(card.match_pct, 2),
                "pay": card.wage_range,
            }
            for i, card in enumerate(ranked, start=1)
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()

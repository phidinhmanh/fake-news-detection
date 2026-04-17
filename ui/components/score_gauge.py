"""
score_gauge.py — Plotly Gauge Chart cho Fake News Score
========================================================
Hiển thị score 0-100% dạng speedometer gauge.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go


def render_score_gauge(score: int, label: str, confidence: float) -> None:
    """Render score gauge chart.

    Args:
        score: Fake news score (0-100).
        label: "Real" / "Fake" / "Suspicious".
        confidence: Model confidence (0-1).
    """
    # Color based on score
    if score <= 30:
        bar_color = "#2ecc71"  # Green — Real
        bg_gradient = ["#2ecc71", "#f1c40f", "#e74c3c"]
    elif score <= 70:
        bar_color = "#f39c12"  # Yellow — Suspicious
        bg_gradient = ["#2ecc71", "#f1c40f", "#e74c3c"]
    else:
        bar_color = "#e74c3c"  # Red — Fake
        bg_gradient = ["#2ecc71", "#f1c40f", "#e74c3c"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={"suffix": "%", "font": {"size": 48, "color": bar_color}},
        title={
            "text": f"<b>{label}</b><br><span style='font-size:14px;color:gray'>Confidence: {confidence:.0%}</span>",
            "font": {"size": 20},
        },
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkgray"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 30], "color": "#d5f5e3"},
                {"range": [30, 70], "color": "#fdebd0"},
                {"range": [70, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 4},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )

    st.plotly_chart(fig, use_container_width=True)


def render_label_badge(label: str, score: int) -> None:
    """Render label badge dạng HTML.

    Args:
        label: "Real" / "Fake" / "Suspicious".
        score: Fake news score (0-100).
    """
    colors = {
        "Real": ("#2ecc71", "#d5f5e3", "✅"),
        "Fake": ("#e74c3c", "#fadbd8", "❌"),
        "Suspicious": ("#f39c12", "#fdebd0", "⚠️"),
    }

    text_color, bg_color, emoji = colors.get(label, ("#95a5a6", "#ecf0f1", "❓"))

    badge_html = f"""
    <div style="
        background: {bg_color};
        border: 2px solid {text_color};
        border-radius: 12px;
        padding: 16px 24px;
        text-align: center;
        margin: 8px 0;
    ">
        <span style="font-size: 36px;">{emoji}</span>
        <h2 style="color: {text_color}; margin: 8px 0 4px 0;">{label.upper()}</h2>
        <p style="color: {text_color}; font-size: 18px; margin: 0;">
            Fake Score: <b>{score}%</b>
        </p>
    </div>
    """

    st.markdown(badge_html, unsafe_allow_html=True)

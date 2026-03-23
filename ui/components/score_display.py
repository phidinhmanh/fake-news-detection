"""
score_display.py — Prediction Score Components
=================================================
Người C phát triển (Tuần 3-4).

TODO:
    - [ ] st.progress bar with custom colors
    - [ ] Confidence gauge (circular)
    - [ ] Animated transitions
"""

from __future__ import annotations

import streamlit as st


def render_score_bar(confidence: float, label: str) -> None:
    """Render prediction score bar with color coding.

    Args:
        confidence: Float 0.0-1.0.
        label: 'fake' or 'real'.
    """
    # TODO: Implement custom styled score bar
    color = "red" if label == "fake" else "green"
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.0%} ({label})")


def render_confidence_gauge(confidence: float) -> None:
    """Render circular confidence gauge using Plotly.

    Args:
        confidence: Float 0.0-1.0.
    """
    # TODO: Implement with plotly.graph_objects.Indicator
    pass

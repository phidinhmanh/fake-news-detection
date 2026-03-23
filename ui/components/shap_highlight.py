"""
shap_highlight.py — SHAP Token Highlight Component
=====================================================
Người C phát triển (Tuần 3-4).

TODO:
    - [ ] Render HTML với color gradient (đỏ = high weight, xanh = low)
    - [ ] st.markdown(html, unsafe_allow_html=True)
    - [ ] CSS styles cho highlight
    - [ ] Tooltip hiển thị weight value
"""

from __future__ import annotations

import streamlit as st


def shap_display(shap_tokens: list[tuple[str, float]]) -> None:
    """Render SHAP token highlights với color gradient.

    Args:
        shap_tokens: List of (token, weight) tuples.
            Positive weight = contributes to 'fake'.
            Negative weight = contributes to 'real'.
    """
    if not shap_tokens:
        st.info("Không có dữ liệu SHAP tokens.")
        return

    # TODO: Implement HTML rendering với color gradient
    # Placeholder implementation
    for token, weight in sorted(shap_tokens, key=lambda x: abs(x[1]), reverse=True):
        if weight > 0:
            st.markdown(f"🔴 `{token}` → **{weight:.2f}** (fake indicator)")
        else:
            st.markdown(f"🟢 `{token}` → **{weight:.2f}** (real indicator)")

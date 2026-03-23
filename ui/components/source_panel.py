"""
source_panel.py — Source Credibility Panel
============================================
Người C phát triển (Tuần 5-6).

TODO:
    - [ ] source_credibility_panel()
    - [ ] Link to source website
    - [ ] Show historical credibility score
    - [ ] Indicators for bias/reliability
"""

from __future__ import annotations

import streamlit as st


def render_source_panel(source_score: float | None = None) -> None:
    """Render a panel with source credibility score.

    Args:
        source_score: Credibility score 0.0-1.0 or None.
    """
    if source_score is None:
        st.warning("⚠️ Không có thông tin về nguồn tin này.")
        return

    st.subheader("📊 Độ tin cậy nguồn")
    st.progress(source_score)
    st.caption(f"Score: {source_score:.2f}")

    if source_score < 0.3:
        st.error("❌ Nguồn này có lịch sử đăng tin không tin cậy.")
    elif source_score < 0.7:
        st.warning("⚠️ Nguồn này cần được kiểm chứng thêm.")
    else:
        st.success("✅ Nguồn này có lịch sử đăng tin tin cậy.")

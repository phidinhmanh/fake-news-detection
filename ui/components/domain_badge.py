"""
domain_badge.py — Domain Badge Component
==========================================
Người C phát triển (Tuần 5-6).

TODO:
    - [ ] domain_badge() component with 4 colors
    - [ ] politics = red, health = blue, finance = green, social = yellow
    - [ ] icons matching each domain
"""

from __future__ import annotations

import streamlit as st


def render_domain_badge(domain: str) -> None:
    """Render a badge with color and icon for the specific domain.

    Args:
        domain: 'politics', 'health', 'finance', 'social'.
    """
    domain_map = {
        "politics": ("⚖️", "politics", "red"),
        "health": ("🏥", "health", "blue"),
        "finance": ("💰", "finance", "green"),
        "social": ("🤝", "social", "orange"),
    }

    icon, label, color = domain_map.get(domain, ("❓", "unknown", "gray"))

    # TODO: Implement fancy badge styling with HTML/CSS
    st.info(f"{icon} Lĩnh vực: **{label.upper()}**")

"""
attention_heatmap.py — Attention Heatmap cho Token Highlighting
================================================================
Highlight từ quan trọng trong bài viết dựa trên attention weights.
"""

from __future__ import annotations

import re

import streamlit as st


def render_attention_heatmap(
    text: str,
    important_words: list[tuple[str, float]] | None = None,
    risk_factors: list[str] | None = None,
) -> None:
    """Render attention heatmap bằng HTML highlight.

    Args:
        text: Bài viết gốc.
        important_words: List of (word, weight) tuples. Weight 0-1.
        risk_factors: List of risk factor strings.
    """
    if important_words:
        _render_highlighted_text(text, important_words)
    else:
        _render_simple_highlight(text, risk_factors or [])


def _render_highlighted_text(text: str, word_weights: list[tuple[str, float]]) -> None:
    """Highlight từ theo weight (đỏ = cao, xanh = thấp).

    Args:
        text: Original text.
        word_weights: List of (word, weight) where weight is 0-1.
    """
    # Build highlight map
    weight_map: dict[str, float] = {}
    for word, weight in word_weights:
        weight_map[word.lower()] = max(weight_map.get(word.lower(), 0), weight)

    words = text.split()
    html_parts = []

    for word in words:
        clean = re.sub(r"[^\w]", "", word.lower())
        weight = weight_map.get(clean, 0.0)

        if weight > 0.1:
            # Interpolate color: green(low) → yellow(mid) → red(high)
            r = int(min(255, weight * 2 * 255))
            g = int(min(255, (1 - weight) * 2 * 255))
            opacity = 0.2 + weight * 0.6

            html_parts.append(
                f'<span style="background-color:rgba({r},{g},0,{opacity:.2f});'
                f'padding:2px 4px;border-radius:3px;font-weight:{"bold" if weight > 0.5 else "normal"}"'
                f' title="Weight: {weight:.2f}">{word}</span>'
            )
        else:
            html_parts.append(word)

    highlighted_html = " ".join(html_parts)

    st.markdown("### 🔬 Phân tích Attention")
    st.markdown(
        f'<div style="line-height:2;padding:16px;background:#fafafa;'
        f'border-radius:8px;border:1px solid #eee;">{highlighted_html}</div>',
        unsafe_allow_html=True,
    )

    # Legend
    st.markdown(
        '<div style="margin-top:8px;font-size:12px;color:#666;">'
        '<span style="background:rgba(0,255,0,0.3);padding:2px 6px;border-radius:3px;">Thấp</span> '
        '<span style="background:rgba(255,255,0,0.4);padding:2px 6px;border-radius:3px;">Trung bình</span> '
        '<span style="background:rgba(255,0,0,0.5);padding:2px 6px;border-radius:3px;">Cao</span> '
        "— Mức độ đáng ngờ của từ</div>",
        unsafe_allow_html=True,
    )


def _render_simple_highlight(text: str, risk_factors: list[str]) -> None:
    """Highlight đơn giản dựa trên risk factors.

    Args:
        text: Original text.
        risk_factors: List of suspicious patterns.
    """
    # Keyword patterns commonly found in fake news (Vietnamese)
    suspicious_patterns = [
        "sốc", "kinh hoàng", "khẩn cấp", "cực kỳ", "bí mật", "rò rỉ",
        "che giấu", "sự thật", "lừa đảo", "cảnh báo", "báo động", "100%",
        "chia sẻ ngay", "chấn động", "hot", "breaking", "exclusive",
        "!!!", "???", "KHẨN", "SỐC", "NÓNG",
    ]

    html_text = text
    highlighted_count = 0

    for pattern in suspicious_patterns:
        if pattern.lower() in text.lower():
            # Case-insensitive replace with highlight
            compiled = re.compile(re.escape(pattern), re.IGNORECASE)
            html_text = compiled.sub(
                f'<mark style="background:#fadbd8;padding:2px 4px;border-radius:3px;'
                f'font-weight:bold;" title="Từ ngữ đáng ngờ">{pattern}</mark>',
                html_text,
            )
            highlighted_count += 1

    if highlighted_count > 0:
        st.markdown("### 🔬 Từ ngữ đáng ngờ")
        st.markdown(
            f'<div style="line-height:2;padding:16px;background:#fafafa;'
            f'border-radius:8px;border:1px solid #eee;">{html_text}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Phát hiện {highlighted_count} từ/cụm từ đáng chú ý")

    # Show risk factors
    if risk_factors:
        st.markdown("### ⚠️ Yếu tố rủi ro")
        for rf in risk_factors:
            st.markdown(f"- 🔸 {rf}")

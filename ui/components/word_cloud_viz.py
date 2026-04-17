"""
word_cloud_viz.py — Word Cloud Visualization
==============================================
Tạo word cloud từ bài viết, highlight từ quan trọng.
"""

from __future__ import annotations

import logging
from io import BytesIO

import streamlit as st

logger = logging.getLogger(__name__)


def render_word_cloud(text: str, title: str = "Word Cloud") -> None:
    """Render word cloud từ text.

    Args:
        text: Input text.
        title: Chart title.
    """
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        # Vietnamese stopwords (extend as needed)
        stopwords = {
            "và", "của", "là", "có", "được", "cho", "các", "này",
            "đã", "với", "trong", "không", "một", "những", "để",
            "thì", "hay", "nhưng", "từ", "đến", "cũng", "theo",
            "về", "tại", "như", "trên", "bị", "ra", "đó", "khi",
            "the", "is", "and", "of", "to", "in", "a", "that",
        }

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=100,
            colormap="RdYlGn_r",  # Red for frequent (suspicious), Green for rare
            stopwords=stopwords,
            font_path=None,  # Use default; can set Vietnamese font path
            contour_width=2,
            contour_color="steelblue",
            prefer_horizontal=0.7,
        )

        wc.generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        plt.tight_layout()

        # Convert to bytes for Streamlit
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        st.image(buf, use_container_width=True)

    except ImportError:
        st.warning("📦 Cần cài đặt `wordcloud`: `pip install wordcloud`")
        # Fallback: show frequent words as tags
        _render_word_tags(text)


def _render_word_tags(text: str, max_words: int = 20) -> None:
    """Fallback: hiển thị từ khóa dạng tags khi wordcloud unavailable."""
    from collections import Counter

    words = text.lower().split()
    stopwords = {"và", "của", "là", "có", "được", "cho", "các", "này", "the", "is", "and", "of"}
    words = [w for w in words if len(w) > 2 and w not in stopwords]

    counter = Counter(words)
    top_words = counter.most_common(max_words)

    if not top_words:
        return

    tags_html = " ".join(
        f'<span style="display:inline-block;background:#ecf0f1;border-radius:4px;'
        f'padding:4px 8px;margin:2px;font-size:{min(12 + count * 2, 24)}px;">'
        f'{word} ({count})</span>'
        for word, count in top_words
    )

    st.markdown(f"**📊 Từ khóa xuất hiện nhiều:**", unsafe_allow_html=True)
    st.markdown(tags_html, unsafe_allow_html=True)

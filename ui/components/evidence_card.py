"""
evidence_card.py — Collapsible Evidence Cards
===============================================
Hiển thị evidence citations dạng expandable cards.
"""

from __future__ import annotations

import streamlit as st


def render_evidence_cards(evidence_citations: list[dict], evidences: list[dict] | None = None) -> None:
    """Render evidence citations dạng collapsible cards.

    Args:
        evidence_citations: List of dicts from Agent 3 (claim, verdict, supporting_evidence).
        evidences: Raw evidence list from Agent 2 (optional, for additional context).
    """
    if not evidence_citations and not evidences:
        st.info("📭 Không tìm thấy bằng chứng liên quan.")
        return

    st.markdown("### 📋 Bằng chứng & Trích dẫn")

    # Evidence Citations (from Agent 3)
    if evidence_citations:
        for i, citation in enumerate(evidence_citations):
            claim = citation.get("claim", "N/A")
            verdict = citation.get("verdict", "UNVERIFIED")
            evidence_text = citation.get("supporting_evidence", "")

            # Verdict styling
            verdict_config = {
                "SUPPORTED": ("✅", "#2ecc71", "#d5f5e3"),
                "REFUTED": ("❌", "#e74c3c", "#fadbd8"),
                "MIXED": ("🟡", "#f39c12", "#fdebd0"),
                "UNVERIFIED": ("❓", "#95a5a6", "#ecf0f1"),
            }
            emoji, color, bg = verdict_config.get(
                verdict.upper(), ("❓", "#95a5a6", "#ecf0f1")
            )

            with st.expander(f"{emoji} Claim {i+1}: {claim[:80]}...", expanded=(i == 0)):
                st.markdown(
                    f'<div style="background:{bg};padding:12px;border-radius:8px;'
                    f'border-left:4px solid {color};margin-bottom:8px;">'
                    f'<b style="color:{color};">Verdict: {verdict}</b></div>',
                    unsafe_allow_html=True,
                )

                if evidence_text:
                    st.markdown(f"**Evidence:** {evidence_text}")
                else:
                    st.caption("Không có evidence cụ thể.")

    # Raw evidence (from Agent 2) — additional context
    if evidences:
        with st.expander("📚 Tất cả evidence tìm được", expanded=False):
            for i, ev in enumerate(evidences):
                source = ev.get("source", "unknown")
                text = ev.get("text", "")
                score = ev.get("relevance_score", 0)
                stance = ev.get("stance", "neutral")

                stance_emoji = {"support": "👍", "refute": "👎", "neutral": "➖"}.get(stance, "➖")

                st.markdown(
                    f"**{i+1}.** {stance_emoji} `{stance}` | Source: `{source}` | "
                    f"Relevance: {score:.2f}"
                )
                st.markdown(f"> {text[:300]}{'...' if len(text) > 300 else ''}")
                st.divider()


def render_sources_summary(sources_used: list[str]) -> None:
    """Render summary of sources used.

    Args:
        sources_used: List of source names.
    """
    if not sources_used:
        return

    st.markdown("### 📎 Nguồn sử dụng")
    cols = st.columns(len(sources_used))
    for i, source in enumerate(sources_used):
        with cols[i]:
            source_icons = {
                "vifactcheck_kb": "📚",
                "wikipedia_vn": "📖",
                "google_search": "🔍",
            }
            icon = source_icons.get(source, "📄")
            st.markdown(
                f'<div style="text-align:center;padding:8px;background:#f8f9fa;'
                f'border-radius:8px;border:1px solid #dee2e6;">'
                f'{icon}<br><small>{source}</small></div>',
                unsafe_allow_html=True,
            )

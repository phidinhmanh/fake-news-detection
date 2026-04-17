"""
app.py — Streamlit Web Demo: Vietnamese Fake News Detector
=============================================================
Upload text hoặc dán link → phân tích → hiển thị score, label, giải thích.

Chạy:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="🔍 Vietnamese Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    div[data-testid="stVerticalBlock"] > div {
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt")

    analysis_mode = st.selectbox(
        "🔧 Chế độ phân tích",
        options=["AI Agent Pipeline", "PhoBERT Baseline", "Cả hai (so sánh)"],
        index=0,
        help="Chọn model để phân tích bài viết",
    )

    use_wikipedia = st.checkbox("📖 Search Wikipedia VN", value=True, help="Bật search Wikipedia cho evidence")
    mock_mode = st.checkbox("🧪 Mock Mode (không cần API)", value=False, help="Dùng mock data cho testing")

    st.divider()

    # History
    st.markdown("### 📜 Lịch sử phân tích")
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            emoji = {"Real": "🟢", "Fake": "🔴", "Suspicious": "🟡"}.get(h["label"], "⚪")
            st.caption(f"{emoji} {h['label']} ({h['score']}%) — {h['text'][:30]}...")
    else:
        st.caption("Chưa có lịch sử phân tích.")

    st.divider()
    st.markdown(
        """
        ### 📄 About
        **Vietnamese Fake News Detector**
        - ViFactCheck (AAAI 2025)
        - PhoBERT + LLM Agent Pipeline
        - Explainable AI output

        *UET — 2026*
        """
    )


# ── Main Content ───────────────────────────────────────
st.markdown(
    '<div class="main-header">'
    '<h1>🔍 Vietnamese Fake News Detector</h1>'
    '<p style="color:#666;font-size:16px;">Phát hiện tin giả tiếng Việt bằng AI — '
    'ViFactCheck + PhoBERT + LLM Agent</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Input Section ──────────────────────────────────────
tab_text, tab_url = st.tabs(["📝 Dán nội dung", "🔗 Dán URL"])

article_text = ""

with tab_text:
    article_text = st.text_area(
        "Nội dung bài viết",
        height=200,
        max_chars=8000,
        placeholder="Dán nội dung bài viết cần kiểm tra tại đây...",
        key="text_input",
    )

with tab_url:
    url_input = st.text_input(
        "URL bài viết",
        placeholder="https://vnexpress.net/...",
        key="url_input",
    )
    if url_input and not article_text:
        with st.spinner("🔄 Đang tải nội dung từ URL..."):
            try:
                from sequential_adversarial.input_processor import InputProcessor
                processor = InputProcessor()
                result = processor.process(url_input)
                article_text = result.get("raw_text", "")
                if article_text:
                    st.success(f"✅ Đã tải {len(article_text)} ký tự")
                    st.text_area("Nội dung đã tải:", article_text[:2000], height=150, disabled=True)
                else:
                    st.warning("⚠️ Không thể trích xuất nội dung từ URL.")
            except Exception as exc:
                st.error(f"❌ Lỗi: {exc}")


# ── Analyze Button ─────────────────────────────────────
if st.button("🔎 Phân tích bài viết", type="primary", use_container_width=True, disabled=(not article_text)):
    if not article_text or len(article_text.strip()) < 20:
        st.warning("⚠️ Vui lòng nhập nội dung bài viết (ít nhất 20 ký tự).")
    else:
        st.divider()

        # ── Run Analysis ──────────────────────────────────
        with st.spinner("🔄 Đang phân tích... (có thể mất 10-30 giây)"):
            agent_result = None
            baseline_result = None

            # Agent Pipeline
            if analysis_mode in ("AI Agent Pipeline", "Cả hai (so sánh)"):
                try:
                    from agents.agent_pipeline import AgentPipeline

                    pipeline = AgentPipeline(
                        mock=mock_mode,
                        use_wikipedia=use_wikipedia,
                    )
                    agent_result = pipeline.analyze(article_text)
                except Exception as exc:
                    st.error(f"❌ Agent Pipeline Error: {exc}")
                    logger.error(f"Agent error: {exc}", exc_info=True)

            # PhoBERT Baseline
            if analysis_mode in ("PhoBERT Baseline", "Cả hai (so sánh)"):
                try:
                    # Try loading trained PhoBERT model
                    from config import MODELS_ARTIFACTS_DIR, PHOBERT_MODEL_NAME
                    import torch
                    from model.phobert_baseline import PhoBERTBaseline

                    model_path = MODELS_ARTIFACTS_DIR / "phobert_baseline_best.pt"
                    if model_path.exists():
                        model = PhoBERTBaseline(model_name=PHOBERT_MODEL_NAME)
                        model.load_model(str(model_path))
                        model.eval()

                        tokenizer = PhoBERTBaseline.get_tokenizer(PHOBERT_MODEL_NAME)
                        encoding = tokenizer(
                            article_text[:512],
                            max_length=256,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                        )

                        probs = model.predict_proba(encoding["input_ids"], encoding["attention_mask"])
                        fake_prob = float(probs[0][1]) * 100
                        baseline_result = {
                            "score": int(fake_prob),
                            "label": "Fake" if fake_prob >= 50 else "Real",
                            "confidence": float(probs.max()),
                        }
                    else:
                        st.info("ℹ️ PhoBERT model chưa được train. Dùng Agent Pipeline.")
                except Exception as exc:
                    st.warning(f"⚠️ PhoBERT không khả dụng: {exc}")

        # ── Display Results ───────────────────────────────
        if agent_result:
            st.markdown("## 📊 Kết quả phân tích")

            # Score + Label row
            col1, col2 = st.columns([2, 1])

            with col1:
                from ui.components.score_gauge import render_score_gauge
                render_score_gauge(
                    score=agent_result.fake_score,
                    label=agent_result.label,
                    confidence=agent_result.confidence,
                )

            with col2:
                from ui.components.score_gauge import render_label_badge
                render_label_badge(agent_result.label, agent_result.fake_score)

                # Processing time
                st.metric("⏱️ Thời gian", f"{agent_result.processing_time_seconds:.1f}s")

                # Agent status
                st.markdown("**🤖 Agent Status:**")
                for agent, status in agent_result.agents_status.items():
                    emoji = "✅" if status == "success" else "❌"
                    st.caption(f"{emoji} {agent}")

            st.divider()

            # Explanation
            if agent_result.explanation:
                st.markdown("### 💡 Giải thích")
                st.info(agent_result.explanation)

            # Reasoning steps
            if agent_result.reasoning_steps:
                with st.expander("🧠 Các bước suy luận", expanded=False):
                    for i, step in enumerate(agent_result.reasoning_steps):
                        st.markdown(f"**{i+1}.** {step}")

            # Columns: Word Cloud + Attention
            col_wc, col_attn = st.columns(2)

            with col_wc:
                from ui.components.word_cloud_viz import render_word_cloud
                render_word_cloud(article_text, title="📊 Word Cloud")

            with col_attn:
                from ui.components.attention_heatmap import render_attention_heatmap
                render_attention_heatmap(
                    text=article_text[:500],
                    risk_factors=agent_result.risk_factors if agent_result.risk_factors else None,
                )

            st.divider()

            # Evidence
            from ui.components.evidence_card import render_evidence_cards, render_sources_summary

            render_evidence_cards(
                evidence_citations=agent_result.evidence_citations,
                evidences=agent_result.evidence,
            )

            render_sources_summary(agent_result.sources_used)

            # Article Summary
            if agent_result.article_summary:
                with st.expander("📰 Tóm tắt bài viết"):
                    st.write(agent_result.article_summary)

            # Claims extracted
            if agent_result.claims:
                with st.expander(f"🔍 Claims trích xuất ({len(agent_result.claims)} claims)"):
                    for i, claim in enumerate(agent_result.claims):
                        importance = claim.get("importance", 0.5)
                        st.markdown(
                            f"**Claim {i+1}** (importance: {importance:.2f}): "
                            f"{claim.get('text', 'N/A')}"
                        )

            # Save to history
            st.session_state.history.append({
                "text": article_text[:100],
                "score": agent_result.fake_score,
                "label": agent_result.label,
            })

        # ── Comparison Mode ───────────────────────────────
        if agent_result and baseline_result:
            st.divider()
            st.markdown("## ⚖️ So sánh Models")

            col_agent, col_baseline = st.columns(2)

            with col_agent:
                st.markdown("### 🤖 AI Agent Pipeline")
                label_emoji = {"Real": "🟢", "Fake": "🔴", "Suspicious": "🟡"}.get(agent_result.label, "⚪")
                st.metric("Score", f"{agent_result.fake_score}%")
                st.markdown(f"**Label:** {label_emoji} {agent_result.label}")
                st.metric("Confidence", f"{agent_result.confidence:.0%}")

            with col_baseline:
                st.markdown("### 📊 PhoBERT Baseline")
                label_emoji = {"Real": "🟢", "Fake": "🔴"}.get(baseline_result["label"], "⚪")
                st.metric("Score", f"{baseline_result['score']}%")
                st.markdown(f"**Label:** {label_emoji} {baseline_result['label']}")
                st.metric("Confidence", f"{baseline_result['confidence']:.0%}")

            # Agreement check
            if agent_result.label == baseline_result["label"]:
                st.success("✅ Hai model thống nhất kết quả!")
            else:
                st.warning(
                    f"⚠️ Hai model không thống nhất: "
                    f"Agent → {agent_result.label}, PhoBERT → {baseline_result['label']}"
                )

elif not article_text:
    # Placeholder when no input
    st.markdown(
        """
        <div style="text-align:center;padding:60px;color:#aaa;">
            <p style="font-size:48px;">📰</p>
            <h3>Dán nội dung hoặc URL bài viết để bắt đầu</h3>
            <p>Hệ thống sẽ phân tích và đưa ra đánh giá về độ tin cậy</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

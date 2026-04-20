"""
ui/app.py — Modern Streamlit UI for Fake News Detection
======================================================
Focuses on the 8-stage Sequential Adversarial Pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sequential_adversarial.pipeline import SequentialAdversarialPipeline
from config import LLM_PROVIDER, SA_MODEL_NAME
from security import sanitize_for_display, validate_text_input
from exceptions import generate_correlation_id

logger = logging.getLogger(__name__)

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Verity — AI Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (High Aesthetics) ─────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 40px 0;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
    }
    .verdict-box {
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 20px;
    }
    .verdict-True { background: #10b981; }
    .verdict-False { background: #ef4444; }
    .verdict-Mixed { background: #f59e0b; }
    
    .stMetric {
        background: #f8fafc;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/shield.png", width=100)
    st.title("Verity Settings")
    
    selected_provider = st.selectbox("LLM Provider", ["gemini", "nvidia", "openai", "qwen", "grok", "gemma"], index=0)
    mock_mode = st.toggle("Mock Mode (Fast Test)", value=False)
    
    st.divider()
    st.caption(f"Backend: {SA_MODEL_NAME}")
    st.caption("UET Fake News Project — 2026")

# ── Main UI ──────────────────────────────────────────
st.markdown('<div class="main-header"><h1>🛡️ Verity: Sequential Adversarial Analysis</h1><p>Phát hiện tin giả tiếng Việt bằng hệ thống 8 tầng liên hoàn</p></div>', unsafe_allow_html=True)

# Input Section
input_col, help_col = st.columns([3, 1])

with input_col:
    source_input = st.text_area("Dán nội dung hoặc URL bài viết", height=200, placeholder="Nhập văn bản cần kiểm tra...")

with help_col:
    st.info("""
    **Cách hoạt động:**
    1. Trích xuất luận điểm.
    2. Đối chứng RAG (Wikipedia/DB).
    3. Phản biện định kiến.
    4. Tổng hợp phán quyết.
    5. Đối soát TF-IDF Baseline.
    """)

if st.button("🚀 BẮT ĐẦU PHÂN TÍCH", type="primary", use_container_width=True):
    # Validate input (FR-1.1, FR-1.2)
    is_valid, error_msg = validate_text_input(source_input)
    if not is_valid:
        st.error(f"Vui lòng nhập nội dung hợp lệ: {error_msg}")
    else:
        # Sanitize input before processing
        clean_input = sanitize_for_display(source_input)
        correlation_id = generate_correlation_id()

        with st.spinner("Đang khởi chạy Sequential Adversarial Pipeline..."):
            try:
                # Direct call to pipeline
                pipeline = SequentialAdversarialPipeline(mock=mock_mode)
                result = pipeline.run(clean_input)
                
                # ── RESULTS SECTION ──
                st.divider()
                
                # Verdict Header (sanitize output - NFR-7.2)
                verdict = result.verity_report.conclusion if result.verity_report else "Unknown"
                safe_verdict = sanitize_for_display(verdict)
                st.markdown(f'<div class="verdict-box verdict-{safe_verdict}">VERDICT: {safe_verdict.upper()}</div>', unsafe_allow_html=True)
                
                # Metrics Row
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Confidence", f"{result.verity_report.confidence:.0%}" if result.verity_report else "0%")
                m_col2.metric("Manipulation Score", f"{result.overall_manipulation_score:.0%}")
                m_col3.metric("Claims Detected", len(result.claims))
                m_col4.metric("TF-IDF Contrast", "Agreement" if (result.tfidf_comparison and result.tfidf_comparison.agreement) else "Conflict")
                
                # Tabs for Stages
                tab_report, tab_claims, tab_bias, tab_logic = st.tabs(["📝 Report", "🔍 Claims & Evidence", "⚖️ Bias Analysis", "🧬 Logic Flow"])
                
                with tab_report:
                    if result.verity_report:
                        # Sanitize markdown report (NFR-7.2)
                        safe_report = sanitize_for_display(result.verity_report.markdown_report)
                        st.markdown(safe_report)
                    else:
                        st.warning("Không có báo cáo tổng hợp.")
                
                with tab_claims:
                    for i, analysis in enumerate(result.claim_analyses):
                        # Sanitize claim text (NFR-7.2)
                        safe_claim_text = sanitize_for_display(analysis.claim.text[:100])
                        safe_verdict = sanitize_for_display(analysis.verdict)
                        with st.expander(f"Claim {i+1}: {safe_claim_text}...", expanded=(i==0)):
                            st.write(f"**Verdict:** {safe_verdict.upper()}")
                            if analysis.sources:
                                st.write("**Sources:**")
                                for s in analysis.sources:
                                    safe_url = sanitize_for_display(s.url)
                                    safe_stance = sanitize_for_display(s.stance)
                                    safe_excerpt = sanitize_for_display(s.excerpt)
                                    st.caption(f"- [{safe_stance.upper()}] {safe_url} (Reliability: {s.reliability:.2f})")
                                    st.markdown(f"> {safe_excerpt}")
                
                with tab_bias:
                    if result.bias_report:
                        # Sanitize bias report fields (NFR-7.2)
                        safe_framing = sanitize_for_display(result.bias_report.framing)
                        safe_notes = sanitize_for_display(result.bias_report.adversarial_notes)
                        distortion_text = "Yes" if result.bias_report.distortion_detected else "No"
                        st.write(f"**Framing:** {safe_framing}")
                        st.write(f"**Distortion Detected:** {distortion_text}")
                        st.info(f"**Adversarial Notes:**\n{safe_notes}")
                
                with tab_logic:
                    if result.mermaid_diagram:
                        # Sanitize Mermaid diagram (NFR-7.2)
                        safe_mermaid = sanitize_for_display(result.mermaid_diagram)
                        st.components.v1.html(f"""
                        <script type="module">
                            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                            mermaid.initialize({{ startOnLoad: true }});
                        </script>
                        <pre class="mermaid">
                            {safe_mermaid}
                        </pre>
                        """, height=600, scrolling=True)
                
                # TF-IDF Details
                if result.tfidf_comparison:
                    # Sanitize TF-IDF comparison output (NFR-7.2)
                    safe_tfidf_label = sanitize_for_display(result.tfidf_comparison.tfidf_label)
                    safe_llm_verdict = sanitize_for_display(result.tfidf_comparison.llm_verdict)
                    safe_notes = sanitize_for_display(result.tfidf_comparison.disagreement_notes)
                    with st.expander("🤖 Chi tiết đối soát Baseline (Stage 8)"):
                        st.write(f"**Baseline Prediction:** {safe_tfidf_label}")
                        st.write(f"**AI Prediction:** {safe_llm_verdict}")
                        st.markdown(f"> {safe_notes}")

            except Exception as e:
                st.error(f"Lỗi hệ thống: {e}")
                logger.exception("Pipeline Error")

elif not source_input:
    st.markdown("""
    <div style="text-align:center; padding:100px; color:#94a3b8;">
        <h3>Sẵn sàng phân tích tin tức</h3>
        <p>Hệ thống sẽ chạy qua 8 giai đoạn để đảm bảo tính khách quan.</p>
    </div>
    """, unsafe_allow_html=True)

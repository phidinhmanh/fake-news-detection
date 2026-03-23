"""
app.py — Streamlit App (Main Entry Point)
============================================
Người C phát triển.

Chạy:
    streamlit run ui/app.py

TODO (Tuần 1-2):
    - [x] Layout skeleton: input box, sidebar, result placeholder
    - [ ] Connect mock server, test API flow end-to-end

TODO (Tuần 3-4):
    - [ ] Prediction score bar + confidence gauge
    - [ ] SHAP token highlight

TODO (Tuần 5-6):
    - [ ] Domain badge + Source credibility panel
    - [ ] History log + CSV export
"""

import streamlit as st
import requests

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────
API_URL = "http://localhost:8000/predict"

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Cài đặt")
    lang = st.selectbox("Ngôn ngữ", options=["vi", "en"], index=0)
    st.divider()
    st.caption("Fake News Detection System v0.1")
    st.caption("Nhóm: A + B + C")

# ── Main Content ───────────────────────────────────────
st.title("🔍 Fake News Detector")
st.markdown("Dán nội dung bài viết bên dưới để kiểm tra tin giả.")

# Input
text_input = st.text_area(
    "Nội dung bài viết",
    height=200,
    max_chars=2048,
    placeholder="Dán nội dung bài viết cần kiểm tra tại đây...",
)

# Predict button
if st.button("🔎 Kiểm tra", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Vui lòng nhập nội dung bài viết.")
    else:
        with st.spinner("Đang phân tích..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": text_input, "lang": lang},
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()

                # ── Result Display ─────────────────────
                st.divider()

                # Label + Confidence
                col1, col2 = st.columns(2)
                with col1:
                    label = result["label"]
                    if label == "fake":
                        st.error(f"⚠️ Kết quả: **TIN GIẢ** ({label.upper()})")
                    else:
                        st.success(f"✅ Kết quả: **TIN THẬT** ({label.upper()})")

                with col2:
                    confidence = result["confidence"]
                    st.metric("Độ tin cậy", f"{confidence:.0%}")
                    st.progress(confidence)

                # Domain
                domain = result.get("domain", "unknown")
                st.info(f"📂 Lĩnh vực: **{domain.capitalize()}**")

                # SHAP tokens (placeholder — Người C sẽ thay bằng component)
                shap_tokens = result.get("shap_tokens", [])
                if shap_tokens:
                    st.subheader("🔬 Từ khóa quan trọng")
                    for token, weight in shap_tokens:
                        col_t, col_w = st.columns([3, 1])
                        with col_t:
                            st.write(f"`{token}`")
                        with col_w:
                            st.write(f"{weight:.2f}")

                # Source score
                source_score = result.get("source_score")
                if source_score is not None:
                    st.subheader("📊 Độ tin cậy nguồn")
                    st.progress(source_score)
                    st.caption(f"Score: {source_score:.2f}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Không thể kết nối API server. "
                    "Chạy `make mock` hoặc `uvicorn mock_server:app --port 8000`"
                )
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Lỗi API: {e}")

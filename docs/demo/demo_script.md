# Demo Script — Video 3-5 phút

## Mục tiêu
Quay video demo hệ thống phát hiện tin giả tiếng Việt, thể hiện đầy đủ chức năng.

---

## Timeline (3-5 phút)

### [0:00 - 0:30] Giới thiệu
- Chào mừng, giới thiệu nhóm
- "Hôm nay chúng tôi demo hệ thống Phát hiện Tin giả Tiếng Việt"
- Mô tả ngắn: PhoBERT + LLM Agent + ViFactCheck

### [0:30 - 1:30] Demo 1: Tin GIẢ
- Dán 1 bài tin giả (vaccine gây tử vong / kinh tế sập)
- Click "Phân tích"
- Highlight:
  - Score gauge hiện 85% → Fake 🔴
  - Giải thích chi tiết từ Agent 3
  - Evidence citations từ ViFactCheck
  - Word cloud + attention heatmap

### [1:30 - 2:30] Demo 2: Tin THẬT
- Dán 1 bài tin thật từ VnExpress
- Click "Phân tích"
- Highlight:
  - Score gauge hiện 15% → Real 🟢
  - Evidence support từ knowledge base
  - So sánh với tin giả trước đó

### [2:30 - 3:30] Demo 3: So sánh Models
- Chọn mode "Cả hai (so sánh)"
- Chạy phân tích → hiện bảng so sánh PhoBERT vs Agent
- Giải thích sự khác biệt

### [3:30 - 4:00] Architecture Overview
- Show kiến trúc diagram
- Giải thích: 3 agents, RAG pipeline, ViFactCheck KB
- Nhấn mạnh novelty

### [4:00 - 4:30] Kết quả & Metrics
- Show bảng comparison table
- Show confusion matrix
- Show ablation study results

### [4:30 - 5:00] Kết luận
- Tóm tắt đóng góp
- Hướng phát triển
- Cảm ơn

---

## Bài tin giả mẫu (để demo)

```
KHẨN CẤP: Vaccine COVID-19 gây ra hàng nghìn ca tử vong trên toàn thế giới! 
Theo một nghiên cứu bí mật bị rò rỉ từ phòng thí nghiệm Wuhan, các hãng dược phẩm 
lớn đã che giấu sự thật về tác dụng phụ nghiêm trọng của vaccine. Chính phủ nhiều 
nước đã biết nhưng vẫn tiếp tục ép buộc tiêm chủng cho người dân. CHIA SẺ NGAY 
để cứu mọi người!!!
```

## Bài tin thật mẫu (để demo)

```
Thủ tướng Phạm Minh Chính chủ trì Hội nghị trực tuyến toàn quốc về phát triển kinh tế 
xã hội năm 2026.  Tại hội nghị, Thủ tướng nhấn mạnh cần tiếp tục thực hiện hiệu quả 
các chính sách hỗ trợ doanh nghiệp và người lao động, đẩy mạnh chuyển đổi số, 
phát triển kinh tế xanh. GDP quý I/2026 ước tính tăng 6.8% so với cùng kỳ năm trước, 
theo số liệu từ Tổng cục Thống kê.
```

---

## Công cụ quay
- OBS Studio hoặc Loom
- Resolution: 1920x1080
- Micro: nói rõ, giọng tự tin
- Chạy Streamlit: `streamlit run ui/app.py`

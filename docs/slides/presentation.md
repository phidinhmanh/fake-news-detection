---
marp: true
theme: default
paginate: true
backgroundColor: #1a1a2e
color: #e0e0e0
style: |
  section {
    font-family: 'Inter', sans-serif;
  }
  h1 {
    color: #667eea;
  }
  h2 {
    color: #764ba2;
  }
  .columns {
    display: flex;
    gap: 20px;
  }
  .column {
    flex: 1;
  }
---

# 🔍 Phát hiện Tin giả Tiếng Việt
## PhoBERT + LLM Agent Pipeline

**UET-VNU — 2026**

---

## 📋 Nội dung

1. Đặt vấn đề
2. Phương pháp đề xuất
3. Dataset & Preprocessing
4. Kết quả thực nghiệm
5. Demo
6. Kết luận

---

## ❓ Đặt vấn đề

- Tin giả lan truyền nhanh trên mạng xã hội Việt Nam
- Thiếu công cụ fact-checking tự động cho tiếng Việt
- Cần: **phát hiện** + **giải thích** lý do

### 🎯 Mục tiêu
- Xây dựng hệ thống phát hiện tin giả tiếng Việt
- Kết hợp deep learning + LLM reasoning
- Cung cấp explainable output

---

## 🏗️ Kiến trúc hệ thống

```
Bài viết → [PhoBERT + Features] → Score 1
         → [Agent 1: Claim] → [Agent 2: Evidence] → [Agent 3: Score] → Score 2
                                                                      → Giải thích
```

### 🎯 Novelty
1. ViFactCheck (AAAI 2025) — dataset mới nhất
2. 3-Agent LLM pipeline cho tiếng Việt
3. Explainable AI output

---

## 📊 Dataset

| Dataset | Số lượng | Nguồn |
|---------|----------|-------|
| ViFactCheck | 7,232 | AAAI 2025 |
| ReINTEL | ~2,000 | VLSP 2020 |
| Self-collected | 300-500 | VnExpress, Tuổi Trẻ |

### Preprocessing
- underthesea word tokenization
- 9 stylistic features extraction
- 80/10/10 stratified split

---

## 🤖 PhoBERT Baseline

- **Model**: `vinai/phobert-base` (pre-trained cho tiếng Việt)
- **Architecture**: PhoBERT → [CLS] → Linear(768, 2) → Softmax
- **Cải tiến**: + 9 stylistic features → Linear(777, 256) → 2

### Stylistic Features
- Số từ, số câu, độ dài câu TB
- Tỷ lệ dấu !, ?, viết hoa
- Sentiment score, URL count, number ratio

---

## 🧠 LLM Agent Pipeline

### Agent 1: Claim Extractor
- Trích xuất 1-5 claim chính cần kiểm chứng

### Agent 2: Evidence Searcher (RAG)
- ViFactCheck knowledge base (LanceDB vector store)
- Wikipedia VN API search

### Agent 3: Reasoning Scorer
- Chain-of-thought reasoning
- Score (0-100%) + Label + Giải thích tiếng Việt

---

## 📈 Kết quả

| Model | Accuracy | F1 | AUC |
|-------|----------|-----|-----|
| TF-IDF + LogReg | X.XX | X.XX | X.XX |
| PhoBERT baseline | X.XX | X.XX | X.XX |
| PhoBERT + Features | X.XX | X.XX | X.XX |
| **Proposed (Full)** | **X.XX** | **X.XX** | **X.XX** |

### Ablation Study
- Bỏ features: giảm X% F1
- Bỏ agent: giảm X% F1

---

## 🖥️ Demo

![Web Demo](figures/web_demo.png)

### Features
- ✅ Score gauge (0-100%)
- ✅ Label: Real / Fake / Suspicious
- ✅ Giải thích chi tiết + evidence
- ✅ Word cloud + attention heatmap

---

## 📝 Kết luận

### Đóng góp
- Hệ thống fake news detection tiếng Việt đầy đủ
- ViFactCheck + PhoBERT + LLM Agent = State-of-the-art
- Explainable output giúp người dùng hiểu lý do

### Hạn chế & Hướng phát triển
- Latency do LLM API → fine-tune model nhỏ
- Mở rộng knowledge base
- Multimodal (text + image)

---

# 🙏 Cảm ơn!

**Q&A**

GitHub: [link]
Demo: [HuggingFace Spaces link]

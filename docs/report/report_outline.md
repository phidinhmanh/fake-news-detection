# Phát hiện Tin giả Tiếng Việt bằng PhoBERT và LLM Agent Pipeline

## Nhóm thực hiện
- **Trường**: Đại học Công nghệ — ĐHQGHN (UET-VNU)
- **Học kỳ**: 2025-2026

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Tổng quan nghiên cứu liên quan](#2-tổng-quan-nghiên-cứu-liên-quan)
3. [Phương pháp đề xuất](#3-phương-pháp-đề-xuất)
4. [Dataset](#4-dataset)
5. [Thực nghiệm](#5-thực-nghiệm)
6. [Kết quả](#6-kết-quả)
7. [Kết luận](#7-kết-luận)

---

## 1. Giới thiệu

### 1.1 Đặt vấn đề
- Tin giả (fake news) là vấn đề nghiêm trọng trong kỷ nguyên số
- Đặc biệt khó khăn cho tiếng Việt do thiếu công cụ và dataset chuyên dụng

### 1.2 Mục tiêu
- Xây dựng hệ thống phát hiện tin giả tiếng Việt tự động
- Kết hợp PhoBERT (deep learning) + LLM Agent (reasoning)
- Cung cấp giải thích cho người dùng (explainable AI)

### 1.3 Đóng góp chính (Novelty)
1. **Ứng dụng ViFactCheck** (AAAI 2025) — dataset mới nhất cho fact-checking tiếng Việt
2. **LLM Agent pipeline** — 3-agent system (Claim Extraction → RAG Evidence → Reasoning)
3. **Explainable output** — giải thích chi tiết + evidence citations

---

## 2. Tổng quan nghiên cứu liên quan

### 2.1 Fake News Detection
- Các phương pháp truyền thống (TF-IDF, SVM, Random Forest)
- Deep learning approaches (LSTM, BERT, RoBERTa)
- Multilingual models (XLM-RoBERTa, mBERT)

### 2.2 Vietnamese NLP
- PhoBERT (Nguyen & Nguyen, 2020) — pretrained model cho tiếng Việt
- vELECTRA — ELECTRA pretrained cho tiếng Việt
- Underthesea — bộ công cụ NLP tiếng Việt

### 2.3 Fact-Checking with LLMs
- LLM as fact-checkers (Qin et al., 2023)
- Retrieval-Augmented Generation (RAG) for fact verification
- Multi-agent systems for complex reasoning

### 2.4 Vietnamese Fake News Datasets
- **ViFactCheck** (AAAI 2025) — 7,232 claim-evidence pairs
- **ReINTEL** (VLSP 2020) — Vietnamese social media reliability
- **VFND** — Vietnamese fake news dataset

---

## 3. Phương pháp đề xuất

### 3.1 Kiến trúc tổng thể
![Architecture Diagram](figures/architecture_diagram.png)

### 3.2 PhoBERT Baseline
- Fine-tune `vinai/phobert-base` cho binary classification
- Architecture: PhoBERT → [CLS] → Linear(768, 2) → Softmax
- Training: AdamW + Linear Warmup + Early Stopping

### 3.3 PhoBERT + Stylistic Features
- 9 stylistic features: word_count, sentiment, exclamation_ratio, ...
- Concatenation: PhoBERT [CLS] (768) ⊕ features (9) → MLP → Softmax

### 3.4 LLM Agent Pipeline
- **Agent 1**: Claim Extractor — trích xuất luận điểm chính
- **Agent 2**: Evidence Searcher — RAG search từ ViFactCheck KB + Wikipedia VN
- **Agent 3**: Reasoning Scorer — chain-of-thought reasoning → score + giải thích

### 3.5 Knowledge Base (RAG)
- ViFactCheck evidence → sentence-transformers embedding → LanceDB vector store
- Semantic search: query → top-k similar evidence

---

## 4. Dataset

### 4.1 Nguồn dữ liệu
| Dataset | Số lượng | Ngôn ngữ | Nguồn |
|---------|----------|----------|-------|
| ViFactCheck | 7,232 | Việt | HuggingFace (AAAI 2025) |
| ReINTEL | ~2,000 | Việt | VLSP 2020 |
| Self-collected | 300-500 | Việt | VnExpress, Tuổi Trẻ |

### 4.2 Preprocessing
- Unicode normalization (NFC)
- Word segmentation (underthesea)
- HTML/URL removal
- Language detection

### 4.3 Train/Val/Test Split
- 80% train / 10% validation / 10% test
- Stratified split (giữ tỷ lệ label)

---

## 5. Thực nghiệm

### 5.1 Setup
- Hardware: (GPU/CPU info)
- Framework: PyTorch, Transformers, Streamlit
- Gemini API cho LLM Agent

### 5.2 Models so sánh
1. TF-IDF + Logistic Regression (baseline)
2. PhoBERT baseline
3. PhoBERT + Stylistic Features
4. Agent Pipeline (LLM)
5. Full Proposed System

### 5.3 Metrics
- Accuracy, F1-score, Precision, Recall
- AUC-ROC
- Confusion Matrix
- Inference time

---

## 6. Kết quả

### 6.1 Bảng so sánh
(Bảng comparison_table.md)

### 6.2 Confusion Matrix
(Hình confusion matrices)

### 6.3 Ablation Study
(Bảng ablation_table.md)

### 6.4 Web Demo
![Web Demo Screenshot](figures/web_demo.png)

---

## 7. Kết luận

### 7.1 Tóm tắt
- Đề xuất hệ thống fake news detection cho tiếng Việt kết hợp PhoBERT + LLM Agent
- Đạt F1-score ≥ X.XX trên ViFactCheck test set
- Agent pipeline cung cấp giải thích chi tiết cho người dùng

### 7.2 Hạn chế
- Phụ thuộc vào LLM API (chi phí, latency)
- Knowledge base giới hạn bởi ViFactCheck

### 7.3 Hướng phát triển
- Mở rộng knowledge base (thêm nguồn fact-check VN)
- Fine-tune LLM nhỏ (Gemma, Qwen) thay vì dùng API
- Multimodal: phát hiện ảnh giả kết hợp text

---

## Tài liệu tham khảo

1. ViFactCheck: A New Benchmark Dataset and Methods for Multi-domain News Fact-Checking in Vietnamese (AAAI 2025)
2. PhoBERT: Pre-trained Language Models for Vietnamese (Nguyen & Nguyen, 2020)
3. Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (NeurIPS 2020)
4. ...

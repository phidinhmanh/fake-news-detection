# Fake News Detection System (UET Group Project)
# 🔍 🚀 🛡️

Hệ thống phân loại tin giả đa ngôn ngữ (Tiếng Việt & Tiếng Anh) sử dụng XLM-RoBERTa fine-tuned với LoRA, kết hợp Ensemble và giải thích mô hình bằng SHAP.

---

## 📋 Mục lục
1. [Tổng quan](#tổng quan)
2. [Kiến trúc hệ thống](#kiến trúc hệ thống)
3. [Phân công nhiệm vụ](#phân công nhiệm vụ)
4. [Hướng dẫn cài đặt](#hướng dẫn cài đặt)
5. [Quy trình phát triển](#quy trình phát triển)
6. [Lộ trình 7 tuần](#lộ trình 7 tuần)

---

## 🔍 Tổng quan
Dự án được thực hiện bởi nhóm 3 người trong vòng 7 tuần.
- **Mục tiêu**: Đạt F1-score ≥ 0.85 trên tập test tiếng Việt.
- **Công nghệ chính**:
  - **Backend/Data**: Scrapy, Underthesea, Pandas, FastAPI.
  - **Model**: PyTorch Lightning, Transformers, PEFT (LoRA), SHAP.
  - **Frontend**: Streamlit, Plotly.

---

## 🏗️ Kiến trúc hệ thống
Hệ thống bao gồm 3 module chính hoạt động song song qua API Contract:
1. **Data Module**: Crawl dữ liệu, tiền xử lý, tăng cường dữ liệu và phân loại lĩnh vực.
2. **Model Module**: Huấn luyện XLM-RoBERTa, Ensemble mô hình và tính toán SHAP tokens.
3. **UI Module**: Giao diện người dùng Streamlit, hiển thị kết quả phân loại và giải thích trực quan.

---

## 👥 Phân công nhiệm vụ

### 🧑‍💻 Người A — Data Pipeline & Domain Classifier
- Chịu trách nhiệm toàn bộ luồng dữ liệu.
- Output: `DataModule` chuẩn PyTorch Lightning để Người B dùng trực tiếp.
- Stack: `scrapy`, `underthesea`, `pandas`.

### 🧠 Người B — Model Training & Ensemble
- Chịu trách nhiệm toàn bộ ML pipeline.
- Output: `predictor.py` với interface khớp `schemas.py`.
- Stack: `transformers`, `peft`, `shap`, `lightning`.

### 🎨 Người C — Streamlit UI & Deployment
- Làm UI hoàn toàn độc lập từ ngày 1 bằng Mock Server.
- Output: Streamlit app v2 hoàn chỉnh.
- Stack: `streamlit`, `requests`, `plotly`.

---

## 🛠️ Hướng dẫn cài đặt

### 1. Clone repository
```bash
git clone https://github.com/your-repo/fake-news-detection.git
cd fake-news-detection
```

### 2. Tạo môi trường ảo và cài đặt dependencies
```bash
# Cài đặt tất cả (cho dev)
make install

# Hoặc cài theo module riêng (cho production/training)
make install-data   # Người A
make install-model  # Người B
make install-ui     # Người C
```

### 3. Chạy Mock Server (Dành cho Người C phát triển UI)
```bash
make mock
# Hoặc: uvicorn mock_server:app --port 8000 --reload
```

### 4. Chạy UI App
```bash
make ui
# Hoặc: streamlit run ui/app.py
```

---

## 🔄 Quy trình phát triển (Git Workflow)
1. **Branch Protection**: `main` branch yêu cầu Pull Request và ít nhất 1 reviewer.
2. **Branch Naming**:
   - `feat/data-pipeline`, `feat/data-augmentation` (Người A)
   - `feat/baseline-model`, `feat/lora-finetune`, `feat/ensemble-shap` (Người B)
   - `feat/streamlit-skeleton`, `feat/prediction-ui`, `feat/full-ui` (Người C)
3. **API Contract**: `schemas.py` là hợp đồng chung, không thay đổi sau tuần 2.

---

## 📅 Lộ trình 7 tuần

| Tuần | Người A (Data) | Người B (Model) | Người C (UI) |
|------|----------------|-----------------|--------------|
| 1-2 | Crawler + Dataset Norm | Baseline (F1 ≥ 0.72) | UI Skeleton + Mock |
| 3-4 | Augmentation + Balancing | LoRA Fine-tune (F1 ≥ 0.85) | Prediction UI components |
| 5-6 | Domain Classifier | Ensemble + SHAP (AUC ≥ 0.90) | Full UI + Components |
| 7 | Integration Test | Integration Test | Deploy Streamlit Cloud |

---

## 🧪 Testing
```bash
# Chạy tất cả tests
make test

# Test riêng API Contract
make test-schemas
```

## 📬 Liên hệ
- Nhóm sinh viên UET - Fake News Detection Project.
- Hỗ trợ kỹ thuật: Xem `TODO_LIST.md` để biết tiến độ chi tiết.

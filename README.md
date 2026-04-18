# Fake News Detection System (UET Group Project)
# 🔍 🚀 🛡️

Hệ thống phân loại tin giả đa ngôn ngữ (Tiếng Việt & Tiếng Anh) sử dụng kiến trúc AI tiên tiến kết hợp **PhoBERT** (với Stylistic Features) và **Adversarial LLM Pipeline** (hệ thống tác tử đa LLM tranh biện kết hợp RAG).

---

## 📋 Mục lục
1. [Tổng quan](#-tổng-quan)
2. [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
3. [Hướng dẫn cài đặt & Chạy Code (Full)](#%EF%B8%8F-hướng-dẫn-cài-đặt--chạy-code-full)
4. [Lộ trình & Phân công](#-phân-công-nhiệm-vụ)
5. [Quy trình phát triển](#-quy-trình-phát-triển-git-workflow)

---

## 🔍 Tổng quan
Dự án được thực hiện bởi nhóm sinh viên UET trong 7 tuần.
- **Mục tiêu**: Phân loại tin giả và tin thật (tập trung vào ngôn ngữ Tiếng Việt), hỗ trợ giải thích lý do (Fact-checking) dựa trên bằng chứng minh bạch thông qua hệ thống RAG.
- **Công nghệ chính**:
  - **Data**: ViFactCheck, ReINTEL, Underthesea.
  - **Model**: PhoBERT (Base/Pro với features), Sequential Adversarial Pipeline (LLMs), PyTorch.
  - **Backend & Tooling**: Python 3.10+, `uv`, FastAPI, pytest.
  - **Frontend**: Streamlit, Plotly.

---

## 🏗️ Kiến trúc hệ thống
Hệ thống bao gồm 3 module chính hoạt động linh hoạt:
1. **Data Module**: Download, thu thập (scrapy), tiền xử lý (underthesea tokenization) và trích xuất đặc trưng phong cách viết (stylistic features).
2. **Model Module**: Huấn luyện PhoBERT (Baseline) và kết hợp đặc trưng hành vi (Pro). Kết hợp pipeline đa đặc vụ LLM (Sequential Adversarial) để phản biện và đánh giá độ uy tín.
3. **UI Module**: Giao diện người dùng Streamlit dễ sử dụng, cung cấp phân tích trực quan.

---

## 🛠️ Hướng dẫn cài đặt & Chạy Code (Full)

Phần này hướng dẫn chi tiết quy trình từ lúc bắt đầu cho đến khi chạy được toàn bộ ứng dụng. **Chú ý:** Các lệnh chạy code sử dụng `uv run` thay vì gọi `python` trực tiếp để đảm bảo dùng đúng môi trường ảo.

### 1. Chuẩn bị môi trường
Yêu cầu hệ thống: Cần cài đặt `uv` package manager.

```bash
# Clone source code
git clone https://github.com/your-repo/fake-news-detection.git
cd fake-news-detection

# Cài đặt toàn bộ dependencies và môi trường ảo (.venv) bằng uv
make install
# Hoặc lệnh tương đương: uv sync
```

### 2. Chuẩn bị dữ liệu (Data Pipeline)
Thực hiện lần lượt các bước sau để tải và chuẩn bị dữ liệu train/test.

```bash
# Bước 2.1: Download dữ liệu gốc (ViFactCheck)
uv run python dataset/download_datasets.py

# Bước 2.2 (Tùy chọn): Thu thập thêm tin tức từ VnExpress, Tuổi Trẻ
uv run python dataset/collect_news.py

# Bước 2.3: Preprocessing (Làm sạch, Tokenize cho Tiếng Việt dùng underthesea)
uv run python dataset/preprocess_vietnamese.py

# Bước 2.4: Gộp các tập dữ liệu thành processed (Chia Train/Val/Test)
uv run python dataset/merge_datasets.py

# Bước 2.5: Trích xuất các đặc trưng văn phong (Stylistic features)
uv run python dataset/feature_extraction.py
```
> ⚠️ **Chú ý quan trọng**: Dữ liệu train/test sau khi chạy các bước trên sẽ nằm trong `dataset/processed/`. Nếu bạn bỏ qua chạy dòng **2.3** và **2.4** mà sang bước 3 luôn, bạn sẽ bắt gặp lỗi `FileNotFoundError: Data not found: dataset/processed/train.csv`.

### 3. Huấn luyện mô hình (Model Training)
Huấn luyện mô hình phân loại lõi dựa trên dữ liệu đã được xử lý.

```bash
# Huấn luyện mô hình Baseline (Chỉ sử dụng text thuần túy)
uv run python model/train_phobert.py --variant baseline --epochs 5 --batch-size 32

# Huấn luyện mô hình Pro (Văn bản + Stylistic Features từ bước 2.5)
uv run python model/train_phobert.py --variant pro --epochs 5 --batch-size 32

# Chạy Cross-Validation (Đánh giá và so sánh chuẩn)
uv run python train_cv_simple.py
uv run python train_cv_comparison.py
```

### 4. Hệ thống Đa đặc vụ (LLM Pipeline)
Thử nghiệm pipeline hệ thống tác tử nâng cao.

```bash
# Cấu hình API key của các nhà cung cấp (NVIDIA NIM, DeepSeek, OpenAI...)
cp .env.example .env
# (Lưu ý: Mở file .env và điền các API keys của bạn vào)

# Chạy test pipeline
bash run_pipeline.sh
```

### 5. Khởi chạy Giao diện (Streamlit UI)
Chạy trải nghiệm Web App:

```bash
# Terminal 1: Chạy API Server giả lập (Mock) hỗ trợ cho UI
make mock
# Tương đương: uv run uvicorn api.mock:app --port 8000 --reload

# Terminal 2: Chạy Web App Streamlit
make ui
# Tương đương: uv run streamlit run ui/app.py
```
Truy cập **http://localhost:8501** trên trình duyệt để trải nghiệm.

### 6. Kiểm thử & Định dạng mã nguồn (Testing & Linting)
Dành cho quá trình phát triển (Dev):
```bash
make test          # Chạy toàn bộ tests bằng pytest
make test-schemas  # Kiểm tra ràng buộc API Contract
make lint          # Quét chuẩn lỗi code với ruff
make format        # Định dạng nhanh code
make clean         # Dọn dẹp cache của python và bộ nhớ đệm
```

---

## 👥 Phân công nhiệm vụ

*Mô hình chia team lý thuyết cho dự án:*
- **Người A (Data Pipeline):** Web crawling, làm sạch, xử lý ngôn ngữ tự nhiên Tiếng Việt (Underthesea), xây dựng Feature Engineering module.
- **Người B (Model & AI):** Huấn luyện PhoBERT, tối ưu hiệu suất, triển khai bộ LLM Agents.
- **Người C (Streamlit & Deployment):** Phát triển giao diện từ file thiết kế và Mock API, build chart trực quan tương tác.

---

## 🔄 Quy trình phát triển (Git Workflow)
1. **Branch Protection**: Nhánh `main` yêu cầu mở Pull Request (PR) đính kèm giải thích, có review chéo.
2. **Branch Naming**:
   - Tính năng mới: `feat/...` (VD: `feat/data-pipeline`, `feat/phobert-baseline`)
   - Sửa lỗi: `fix/...`
3. **API Contract**: Các kỹ sư thống nhất schemas giao tiếp giữa UI và Model qua thư mục `api/` nhằm độc lập phát triển.

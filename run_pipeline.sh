#!/bin/bash

# ==============================================================================
# 🔍 VIETNAMESE FAKE NEWS DETECTION SYSTEM - RUNNER
# ==============================================================================
# Tác giả: UET Project Team
# Đặ điểm: Chạy toàn bộ quy trình từ Terminal, giao diện tương tác đẹp mắt.
# ==============================================================================

# ── ANSI Colors ───────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Icons ─────────────────────────────────────────────────────────────────────
CHECK="${GREEN}✔${NC}"
CROSS="${RED}✘${NC}"
INFO="${BLUE}ℹ${NC}"
WARN="${YELLOW}⚠${NC}"
ROCKET="🚀"

# ── Functions ─────────────────────────────────────────────────────────────────
print_banner() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "=========================================================="
    echo "   🔍 VIETNAMESE FAKE NEWS DETECTION SYSTEM RUNNER"
    echo "=========================================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${BOLD}${BLUE}==> Step $1: $2${NC}"
}

check_api_keys() {
    print_step "1" "Kiểm tra API Keys"
    
    # Check GOOGLE_API_KEY
    key=$GOOGLE_API_KEY
    if [ -z "$key" ]; then
        echo -e "${WARN} Không tìm thấy GOOGLE_API_KEY trong môi trường."
        read -p "   Nhập GOOGLE_API_KEY (Gemini): " key_input
        # Strip prefixes if any
        key=$(echo "$key_input" | sed 's/GOOGLE_API_KEY=//g' | xargs)
        if [ ! -z "$key" ]; then
            export GOOGLE_API_KEY=$key
            echo -e "   ${CHECK} Đã thiết lập GOOGLE_API_KEY."
        else
            echo -e "   ${WARN} Chạy ở chế độ MOCK cho LLM Agent."
        fi
    else
        echo -e "   ${CHECK} GOOGLE_API_KEY đã sẵn sàng."
    fi

    # Check HF_TOKEN
    token=$HF_TOKEN
    if [ -z "$token" ]; then
        echo -e "${WARN} Không tìm thấy HF_TOKEN."
        read -p "   Nhập HF_TOKEN (Hugging Face): " token_input
        # Strip prefixes if any
        token=$(echo "$token_input" | sed 's/HF_TOKEN=//g' | xargs)
        if [ ! -z "$token" ]; then
            export HF_TOKEN=$token
            if python3 -c "from huggingface_hub import login; login(token='$token', add_to_git_credential=False)"; then
                echo -e "   ${CHECK} Đã đăng nhập HF Hub."
            else
                echo -e "   ${CROSS} Đăng nhập HF Hub thất bại. Kiểm tra lại Token."
            fi
        else
            echo -e "   ${WARN} Một số model có thể không tải được nếu thiếu token."
        fi
    else
        echo -e "   ${CHECK} HF_TOKEN đã sẵn sàng."
    fi
}

install_deps() {
    print_step "2" "Cài đặt Dependencies"
    echo -e "⏳ Đang cài đặt thư viện phần mềm (underthesea, transformers, lancedb...)"
    pip install -q underthesea transformers datasets lancedb sentence-transformers google-generativeai langdetect wordcloud plotly streamlit pyngrok python-dotenv huggingface_hub
    echo -e "${CHECK} Cài đặt hoàn tất."
}

data_preparation() {
    print_step "3" "Chuẩn bị Dữ liệu"
    echo "   1. Tải dữ liệu thật (HF Hub/ViFactCheck)"
    echo "   2. Sử dụng Mock Data (Chạy nhanh để test)"
    read -p "   Lựa chọn của bạn (1/2): " data_choice
    
    if [ "$data_choice" == "1" ]; then
        echo -e "${ROCKET} Đang tải dữ liệu thật..."
        python3 dataset/download_datasets.py
    else
        echo -e "${INFO} Đang tạo Mock Data..."
        # Gọi script nội tại để tạo mock data - FIX IMPORT Path
        python3 <<EOF
import config, pandas as pd
from pathlib import Path
import os
config.DATASET_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
texts = ["Vaccine cực kỳ nguy hiểm!", "Thủ tướng họp báo.", "Giá xăng tăng."] * 10
mock_data = pd.DataFrame({
    'text': texts, 
    'label': ['fake', 'real', 'fake']*10, 
    'source': ['social', 'news', 'news']*10, 
    'lang': ['vi']*30
})
mock_path = config.DATASET_PROCESSED_DIR / "preprocessed_all.csv"
mock_data.to_csv(mock_path, index=False)
print(f"✅ Created mock data at {mock_path}")
EOF
    fi
    
    echo -e "⏳ Đang tiền xử lý và trích xuất đặc trưng..."
    python3 dataset/merge_datasets.py
    python3 dataset/feature_extraction.py
    echo -e "${CHECK} Dữ liệu đã sẵn sàng."
}

run_test_pipeline() {
    print_step "4" "Test Agentic Pipeline"
    read -p "   Nhập nội dung bài báo cần phân tích (Để trống để dùng mẫu): " news_text
    if [ -z "$news_text" ]; then
        news_text="Cảnh báo: Sóng thần sắp đổ bộ vào miền Trung Việt Nam tối nay."
    fi
    
    echo -e "${ROCKET} Đang khởi động LLM Agent..."
    # Dùng python3 để chạy pipeline trực tiếp
    python3 <<EOF
import os
from sequential_adversarial.pipeline import SequentialAdversarialPipeline
pipeline = SequentialAdversarialPipeline(mock=(not os.getenv("GOOGLE_API_KEY")))
print("--- ANALYSIS RESULTS ---")
result = pipeline.run("$news_text")
if result.verity_report:
    print(f"CONCLUSION: {result.verity_report.conclusion}")
    print(f"CONFIDENCE: {result.verity_report.confidence:.0%}")
else:
    print("Error: Could not generate report.")
EOF
}

run_web_ui() {
    print_step "5" "Khởi động Giao diện Web (Streamlit)"
    echo -e "${INFO} Streamlit sẽ chạy trên cổng 8501."
    streamlit run ui/app.py
}

# ── Main Loop ─────────────────────────────────────────────────────────────────
while true; do
    print_banner
    echo -e "${BOLD}LỰA CHỌN QUY TRÌNH:${NC}"
    echo -e "  1.  [Full] Chạy toàn bộ quy trình từ A-Z"
    echo -e "  2.  Cài đặt thư viện (Install Deps)"
    echo -e "  3.  Phân tích bài viết (LLM Agent)"
    echo -e "  4.  Huấn luyện mô hình PhoBERT (Cần GPU)"
    echo -e "  5.  Chạy Giao diện Web (Streamlit UI)"
    echo -e "  6.  Thoát"
    echo ""
    read -p "Chọn (1-6): " opt

    case $opt in
        1)
            check_api_keys
            install_deps
            data_preparation
            run_test_pipeline
            run_web_ui
            ;;
        2) install_deps ;;
        3) check_api_keys; run_test_pipeline ;;
        4) python3 model/train_phobert.py --variant baseline ;;
        5) run_web_ui ;;
        6) echo "Tạm biệt!"; exit 0 ;;
        *) echo -e "${RED}Lựa chọn không hợp lệ.${NC}"; sleep 1 ;;
    esac
    
    echo -e "\n${YELLOW}Ấn phím bất kỳ để quay lại menu...${NC}"
    read -n 1 -s
done

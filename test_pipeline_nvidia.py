import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env (nếu có chứa NVIDIA_API_KEY)
load_dotenv()

# Bắt buộc sử dụng NVIDIA cung cấp thay vì Gemini
os.environ["LLM_PROVIDER"] = "nvidia"

# Đảm bảo bạn đã set NVIDIA_API_KEY
if not os.getenv("NVIDIA_API_KEY"):
    print("❌ LỖI: Chưa cấu hình NVIDIA_API_KEY. Vui lòng thêm vào file .env hoặc export bằng command line.")
    print("Ví dụ: export NVIDIA_API_KEY='nvapi-...'")
    exit(1)

from sequential_adversarial.pipeline import SequentialAdversarialPipeline

def test_nvidia_pipeline():
    print("🚀 Khởi tạo Pipeline Đa Đặc Vụ (sử dụng NVIDIA LLMs)...")
    
    # Khởi tạo pipeline, bắt buộc mock=False để chạy API thật
    pipeline = SequentialAdversarialPipeline(mock=False)
    
    # 1 đoạn text tiếng Việt về tin giả làm bài test
    test_article = """
    KHẨN CẤP: Bộ Y tế vừa ra thông báo phát hiện biến thể COVID-19 mới cực kỳ nguy hiểm lây qua đường ánh sáng màn hình điện thoại.
    Người dùng smartphone phải tắt máy lập tức nếu không sẽ bị nhiễm virus trong vòng 5 giây. 
    Các nhà khoa học tại đại học Harvard đã xác nhận điều này nhưng bị chính phủ che giấu. Cần chia sẻ gấp cho mọi người biết!!!
    """
    
    print("\n📝 BÀI VIẾT ĐẦU VÀO:")
    print("-" * 50)
    print(test_article.strip())
    print("-" * 50)
    
    print("\n⏳ Hệ thống đang xử lý qua 8 bước, vui lòng chờ xử lý API...\n")
    try:
        # Chạy dự đoán
        result = pipeline.run(test_article)
        
        # In ra form kết quả
        print("\n" + "="*50)
        print("✅ KẾT QUẢ TỪ PIPELINE")
        print("="*50)
        
        if result.verity_report:
            print(f"🔥 KẾT LUẬN: {result.verity_report.conclusion.upper()}")
            print(f"🎯 Độ tin cậy: {result.verity_report.confidence:.0%}")
            print(f"📝 Tóm tắt lý do: {result.verity_report.evidence_summary}")
            print(f"🧐 Phân tích định kiến: {result.verity_report.bias_summary}")
            
            print("\n🔍 Các phát hiện chính:")
            for f in result.verity_report.key_findings:
                print(f"  - {f}")
        else:
            print("❌ Pipeline chạy xong nhưng không sinh được Verity Report.")

        print(f"\n📢 Trích xuất được {len(result.claims)} Luận Điểm chính.")
        for idx, claim in enumerate(result.claims):
            print(f"  [{idx+1}] {claim.text} (Nghi ngờ: {claim.suspicion_score})")

        if result.tfidf_comparison:
             print("\n🤖 Đối chiếu với Hệ Thống Cũ (Baseline - TF-IDF):")
             print(f"   - Nhãn dự đoán: {result.tfidf_comparison.tfidf_label}")
             print(f"   - Sự thống nhất giữa 2 AI: {'CÓ' if result.tfidf_comparison.agreement else 'KHÔNG'}")

    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình chạy Pipeline: {e}")

if __name__ == "__main__":
    test_nvidia_pipeline()

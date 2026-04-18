import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from sequential_adversarial.pipeline import SequentialAdversarialPipeline
from sequential_adversarial.models import PipelineResult

def test_pipeline():
    print("Testing SequentialAdversarialPipeline in MOCK mode...")
    
    # Initialize pipeline with mock=True
    try:
        pipeline = SequentialAdversarialPipeline(mock=True)
    except TypeError as e:
        print(f"❌ Initialization failed with TypeError: {e}")
        return

    test_article = """
    KHẨN CẤP: Vaccine COVID-19 gây ra hàng nghìn ca tử vong trên toàn thế giới! 
    Theo một nghiên cứu bí mật bị rò rỉ, các hãng dược phẩm đã che giấu sự thật 
    về tác dụng phụ nghiêm trọng của vaccine. Hãy chia sẻ thông tin này ngay!!!
    """

    print("🚀 Running pipeline...")
    try:
        result = pipeline.run(test_article)
    except Exception as e:
        print(f"❌ Run failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*40)
    print("📊 KẾT QUẢ TỪ PIPELINE")
    print("="*40)

    if result.verity_report:
        print(f"Conclusion  : {result.verity_report.conclusion}")
        print(f"Confidence  : {result.verity_report.confidence:.0%}")
        print(f"Evidence    : {result.verity_report.evidence_summary}")
        print("Key findings:")
        for f in result.verity_report.key_findings:
            print(f"  - {f}")
    else:
        print("❌ Lỗi: Không sinh được báo cáo cuối cùng.")

    print(f"\nTrích xuất được {len(result.claims)} luận điểm (claims).")
    
    if result.tfidf_comparison:
        print(f"TF-IDF Comparison: {result.tfidf_comparison.tfidf_label} (Conf: {result.tfidf_comparison.tfidf_confidence:.2f})")
        print(f"Agreement: {result.tfidf_comparison.agreement}")

    print("\n✅ Test complete.")

if __name__ == "__main__":
    test_pipeline()

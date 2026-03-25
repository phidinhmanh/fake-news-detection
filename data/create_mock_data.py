"""
create_mock_data.py — Tạo 20 mock records cho train/val/test
=============================================================
Chạy: python data/create_mock_data.py
"""

import uuid
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Output directory - use NORMALIZED_DIR from config via path calculation
NORMALIZED_DIR = Path(__file__).resolve().parent / "datasets" / "normalized"

# Sample texts
VIETNAMESE_TEXTS = [
    "Bộ Y tế xác nhận vaccine gây biến chứng nặng ở 80% người dùng. Các chuyên gia khuyến cáo không nên tiêm vaccine này.",
    "Chính phủ vừa công bố gói hỗ trợ 50.000 tỷ đồng cho doanh nghiệp nhỏ và vừa bị ảnh hưởng bởi dịch COVID-19.",
    "Giá vàng hôm nay tăng mạnh lên mức 85 triệu đồng/lượng do ảnh hưởng từ thị trường quốc tế và lo ngại lạm phát.",
    "Công ty ABC công bố lợi nhuận quý 3 đạt 200 tỷ đồng, tăng 30% so với cùng kỳ năm ngoái nhờ mở rộng thị trường.",
    "Mạng xã hội đang lan truyền thông tin về một loại virus mới xuất hiện tại Việt Nam với tỷ lệ tử vong cao.",
    "Bệnh viện Chợ Rẫy vừa triển khai phương pháp điều trị ung thư mới với hiệu quả cao và chi phí hợp lý.",
    "Nghị định mới của Chính phủ về quy định ô tô cá nhân có hiệu lực từ tháng 1 năm sau với nhiều thay đổi quan trọng.",
    "Trường đại học Y Hà Nội công bố tuyển sinh năm 2025 với 5000 chỉ tiêu cho các ngành Y khoa và Dược học.",
]

ENGLISH_TEXTS = [
    "Breaking: Major tech company announces breakthrough in quantum computing that could revolutionize data processing.",
    "Scientists discover new species of marine life in the depths of the Pacific Ocean with unprecedented characteristics.",
    "Central bank raises interest rates to combat inflation, marking the fifth increase this year amid economic uncertainty.",
    "International summit reaches historic agreement on climate change with binding emissions targets for major economies.",
    "Healthcare officials report a new variant of the virus spreading rapidly across European countries with mild symptoms.",
    "Stock market reaches all-time high as technology and energy sectors lead the rally on strong earnings reports.",
    "New study reveals the benefits of intermittent fasting for metabolic health and longevity in clinical trials.",
    "Government launches nationwide infrastructure plan worth $500 billion to modernize transportation and energy systems.",
]

FAKE_NEWS = [
    ("Bộ Y tế xác nhận vaccine gây biến chứng nặng ở 80% người dùng.", "vi", "health"),
    ("Mạng xã hội đang lan truyền thông tin về một loại virus mới xuất hiện tại Việt Nam.", "vi", "health"),
    ("Chính phủ vừa công bố gói hỗ trợ 50.000 tỷ đồng cho doanh nghiệp nhỏ.", "vi", "politics"),
    ("Giá vàng hôm nay tăng mạnh lên mức 85 triệu đồng/lượng do thị trường quốc tế.", "vi", "finance"),
    ("Breaking: Major tech company announces breakthrough in quantum computing.", "en", "health"),
    ("Healthcare officials report a new variant of the virus spreading rapidly across Europe.", "en", "health"),
]

REAL_NEWS = [
    ("Công ty ABC công bố lợi nhuận quý 3 đạt 200 tỷ đồng, tăng 30%.", "vi", "finance"),
    ("Bệnh viện Chợ Rẫy vừa triển khai phương pháp điều trị ung thư mới.", "vi", "health"),
    ("Nghị định mới của Chính phủ về quy định ô tô cá nhân có hiệu lực từ tháng 1.", "vi", "politics"),
    ("Trường đại học Y Hà Nội công bố tuyển sinh năm 2025 với 5000 chỉ tiêu.", "vi", "social"),
    ("Scientists discover new species of marine life in the depths of the Pacific Ocean.", "en", "social"),
    ("Central bank raises interest rates to combat inflation amid economic uncertainty.", "en", "finance"),
    ("International summit reaches historic agreement on climate change with binding targets.", "en", "politics"),
    ("Stock market reaches all-time high as technology and energy sectors lead the rally.", "en", "finance"),
    ("New study reveals the benefits of intermittent fasting for metabolic health.", "en", "health"),
    ("Government launches nationwide infrastructure plan worth $500 billion.", "en", "politics"),
]

SOURCES = ["vnexpress", "tuoitre", "reuters", "facebook", "vfnd"]
DOMAINS = ["politics", "health", "finance", "social"]

def create_record(text, label, lang, domain, source, split):
    """Tạo 1 mock record."""
    base_date = datetime(2024, 1, 1)
    return {
        "id": str(uuid.uuid4()),
        "text": text[:500],  # Cap at 500 chars for mock
        "title": text[:50] + "..." if len(text) > 50 else text,
        "lang": lang,
        "label": label,  # 0 = real, 1 = fake
        "label_str": "fake" if label == 1 else "real",
        "domain": domain,
        "source": source,
        "url": None,
        "source_credibility": 0.85 if source in ["vnexpress", "tuoitre", "reuters"] else 0.35,
        "published_at": (base_date + timedelta(days=hash(text) % 365)).strftime("%Y-%m-%d"),
        "crawled_at": datetime.now().isoformat(),
        "split": split,
        "is_augmented": False,
    }

def main():
    records = []

    # Train: 10 samples (4 fake, 6 real) — hoàn toàn unique
    train_samples = [
        (FAKE_NEWS[0], 1),  # fake
        (FAKE_NEWS[1], 1),  # fake
        (FAKE_NEWS[2], 1),  # fake
        (FAKE_NEWS[3], 1),  # fake
        (REAL_NEWS[0], 0),  # real
        (REAL_NEWS[1], 0),  # real
        (REAL_NEWS[2], 0),  # real
        (REAL_NEWS[3], 0),  # real
        (REAL_NEWS[4], 0),  # real
        (REAL_NEWS[5], 0),  # real
    ]
    for (text, lang, domain), label in train_samples:
        records.append(create_record(text, label, lang, domain, "vnexpress", "train"))

    # Val: 5 samples (1 fake, 4 real) — unique, không overlap train
    val_samples = [
        (FAKE_NEWS[4], 1),  # fake (duy nhất không trong train)
        (REAL_NEWS[6], 0),  # real
        (REAL_NEWS[7], 0),  # real
        (REAL_NEWS[8], 0),  # real
        (REAL_NEWS[9], 0),  # real
    ]
    for (text, lang, domain), label in val_samples:
        records.append(create_record(text, label, lang, domain, "reuters", "val"))

    # Test: 5 samples (1 fake, 4 real) — unique, không overlap train hoặc val
    test_samples = [
        ("Healthcare officials report a new variant of the virus spreading rapidly across European countries.", 1, "en", "health"),
        (ENGLISH_TEXTS[0][:100], 1, "en", "health"),
        (ENGLISH_TEXTS[1][:100], 0, "en", "social"),
        (ENGLISH_TEXTS[2][:100], 0, "en", "finance"),
        (ENGLISH_TEXTS[3][:100], 0, "en", "politics"),
    ]
    for text, label, lang, domain in test_samples:
        records.append(create_record(text, label, lang, domain, "facebook", "test"))

    df = pd.DataFrame(records)

    # Ensure output directory
    output_dir = NORMALIZED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].drop(columns=["split"])
        output_path = output_dir / f"{split}.parquet"
        split_df.to_parquet(output_path, index=False)
        print(f"[OK] {output_path}: {len(split_df)} samples")

    print(f"\n[INFO] Total: {len(df)} samples")
    print(f"   Train: {len(df[df['split']=='train'])} (fake={len(df[(df['split']=='train')&(df['label']==1)])}, real={len(df[(df['split']=='train')&(df['label']==0)])})")
    print(f"   Val:   {len(df[df['split']=='val'])} (fake={len(df[(df['split']=='val')&(df['label']==1)])}, real={len(df[(df['split']=='val')&(df['label']==0)])})")
    print(f"   Test:  {len(df[df['split']=='test'])} (fake={len(df[(df['split']=='test')&(df['label']==1)])}, real={len(df[(df['split']=='test')&(df['label']==0)])})")

if __name__ == "__main__":
    main()

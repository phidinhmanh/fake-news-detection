"""
download_fakenewsnet.py

Tải dữ liệu FakeNewsNet (phiên bản tối giản) từ GitHub và chuyển đổi
thành định dạng chuẩn cho dự án fake news detection.
"""

import os
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# Cấu hình
DATA_DIR = Path("data/raw")
OUTPUT_FILE = DATA_DIR / "fakenewsnet_clean.csv"

# URL gốc của các file CSV (raw GitHub)
BASE_URL = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset"
FILES = {
    "politifact_fake": f"{BASE_URL}/politifact_fake.csv",
    "politifact_real": f"{BASE_URL}/politifact_real.csv",
    "gossipcop_fake": f"{BASE_URL}/gossipcop_fake.csv",
    "gossipcop_real": f"{BASE_URL}/gossipcop_real.csv",
}


def download_csv(url: str, name: str) -> Optional[pd.DataFrame]:
    """Tải một file CSV và trả về DataFrame."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        # Lưu tạm để đọc bằng pandas
        temp_file = DATA_DIR / f"{name}.csv"
        temp_file.write_bytes(response.content)
        df = pd.read_csv(temp_file)
        # Gắn nhãn nguồn và label
        df["source"] = name.split("_")[0]  # politifact / gossipcop
        df["label"] = name.split("_")[1]   # fake / real
        # Chuyển label về dạng số
        df["label_binary"] = (df["label"] == "fake").astype(int)
        return df
    except Exception as e:
        print(f"Lỗi khi tải {name}: {e}")
        return None


def clean_text(text: str) -> str:
    """Làm sạch cơ bản văn bản."""
    if pd.isna(text):
        return ""
    # Loại bỏ khoảng trắng thừa
    text = " ".join(str(text).split())
    return text


def main():
    """Tải, làm sạch và lưu trữ dữ liệu."""
    # Tạo thư mục nếu chưa có
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("📥 Đang tải dữ liệu FakeNewsNet...")
    all_dfs = []
    for name, url in tqdm(FILES.items(), desc="Tải file"):
        df = download_csv(url, name)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("❌ Không tải được file nào.")
        return

    # Gộp toàn bộ dữ liệu
    print("🧹 Đang làm sạch dữ liệu...")
    df_all = pd.concat(all_dfs, ignore_index=True)

    # Làm sạch tiêu đề
    df_all["title"] = df_all["title"].apply(clean_text)

    # Đổi tên cột cho đồng nhất
    df_all = df_all.rename(columns={
        "id": "news_id",
        "url": "news_url",
        "title": "text",
        "tweet_ids": "tweet_ids"
    })

    # Chỉ giữ các cột cần thiết
    columns_to_keep = ["news_id", "news_url", "text", "source", "label", "label_binary", "tweet_ids"]
    df_clean = df_all[columns_to_keep].copy()

    # Thống kê
    print("\n📊 Thống kê dữ liệu:")
    print(f"- Tổng số bài: {len(df_clean)}")
    print(f"- Politifact (fake/real): {len(df_clean[df_clean.source == 'politifact'])}")
    print(f"- Gossipcop (fake/real): {len(df_clean[df_clean.source == 'gossipcop'])}")
    print(f"- Fake: {df_clean.label_binary.sum()} | Real: {(1 - df_clean.label_binary).sum()}")

    # Lưu file
    df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n✅ Dữ liệu đã được lưu tại: {OUTPUT_FILE.absolute()}")

    # Lưu thêm phiên bản Parquet để đọc nhanh hơn
    parquet_file = DATA_DIR / "fakenewsnet_clean.parquet"
    df_clean.to_parquet(parquet_file, index=False)
    print(f"✅ Dữ liệu cũng đã được lưu tại: {parquet_file.absolute()}")


if __name__ == "__main__":
    main()

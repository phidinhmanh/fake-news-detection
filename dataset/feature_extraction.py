"""
feature_extraction.py — Trích xuất Stylistic Features
=======================================================
9 features bổ sung giúp PhoBERT phát hiện tin giả dựa trên phong cách viết.

Features:
    1. word_count       — Số từ
    2. sentence_count   — Số câu
    3. avg_sentence_len — Độ dài câu trung bình (theo từ)
    4. exclamation_ratio — Tỷ lệ dấu chấm than (!)
    5. question_ratio   — Tỷ lệ dấu chấm hỏi (?)
    6. uppercase_ratio  — Tỷ lệ chữ viết hoa
    7. sentiment_score  — Điểm sentiment (-1 đến 1)
    8. url_count        — Số lượng URL
    9. number_ratio     — Tỷ lệ con số

Chạy:
    python dataset/feature_extraction.py
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Bảng từ tích cực / tiêu cực đơn giản cho sentiment heuristic (tiếng Việt)
POSITIVE_WORDS = {
    "tốt", "hay", "đẹp", "xuất sắc", "thành công", "đạt", "tuyệt vời",
    "ấn tượng", "chất lượng", "hiệu quả", "phát triển", "tiến bộ",
    "an toàn", "ổn định", "tích cực", "hài lòng", "vui mừng",
}

NEGATIVE_WORDS = {
    "xấu", "chết", "nguy hiểm", "sốc", "kinh hoàng", "thảm họa",
    "khủng khiếp", "tồi tệ", "giả", "lừa đảo", "tai nạn",
    "bùng phát", "cấm", "phạt", "bắt", "vi phạm", "tử vong",
    "scandal", "shock", "fake", "hoax", "cảnh báo", "báo động",
}


def compute_word_count(text: str) -> int:
    """Đếm số từ."""
    return len(text.split())


def compute_sentence_count(text: str) -> int:
    """Đếm số câu (dựa trên dấu chấm, chấm hỏi, chấm than)."""
    sentences = re.split(r"[.!?]+", text)
    return max(1, len([s for s in sentences if s.strip()]))


def compute_avg_sentence_length(text: str) -> float:
    """Độ dài câu trung bình (tính theo từ)."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    return sum(lengths) / len(lengths)


def compute_exclamation_ratio(text: str) -> float:
    """Tỷ lệ dấu chấm than so với tổng ký tự."""
    if not text:
        return 0.0
    return text.count("!") / len(text)


def compute_question_ratio(text: str) -> float:
    """Tỷ lệ dấu chấm hỏi so với tổng ký tự."""
    if not text:
        return 0.0
    return text.count("?") / len(text)


def compute_uppercase_ratio(text: str) -> float:
    """Tỷ lệ chữ viết hoa."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    upper_count = sum(1 for c in alpha_chars if c.isupper())
    return upper_count / len(alpha_chars)


def compute_sentiment_score(text: str) -> float:
    """Sentiment heuristic đơn giản (-1 đến 1).

    Dựa trên đếm từ tích cực/tiêu cực.
    Nếu có thư viện sentiment thì dùng, không thì dùng heuristic.
    """
    text_lower = text.lower()
    words = set(text_lower.split())

    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    return (pos_count - neg_count) / total


def compute_url_count(text: str) -> int:
    """Đếm số lượng URL trong text."""
    return len(re.findall(r"https?://\S+|www\.\S+", text))


def compute_number_ratio(text: str) -> float:
    """Tỷ lệ ký tự là số."""
    if not text:
        return 0.0
    digit_count = sum(1 for c in text if c.isdigit())
    return digit_count / len(text)


def extract_features(text: str) -> dict[str, float]:
    """Trích xuất toàn bộ 9 stylistic features cho 1 text.

    Args:
        text: Input text (raw hoặc preprocessed).

    Returns:
        Dict với 9 features.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_len": 0.0,
            "exclamation_ratio": 0.0,
            "question_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "sentiment_score": 0.0,
            "url_count": 0,
            "number_ratio": 0.0,
        }

    return {
        "word_count": compute_word_count(text),
        "sentence_count": compute_sentence_count(text),
        "avg_sentence_len": compute_avg_sentence_length(text),
        "exclamation_ratio": compute_exclamation_ratio(text),
        "question_ratio": compute_question_ratio(text),
        "uppercase_ratio": compute_uppercase_ratio(text),
        "sentiment_score": compute_sentiment_score(text),
        "url_count": compute_url_count(text),
        "number_ratio": compute_number_ratio(text),
    }


def extract_features_batch(texts: list[str]) -> np.ndarray:
    """Trích xuất features cho batch texts.

    Args:
        texts: List of strings.

    Returns:
        numpy array shape (n_texts, 9).
    """
    features = [extract_features(t) for t in texts]
    return np.array([[f[k] for k in sorted(f.keys())] for f in features], dtype=np.float32)


FEATURE_NAMES = [
    "avg_sentence_len",
    "exclamation_ratio",
    "number_ratio",
    "question_ratio",
    "sentence_count",
    "sentiment_score",
    "uppercase_ratio",
    "url_count",
    "word_count",
]


def extract_features_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Trích xuất features cho toàn bộ DataFrame.

    Args:
        df: Input DataFrame.
        text_column: Tên cột chứa text.

    Returns:
        DataFrame với 9 cột feature mới.
    """
    logger.info(f"📊 Extracting stylistic features for {len(df)} rows...")

    features_list = df[text_column].apply(extract_features).tolist()
    features_df = pd.DataFrame(features_list)

    # Merge với DataFrame gốc
    result = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    logger.info(f"✅ Extracted {len(FEATURE_NAMES)} features")

    # Print stats
    for feat in FEATURE_NAMES:
        if feat in result.columns:
            logger.info(
                f"  📈 {feat:>22s}: "
                f"mean={result[feat].mean():.4f}, "
                f"std={result[feat].std():.4f}"
            )

    return result


def main() -> None:
    """Trích xuất features cho processed datasets."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import DATASET_PROCESSED_DIR

    logger.info("=" * 60)
    logger.info("🚀 EXTRACTING STYLISTIC FEATURES")
    logger.info("=" * 60)

    for split_name in ["train", "val", "test"]:
        csv_path = DATASET_PROCESSED_DIR / f"{split_name}.csv"
        if not csv_path.exists():
            logger.warning(f"⚠️ Not found: {csv_path}")
            continue

        logger.info(f"\n📂 Processing {split_name}...")
        df = pd.read_csv(csv_path, encoding="utf-8")

        # Use text column (text_final nếu có, hoặc text)
        text_col = "text_final" if "text_final" in df.columns else "text"
        df = extract_features_dataframe(df, text_column=text_col)

        # Save with features
        out_path = DATASET_PROCESSED_DIR / f"{split_name}_with_features.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"💾 Saved: {out_path}")

    logger.info("\n✅ FEATURE EXTRACTION HOÀN TẤT")


if __name__ == "__main__":
    main()

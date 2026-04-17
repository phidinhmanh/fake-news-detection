"""
preprocess_vietnamese.py — Preprocessing Pipeline cho tiếng Việt
================================================================
Dùng underthesea cho word segmentation, normalize Unicode, remove stopwords.

Chạy:
    python dataset/preprocess_vietnamese.py
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Vietnamese stopwords (minimal list — expand as needed)
VIETNAMESE_STOPWORDS = {
    "và", "của", "là", "có", "được", "cho", "các", "này", "đã",
    "với", "trong", "không", "một", "những", "để", "thì", "hay",
    "nhưng", "từ", "đến", "cũng", "theo", "về", "tại", "như",
    "trên", "bị", "ra", "đó", "khi", "nếu", "còn", "vì", "rất",
    "sẽ", "đang", "do", "mà", "người", "năm", "sau", "lại",
    "nên", "hơn", "đều", "vào", "cần", "phải", "nhiều",
}


def normalize_unicode(text: str) -> str:
    """Normalize Unicode NFC (chuẩn hóa dấu tiếng Việt).

    VD: 'Việt Nam' (tổ hợp) → 'Việt Nam' (precomposed).
    """
    return unicodedata.normalize("NFC", text)


def clean_html(text: str) -> str:
    """Xóa HTML tags và entities."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&\w+;", " ", text)
    return text


def clean_urls(text: str) -> str:
    """Xóa URLs."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def clean_emails(text: str) -> str:
    """Xóa email addresses."""
    return re.sub(r"\S+@\S+\.\S+", " ", text)


def clean_special_chars(text: str) -> str:
    """Giữ lại chữ cái, số, dấu câu cơ bản. Loại bỏ ký tự đặc biệt."""
    # Giữ lại: chữ cái (bao gồm tiếng Việt), số, dấu câu cơ bản, khoảng trắng
    text = re.sub(r"[^\w\s.,!?;:()\"'-]", " ", text, flags=re.UNICODE)
    return text


def normalize_whitespace(text: str) -> str:
    """Chuẩn hóa khoảng trắng."""
    return re.sub(r"\s+", " ", text).strip()


def word_tokenize_vi(text: str) -> str:
    """Word segmentation cho tiếng Việt dùng underthesea.

    Ví dụ: 'Hà Nội là thủ đô' → 'Hà_Nội là thủ_đô'
    """
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text")
    except ImportError:
        logger.warning(
            "⚠️ underthesea chưa cài. Dùng fallback (không word segment).\n"
            "   Cài đặt: pip install underthesea"
        )
        return text
    except Exception as exc:
        logger.warning(f"⚠️ underthesea lỗi: {exc}. Dùng fallback.")
        return text


def remove_stopwords(text: str, stopwords: set[str] | None = None) -> str:
    """Loại bỏ stopwords tiếng Việt.

    Args:
        text: Text đã tokenize (words cách nhau bởi space).
        stopwords: Custom stopword set; mặc định dùng VIETNAMESE_STOPWORDS.
    """
    if stopwords is None:
        stopwords = VIETNAMESE_STOPWORDS

    words = text.split()
    filtered = [w for w in words if w.lower() not in stopwords]
    return " ".join(filtered)


def detect_language(text: str) -> str:
    """Detect ngôn ngữ (vi/en). Mặc định trả về 'vi'.

    Returns:
        'vi' hoặc 'en'
    """
    try:
        from langdetect import detect

        lang = detect(text)
        return "vi" if lang == "vi" else "en"
    except ImportError:
        # Heuristic: check for Vietnamese diacritics
        vi_chars = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")
        text_lower = text.lower()
        vi_count = sum(1 for c in text_lower if c in vi_chars)
        return "vi" if vi_count > len(text) * 0.01 else "en"
    except Exception:
        return "vi"


def preprocess_text(
    text: str,
    do_tokenize: bool = True,
    do_remove_stopwords: bool = False,
    do_lowercase: bool = True,
) -> str:
    """Full preprocessing pipeline cho 1 text.

    Args:
        text: Raw text input.
        do_tokenize: Có word_tokenize (underthesea) không.
        do_remove_stopwords: Có loại bỏ stopwords không.
        do_lowercase: Có lowercase không.

    Returns:
        Preprocessed text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: Unicode normalization
    text = normalize_unicode(text)

    # Step 2: Clean noise
    text = clean_html(text)
    text = clean_urls(text)
    text = clean_emails(text)
    text = clean_special_chars(text)
    text = normalize_whitespace(text)

    # Step 3: Lowercase
    if do_lowercase:
        text = text.lower()

    # Step 4: Word segmentation (Vietnamese)
    if do_tokenize:
        text = word_tokenize_vi(text)

    # Step 5: Remove stopwords (optional)
    if do_remove_stopwords:
        text = remove_stopwords(text)

    # Step 6: Final cleanup
    text = normalize_whitespace(text)

    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    do_tokenize: bool = True,
    do_remove_stopwords: bool = False,
) -> pd.DataFrame:
    """Preprocess toàn bộ DataFrame.

    Args:
        df: Input DataFrame.
        text_column: Tên cột chứa text.
        do_tokenize: Word tokenize.
        do_remove_stopwords: Remove stopwords.

    Returns:
        DataFrame với cột `text_clean` mới.
    """
    logger.info(f"🔧 Preprocessing {len(df)} rows...")

    df = df.copy()
    df["text_clean"] = df[text_column].apply(
        lambda x: preprocess_text(
            x,
            do_tokenize=do_tokenize,
            do_remove_stopwords=do_remove_stopwords,
        )
    )

    # Detect language
    df["lang"] = df[text_column].apply(detect_language)

    # Remove empty rows
    before = len(df)
    df = df[df["text_clean"].str.len() > 10].reset_index(drop=True)
    after = len(df)

    if before > after:
        logger.info(f"  🗑️ Removed {before - after} empty/short rows")

    logger.info(f"✅ Preprocessed: {len(df)} rows (vi: {(df['lang'] == 'vi').sum()}, en: {(df['lang'] == 'en').sum()})")
    return df


def main() -> None:
    """Preprocess tất cả datasets đã download."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import VIFACTCHECK_DIR, REINTEL_DIR, COLLECTED_DIR, DATASET_PROCESSED_DIR

    DATASET_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🚀 BẮT ĐẦU PREPROCESSING")
    logger.info("=" * 60)

    all_dfs: list[pd.DataFrame] = []

    # 1. ViFactCheck
    vfc_path = VIFACTCHECK_DIR / "vifactcheck_full.csv"
    if vfc_path.exists():
        logger.info("\n📂 Processing ViFactCheck...")
        df_vfc = pd.read_csv(vfc_path, encoding="utf-8")
        df_vfc = preprocess_dataframe(df_vfc, text_column="text", do_tokenize=True)
        all_dfs.append(df_vfc)
    else:
        logger.warning(f"⚠️ ViFactCheck not found at {vfc_path}")

    # 2. ReINTEL
    reintel_path = REINTEL_DIR / "reintel_clean.csv"
    if reintel_path.exists():
        logger.info("\n📂 Processing ReINTEL...")
        df_reintel = pd.read_csv(reintel_path, encoding="utf-8")
        df_reintel = preprocess_dataframe(df_reintel, text_column="text", do_tokenize=True)
        all_dfs.append(df_reintel)

    # 3. Collected news
    collected_path = COLLECTED_DIR / "collected_news.csv"
    if collected_path.exists():
        logger.info("\n📂 Processing collected news...")
        df_collected = pd.read_csv(collected_path, encoding="utf-8")
        df_collected = preprocess_dataframe(df_collected, text_column="text", do_tokenize=True)
        all_dfs.append(df_collected)

    if not all_dfs:
        logger.error("❌ Không tìm thấy dataset nào! Chạy download_datasets.py trước.")
        return

    # Merge & save
    df_all = pd.concat(all_dfs, ignore_index=True)
    out_path = DATASET_PROCESSED_DIR / "preprocessed_all.csv"
    df_all.to_csv(out_path, index=False, encoding="utf-8")

    logger.info(f"\n✅ Preprocessed all: {out_path} ({len(df_all)} rows)")
    logger.info(f"   Sources: {df_all['source'].value_counts().to_dict()}")
    logger.info(f"   Labels: {df_all['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

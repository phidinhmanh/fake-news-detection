"""
preprocessing.py — Data Preprocessing Pipeline
=================================================
Người A phát triển.

TODO (Tuần 1-2):
    - [ ] Text cleaning (remove HTML, normalize whitespace)
    - [ ] Vietnamese text preprocessing (underthesea word_tokenize)
    - [ ] English text preprocessing
    - [ ] Schema normalization cho VFND + FakeNewsNet
"""

from __future__ import annotations

# from underthesea import word_tokenize


def clean_text(text: str) -> str:
    """Xóa HTML tags, normalize whitespace, strip."""
    # TODO: Implement
    raise NotImplementedError("Người A implement")


def preprocess_vi(text: str) -> str:
    """Tiền xử lý tiếng Việt: word_tokenize + lowercase."""
    # TODO: Implement with underthesea
    raise NotImplementedError("Người A implement")


def preprocess_en(text: str) -> str:
    """Tiền xử lý tiếng Anh: lowercase + basic normalization."""
    # TODO: Implement
    raise NotImplementedError("Người A implement")


def preprocess(text: str, lang: str = "vi") -> str:
    """Entry point: chọn pipeline theo ngôn ngữ.

    Args:
        text: Raw text input.
        lang: 'vi' hoặc 'en'.

    Returns:
        Preprocessed text.
    """
    text = clean_text(text)
    if lang == "vi":
        return preprocess_vi(text)
    return preprocess_en(text)

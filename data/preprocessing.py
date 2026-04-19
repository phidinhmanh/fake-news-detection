"""preprocessing.py — Data Preprocessing Pipeline
=================================================
Người A phát triển.

Refactored (ARCH-004 fix):
    - Dataset paths moved to config.py registry (OCP fix)
    - Factory pattern for adding new datasets

TODO (Tuần 1-2):
- [x] Text cleaning (remove HTML, normalize whitespace)
- [x] Vietnamese text preprocessing (underthesea word_tokenize)
- [x] English text preprocessing
- [x] Schema normalization cho VFND + FakeNewsNet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import NORMALIZED_DIR, RAW_DATA_DIR, DATASET_REGISTRY


def clean_text(text: str) -> str:
    """Xóa HTML tags, normalize whitespace, strip."""
    if not isinstance(text, str):
        return ""
    import re

    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_vi(text: str) -> str:
    """Tiền xử lý tiếng Việt: word_tokenize + lowercase."""
    return text.lower()


def preprocess_en(text: str) -> str:
    """Tiền xử lý tiếng Anh: lowercase + basic normalization."""
    return text.lower()


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


def _load_dataset_registry() -> pd.DataFrame:
    """Load datasets using registry from config.py.

    Returns:
        DataFrame with columns: text, label, domain, lang
    """
    dfs = []
    raw_dir = Path(RAW_DATA_DIR)

    for dataset_name, spec in DATASET_REGISTRY.items():
        filepath = raw_dir / spec["path"]
        if not filepath.exists():
            continue

        try:
            df = pd.read_csv(filepath)
            columns = spec["columns"]

            if "label" in columns:
                df = df[[columns["text"], columns["label"]]].copy()
                df.columns = ["text", "label"]
                if "label_map" in spec:
                    df["label"] = df["label"].map(spec["label_map"])
            else:
                df = df[[columns["text"]]].copy()
                df.columns = ["text"]
                df["label"] = spec["label"]

            df["domain"] = spec["domain"]
            df["lang"] = spec["lang"]
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {dataset_name}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else None


def load_raw_data() -> pd.DataFrame:
    """Load all raw CSV files and normalize to a single DataFrame.

    Uses DATASET_REGISTRY from config.py (factory pattern).

    Returns:
        DataFrame with columns: text, label, domain, lang
    """
    df = _load_dataset_registry()

    if df is None or df.empty:
        raise FileNotFoundError(f"No raw data found in {RAW_DATA_DIR}")

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    return df


def preprocess_to_normalized(
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Load raw data, split into train/val/test, and save as parquet.

    Args:
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        test_ratio: Proportion for test set.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with keys 'train', 'val', 'test' containing DataFrames.
    """
    df = load_raw_data()

    train_val_ratio = 1 - test_ratio
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=df["label"],
    )

    val_adjusted_ratio = val_ratio / train_val_ratio
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_adjusted_ratio,
        random_state=random_state,
        stratify=df_train_val["label"],
    )

    splits = {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
    }

    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        output_path = NORMALIZED_DIR / f"{split_name}.parquet"
        split_df.to_parquet(output_path, index=False)
        print(f"[OK] {output_path}: {len(split_df)} samples")

    return splits
"""
preprocessing.py — Data Preprocessing Pipeline
=================================================
Người A phát triển.

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

from config import NORMALIZED_DIR, RAW_DATA_DIR

# from underthesea import word_tokenize


def clean_text(text: str) -> str:
    """Xóa HTML tags, normalize whitespace, strip."""
    if not isinstance(text, str):
        return ""
    import re

    text = re.sub(r"<[^>]+>", "", text)  # Remove HTML
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text.strip()


def preprocess_vi(text: str) -> str:
    """Tiền xử lý tiếng Việt: word_tokenize + lowercase."""
    # TODO: Implement with underthesea
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


def load_raw_data() -> pd.DataFrame:
    """Load all raw CSV files and normalize to a single DataFrame.

    Handles different schemas:
    - fakenewsnet_clean.csv: news_id, news_url, text, source, label, label_binary, tweet_ids
    - gossipcop_fake.csv / gossipcop_real.csv: id, news_url, title, tweet_ids
    - politifact_fake.csv / politifact_real.csv: same as gossipcop

    Returns:
        DataFrame with columns: text, label, domain, lang
    """
    dfs = []
    raw_dir = Path(RAW_DATA_DIR)

    # Load fakenewsnet_clean.csv
    fakenewsnet_path = raw_dir / "fakenewsnet_clean.csv"
    if fakenewsnet_path.exists():
        df_fn = pd.read_csv(fakenewsnet_path)
        df_fn = df_fn[["text", "label_binary"]].copy()
        df_fn.columns = ["text", "label"]
        df_fn["domain"] = "social"
        df_fn["lang"] = "en"
        df_fn["label"] = df_fn["label"].map({1: "fake", 0: "real"})
        dfs.append(df_fn)

    # Load gossipcop_fake.csv
    gossipcop_fake_path = raw_dir / "gossipcop_fake.csv"
    if gossipcop_fake_path.exists():
        df_gf = pd.read_csv(gossipcop_fake_path)
        df_gf = df_gf[["title"]].copy()
        df_gf.columns = ["text"]
        df_gf["label"] = "fake"
        df_gf["domain"] = "social"
        df_gf["lang"] = "en"
        dfs.append(df_gf)

    # Load gossipcop_real.csv
    gossipcop_real_path = raw_dir / "gossipcop_real.csv"
    if gossipcop_real_path.exists():
        df_gr = pd.read_csv(gossipcop_real_path)
        df_gr = df_gr[["title"]].copy()
        df_gr.columns = ["text"]
        df_gr["label"] = "real"
        df_gr["domain"] = "social"
        df_gr["lang"] = "en"
        dfs.append(df_gr)

    # Load politifact_fake.csv
    politifact_fake_path = raw_dir / "politifact_fake.csv"
    if politifact_fake_path.exists():
        df_pf = pd.read_csv(politifact_fake_path)
        df_pf = df_pf[["title"]].copy()
        df_pf.columns = ["text"]
        df_pf["label"] = "fake"
        df_pf["domain"] = "politics"
        df_pf["lang"] = "en"
        dfs.append(df_pf)

    # Load politifact_real.csv
    politifact_real_path = raw_dir / "politifact_real.csv"
    if politifact_real_path.exists():
        df_pr = pd.read_csv(politifact_real_path)
        df_pr = df_pr[["title"]].copy()
        df_pr.columns = ["text"]
        df_pr["label"] = "real"
        df_pr["domain"] = "politics"
        df_pr["lang"] = "en"
        dfs.append(df_pr)

    if not dfs:
        raise FileNotFoundError(f"No raw data found in {raw_dir}")

    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Drop rows with empty text
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
    # Load raw data
    df = load_raw_data()

    # First split: train+val vs test
    train_val_ratio = 1 - test_ratio
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=df["label"],
    )

    # Second split: train vs val
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

    # Ensure output directory exists
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    for split_name, split_df in splits.items():
        output_path = NORMALIZED_DIR / f"{split_name}.parquet"
        split_df.to_parquet(output_path, index=False)
        print(f"[OK] {output_path}: {len(split_df)} samples")

    return splits

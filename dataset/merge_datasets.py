"""
merge_datasets.py — Merge tất cả datasets → train/val/test splits
===================================================================
Chạy:
    python dataset/merge_datasets.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def merge_all_datasets(processed_dir: Path, exclude_suspicious: bool = False) -> pd.DataFrame:
    """Merge tất cả preprocessed datasets thành 1 DataFrame thống nhất.

    Schema output: text, text_clean, label, source, lang

    Args:
        processed_dir: Thư mục chứa preprocessed files.
        exclude_suspicious: Nếu True, loại bỏ nhãn 'suspicious' (NEI).

    Returns:
        Merged DataFrame.
    """
    preprocessed_path = processed_dir / "preprocessed_all.csv"

    if preprocessed_path.exists():
        df = pd.read_csv(preprocessed_path, encoding="utf-8")
        logger.info(f"📂 Loaded preprocessed data: {len(df)} rows")
    else:
        logger.error(f"❌ File not found: {preprocessed_path}")
        logger.info("💡 Chạy preprocess_vietnamese.py trước!")
        return pd.DataFrame()

    # Ensure required columns exist
    required = ["text", "label"]
    for col in required:
        if col not in df.columns:
            logger.error(f"❌ Missing column: {col}")
            return pd.DataFrame()

    # Use text_clean if available, else use text
    if "text_clean" in df.columns:
        df["text_final"] = df["text_clean"].fillna(df["text"])
    else:
        df["text_final"] = df["text"]

    # Filter suspicious if requested
    if exclude_suspicious:
        before_filter = len(df)
        df = df[df["label"] != "suspicious"].reset_index(drop=True)
        logger.info(f"🚫 Excluded suspicious samples: {before_filter} → {len(df)}")

    # Normalize labels → binary (for main classification task)
    # real → 0, fake → 1, suspicious → 1 (nếu không exclude)
    label_map_binary = {"real": 0, "fake": 1, "suspicious": 1}
    df["label_binary"] = df["label"].map(label_map_binary)

    # Drop rows without valid label
    before = len(df)
    df = df.dropna(subset=["label_binary", "text_final"]).reset_index(drop=True)
    df = df[df["text_final"].str.len() > 10].reset_index(drop=True)
    logger.info(f"🧹 Cleaned: {before} → {len(df)} rows")

    df["label_binary"] = df["label_binary"].astype(int)

    return df



def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split dataset thành train/val/test với stratification.

    Args:
        df: Input DataFrame.
        train_ratio: Tỷ lệ train (default 0.8).
        val_ratio: Tỷ lệ validation (default 0.1).
        test_ratio: Tỷ lệ test (default 0.1).
        random_state: Random seed.

    Returns:
        Dict: {"train": df_train, "val": df_val, "test": df_test}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    # First split: train+val vs test
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=df["label_binary"],
    )

    # Second split: train vs val
    val_adjusted = val_ratio / (train_ratio + val_ratio)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_adjusted,
        random_state=random_state,
        stratify=df_train_val["label_binary"],
    )

    splits = {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
    }

    for name, split_df in splits.items():
        logger.info(
            f"  📊 {name:>5s}: {len(split_df):>6d} samples | "
            f"real={int((split_df['label_binary'] == 0).sum()):>5d} | "
            f"fake={int((split_df['label_binary'] == 1).sum()):>5d}"
        )

    return splits


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Lưu train/val/test splits thành CSV.

    Args:
        splits: Dict từ split_dataset().
        output_dir: Thư mục output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in splits.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"💾 Saved: {path}")


def generate_statistics(df: pd.DataFrame, stats_dir: Path) -> None:
    """Tạo thống kê và biểu đồ EDA.

    Args:
        df: Full merged DataFrame.
        stats_dir: Thư mục output cho statistics.
    """
    import json

    stats_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_samples": len(df),
        "label_distribution": df["label"].value_counts().to_dict(),
        "label_binary_distribution": df["label_binary"].value_counts().to_dict(),
        "source_distribution": df["source"].value_counts().to_dict() if "source" in df.columns else {},
        "lang_distribution": df["lang"].value_counts().to_dict() if "lang" in df.columns else {},
        "text_length_stats": {
            "mean": float(df["text_final"].str.len().mean()),
            "median": float(df["text_final"].str.len().median()),
            "min": int(df["text_final"].str.len().min()),
            "max": int(df["text_final"].str.len().max()),
        },
    }

    stats_path = stats_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"📊 Stats saved: {stats_path}")

    # Generate plots (optional, requires matplotlib)
    try:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-darkgrid")

        # 1. Class distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        df["label"].value_counts().plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c", "#f39c12"])
        ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        plt.tight_layout()
        fig.savefig(stats_dir / "class_distribution.png", dpi=150)
        plt.close()

        # 2. Text length distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        df["text_final"].str.len().hist(bins=50, ax=ax, color="#3498db", alpha=0.7)
        ax.set_title("Text Length Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Character Count")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        fig.savefig(stats_dir / "text_length_dist.png", dpi=150)
        plt.close()

        logger.info(f"📈 Plots saved to {stats_dir}")

    except ImportError:
        logger.warning("⚠️ matplotlib not installed, skipping plots.")


def main() -> None:
    """Merge, split, và generate statistics."""
    import sys
    import argparse

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import DATASET_PROCESSED_DIR, DATASET_STATS_DIR

    parser = argparse.ArgumentParser(description="Merge & Split datasets")
    parser.add_argument("--exclude-suspicious", action="store_true", help="Loại bỏ nhãn suspicious (NEI)")
    args = parser.parse_args()

    DATASET_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🚀 MERGE & SPLIT DATASETS")
    if args.exclude_suspicious:
        logger.info("🚫 Mode: Binary only (exclude suspicious/NEI)")
    logger.info("=" * 60)

    # 1. Merge
    df = merge_all_datasets(DATASET_PROCESSED_DIR, exclude_suspicious=args.exclude_suspicious)
    if df.empty:
        return

    # 2. Split
    logger.info("\n📂 Splitting dataset (80/10/10)...")
    splits = split_dataset(df)

    # 3. Save
    logger.info("\n💾 Saving splits...")
    save_splits(splits, DATASET_PROCESSED_DIR)

    # 4. Statistics
    logger.info("\n📊 Generating statistics...")
    generate_statistics(df, DATASET_STATS_DIR)

    logger.info("\n" + "=" * 60)
    logger.info("✅ MERGE & SPLIT HOÀN TẤT")
    logger.info("=" * 60)



if __name__ == "__main__":
    main()

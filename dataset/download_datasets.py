"""
download_datasets.py — Tự động download ViFactCheck + ReINTEL
================================================================
Chạy:
    python dataset/download_datasets.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_vifactcheck(output_dir: Path) -> pd.DataFrame:
    """Download ViFactCheck từ HuggingFace và lưu CSV.

    Dataset: ngtram/vifactcheck (7,232 claim-evidence pairs, AAAI 2025)
    Columns: claim, evidence, label (SUPPORTED / REFUTED / NOT ENOUGH INFO)

    Returns:
        DataFrame with columns: text, evidence, label, original_label, source
    """
    logger.info("📥 Downloading ViFactCheck từ HuggingFace...")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        splits = {
            'train': 'data/train-00000-of-00001.parquet',
            'test': 'data/test-00000-of-00001.parquet',
            'dev': 'data/dev-00000-of-00001.parquet'
        }
        ds = {}
        for split_name, split_path in splits.items():
            df_split = pd.read_parquet("hf://datasets/tranthaihoa/vifactcheck/" + split_path)
            ds[split_name] = df_split.to_dict("records")
    except Exception as exc:
        logger.error(f"❌ Không thể download ViFactCheck: {exc}")
        logger.info(
            "💡 Có thể cần cài đặt thêm pyarrow hoặc fsspec\n"
            "   Hoặc download thủ công: https://huggingface.co/datasets/tranthaihoa/vifactcheck"
        )
        return pd.DataFrame()

    all_rows: list[dict] = []

    for split_name in ds:
        split = ds[split_name]
        logger.info(f"  📂 Split '{split_name}': {len(split)} samples")

        for row in split:
            # Map label: SUPPORTED → real, REFUTED → fake, NEI → suspicious
            original_label = row.get("label", row.get("labels", ""))
            if isinstance(original_label, int):
                label_map_int = {0: "SUPPORTED", 1: "REFUTED", 2: "NOT ENOUGH INFO"}
                original_label = label_map_int.get(original_label, str(original_label))

            label_map = {
                "SUPPORTED": "real",
                "REFUTED": "fake",
                "NOT ENOUGH INFO": "suspicious",
                "NEI": "suspicious",
            }
            label = label_map.get(str(original_label).upper().strip(), "suspicious")

            all_rows.append({
                "text": row.get("claim", row.get("Statement", "")),
                "evidence": row.get("evidence", row.get("Evidence", "")),
                "label": label,
                "original_label": original_label,
                "source": "vifactcheck",
                "split": split_name,
            })

    df = pd.DataFrame(all_rows)

    # Save
    csv_path = output_dir / "vifactcheck_full.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"✅ ViFactCheck saved: {csv_path} ({len(df)} rows)")

    # Save evidence as knowledge base (JSONL)
    kb_path = output_dir / "vifactcheck_kb.jsonl"
    with open(kb_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if row["evidence"] and str(row["evidence"]).strip():
                entry = {
                    "claim": row["text"],
                    "evidence": row["evidence"],
                    "label": row["label"],
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"✅ Knowledge base saved: {kb_path}")

    return df


def download_reintel(output_dir: Path) -> pd.DataFrame:
    """Download/load ReINTEL (VLSP 2020) dataset.

    ReINTEL = Reliable Intelligence Identification on Vietnamese Social Network.
    Nếu không có public link → tạo placeholder và hướng dẫn user tải thủ công.

    Returns:
        DataFrame with columns: text, label, source
    """
    logger.info("📥 Loading ReINTEL (VLSP 2020)...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if manually downloaded
    manual_paths = [
        output_dir / "reintel_train.csv",
        output_dir / "reintel.csv",
        output_dir / "train.csv",
    ]

    for path in manual_paths:
        if path.exists():
            logger.info(f"  📂 Found existing file: {path}")
            df = pd.read_csv(path, encoding="utf-8")

            # Normalize columns
            text_col = None
            label_col = None
            for col in df.columns:
                if col.lower() in ("text", "content", "post_text", "title"):
                    text_col = col
                if col.lower() in ("label", "class", "reliable"):
                    label_col = col

            if text_col and label_col:
                df_clean = pd.DataFrame({
                    "text": df[text_col],
                    "label": df[label_col].map(
                        lambda x: "fake" if str(x) in ("1", "unreliable", "fake") else "real"
                    ),
                    "source": "reintel",
                })
                out_path = output_dir / "reintel_clean.csv"
                df_clean.to_csv(out_path, index=False, encoding="utf-8")
                logger.info(f"✅ ReINTEL cleaned: {out_path} ({len(df_clean)} rows)")
                return df_clean

    # No data found → create placeholder
    logger.warning(
        "⚠️ ReINTEL data not found. Please download manually:\n"
        "   1. Visit: https://vlsp.org.vn/vlsp2020/eval/reintel\n"
        "   2. Download train/test data\n"
        f"   3. Place CSV files in: {output_dir}/\n"
        "   4. Re-run this script"
    )

    placeholder = output_dir / "README_REINTEL.txt"
    placeholder.write_text(
        "ReINTEL (VLSP 2020) — Reliable Intelligence Identification\n"
        "============================================================\n"
        "Download: https://vlsp.org.vn/vlsp2020/eval/reintel\n"
        "Place train CSV here, then re-run download_datasets.py\n",
        encoding="utf-8",
    )

    return pd.DataFrame()


def main() -> None:
    """Download tất cả datasets."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import VIFACTCHECK_DIR, REINTEL_DIR

    logger.info("=" * 60)
    logger.info("🚀 BẮT ĐẦU DOWNLOAD DATASETS")
    logger.info("=" * 60)

    # 1. ViFactCheck
    df_vfc = download_vifactcheck(VIFACTCHECK_DIR)
    if not df_vfc.empty:
        logger.info(f"\n📊 ViFactCheck Stats:")
        logger.info(f"   Total: {len(df_vfc)}")
        logger.info(f"   Labels: {df_vfc['label'].value_counts().to_dict()}")

    # 2. ReINTEL
    df_reintel = download_reintel(REINTEL_DIR)
    if not df_reintel.empty:
        logger.info(f"\n📊 ReINTEL Stats:")
        logger.info(f"   Total: {len(df_reintel)}")
        logger.info(f"   Labels: {df_reintel['label'].value_counts().to_dict()}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ DOWNLOAD HOÀN TẤT")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

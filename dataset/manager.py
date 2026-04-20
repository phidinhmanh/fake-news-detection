"""
dataset/manager.py - Unified Dataset Pipeline
==============================================
Handles Download -> Preprocess -> Merge -> Feature Extraction in one flow.

Follows FR-4.7: Stream data in chunks for large files.
"""

from __future__ import annotations

import logging
import pandas as pd
from pathlib import Path
from typing import Iterator, Optional, Callable

import sys
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET_DIR, DATASET_RAW_DIR, DATASET_PROCESSED_DIR, VIFACTCHECK_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Chunk size for streaming large files (FR-4.7)
DEFAULT_CHUNK_SIZE = 5000

class DatasetManager:
    """Orchestrates the entire data pipeline for Vietnamese Fake News."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.chunk_size = chunk_size
        DATASET_RAW_DIR.mkdir(parents=True, exist_ok=True)
        DATASET_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def iter_chunks(self, path: Path, chunk_size: int | None = None) -> Iterator[pd.DataFrame]:
        """
        Iterate over large CSV files in chunks (FR-4.7).

        Yields DataFrames of size chunk_size to avoid loading entire file into memory.
        """
        size = chunk_size or self.chunk_size
        logger.info(f"Streaming {path.name} in chunks of {size} rows")

        try:
            for chunk_idx, chunk in enumerate(pd.read_csv(path, chunksize=size)):
                logger.debug(f"Processing chunk {chunk_idx + 1} ({len(chunk)} rows)")
                yield chunk
        except Exception as e:
            logger.error(f"Failed to read {path} in chunks: {e}")
            raise

    def process_in_chunks(
        self,
        input_path: Path,
        output_path: Path,
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
        chunk_size: int | None = None,
    ) -> int:
        """
        Process large CSV file in chunks, writing output incrementally (FR-4.7).

        Returns total number of rows processed.
        """
        size = chunk_size or self.chunk_size
        total_rows = 0
        first_chunk = True

        for chunk in self.iter_chunks(input_path, size):
            # Apply transformation
            processed = transform_fn(chunk)

            # Write mode: write header only on first chunk
            mode = "w" if first_chunk else "a"
            header = first_chunk
            first_chunk = False

            processed.to_csv(output_path, mode=mode, header=header, index=False)
            total_rows += len(processed)

        logger.info(f"Processed {total_rows} rows from {input_path.name}")
        return total_rows

    def validate_chunked_output(
        self,
        input_path: Path,
        output_path: Path,
        expected_columns: list[str],
    ) -> bool:
        """
        Validate chunked output integrity (FR-4.3).
        """
        if not output_path.exists():
            return False

        # Count rows in output
        output_rows = 0
        for chunk in self.iter_chunks(output_path):
            output_rows += len(chunk)

        # Check expected columns in first chunk
        first_chunk = next(self.iter_chunks(output_path, chunk_size=1), None)
        if first_chunk is None:
            return False

        missing_cols = set(expected_columns) - set(first_chunk.columns)
        if missing_cols:
            logger.error(f"Missing columns in output: {missing_cols}")
            return False

        logger.info(f"Validation passed: {output_rows} rows, all columns present")
        return True

    def download(self):
        """Downloads ViFactCheck dataset from HuggingFace."""
        logger.info("Step 1: Downloading ViFactCheck...")
        try:
            from datasets import load_dataset
            dataset = load_dataset("phidinhmanh/ViFactCheck")
            VIFACTCHECK_DIR.mkdir(parents=True, exist_ok=True)
            for split in ["train", "test", "dev"]:
                df = pd.DataFrame(dataset[split])
                df.to_csv(VIFACTCHECK_DIR / f"{split}.csv", index=False)
            logger.info("Download complete.")
        except Exception as e:
            logger.error(f"Download failed: {e}")

    def preprocess(self):
        """Standardizes and tokenizes Vietnamese text using underthesea (FR-4.7)."""
        logger.info("Step 2: Preprocessing (underthesea tokenization)...")
        from underthesea import word_tokenize
        import re

        def clean_text(text: str) -> str:
            text = str(text).lower()
            text = re.sub(r'http\S+', '', text)
            text = word_tokenize(text, format="text")
            return text

        def transform_chunk(df: pd.DataFrame) -> pd.DataFrame:
            df['text_final'] = df['claim'].apply(clean_text)
            return df

        for split in ["train", "test", "dev"]:
            in_path = VIFACTCHECK_DIR / f"{split}.csv"
            if not in_path.exists():
                continue

            out_path = DATASET_PROCESSED_DIR / f"{split}_clean.csv"
            total_rows = self.process_in_chunks(in_path, out_path, transform_chunk)
            logger.info(f"Preprocessing {split} complete: {total_rows} rows")

    def merge_and_split(self):
        """Finalizes training splits and ensures binary labels."""
        logger.info("Step 3: Merging & Splitting...")

        splits = {"train": "train_clean.csv", "val": "dev_clean.csv", "test": "test_clean.csv"}

        for name, filename in splits.items():
            path = DATASET_PROCESSED_DIR / filename
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if 'label' in df.columns:
                df['label_binary'] = df['label'].apply(
                    lambda x: 1 if str(x).lower() in ['false', 'fake', 'refuted'] else 0
                )
                df.to_csv(DATASET_PROCESSED_DIR / f"{name}.csv", index=False)

        # Create a combined file for baseline TF-IDF training
        dfs = []
        for n in ["train", "val"]:
            p = DATASET_PROCESSED_DIR / f"{n}.csv"
            if p.exists():
                dfs.append(pd.read_csv(p))
        if dfs:
            pd.concat(dfs).to_csv(DATASET_PROCESSED_DIR / "preprocessed_all.csv", index=False)

        logger.info("Merge & Split complete.")

    def extract_features(self):
        """Extracts stylistic features using chunked processing (FR-4.7)."""
        logger.info("Step 4: Extracting Stylistic Features...")
        from dataset.feature_extraction import FEATURE_NAMES, extract_features_batch

        def transform_features_chunk(df: pd.DataFrame) -> pd.DataFrame:
            text_col = "text_final" if "text_final" in df.columns else "claim"
            features = extract_features_batch(df[text_col].tolist())
            feat_df = pd.DataFrame(features, columns=FEATURE_NAMES)
            return pd.concat([df.reset_index(drop=True), feat_df], axis=1)

        for split in ["train", "val", "test"]:
            path = DATASET_PROCESSED_DIR / f"{split}.csv"
            if not path.exists():
                continue

            out_path = DATASET_PROCESSED_DIR / f"{split}_with_features.csv"
            total_rows = self.process_in_chunks(path, out_path, transform_features_chunk)
            logger.info(f"Feature extraction {split}: {total_rows} rows")

        logger.info("Feature extraction complete.")

    def run_full_pipeline(self):
        """Runs the entire pipeline end-to-end."""
        self.download()
        self.preprocess()
        self.merge_and_split()
        self.extract_features()
        logger.info("Full Dataset Pipeline finished!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    args = parser.parse_args()

    manager = DatasetManager()
    if args.full:
        manager.run_full_pipeline()
    else:
        logger.info("Run with --full to execute all steps.")

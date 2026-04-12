"""
dataset.py — PyTorch Dataset for Fake News Detection
===================================================
Implemented for Task IMPL-B-001.
"""

from __future__ import annotations

import pathlib
import sys

_project_root = pathlib.Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import os
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import DEFAULT_MAX_SEQ_LEN, LABELS, NORMALIZED_DIR, AUGMENTED_DIR


class FakeNewsDataset(Dataset):
    """Dataset class for loading fake news data from Parquet files.

    Supports loading from both normalized and augmented data directories.
    """

    def __init__(
        self,
        data_type: Literal["normalized", "augmented"] = "normalized",
        split: Literal["train", "val", "test"] = "train",
        tokenizer_name: str = "xlm-roberta-base",  # Multilingual tokenizer
        max_length: int = DEFAULT_MAX_SEQ_LEN,
    ):
        """Initializes the dataset.

        Args:
            data_type: Type of data to load ('normalized' or 'augmented').
            split: Data split to load ('train', 'val', or 'test').
            tokenizer_name: Name of the transformer tokenizer to use.
            max_length: Maximum sequence length for tokenization.
        """
        self.data_dir = NORMALIZED_DIR if data_type == "normalized" else AUGMENTED_DIR
        self.split = split
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.file_path = self.data_dir / f"{split}.parquet"

        if not self.file_path.exists():
            # If the file doesn't exist, we'll initialize an empty dataframe
            # but log a warning. In a real scenario, this should probably raise an error
            # or handle missing data appropriately.
            print(f"Warning: File {self.file_path} not found.")
            self.df = pd.DataFrame(columns=["text", "label", "domain", "lang"])
        else:
            self.df = pd.read_parquet(self.file_path)

        # Map labels to integers if they are strings
        self.label_map = {label: i for i, label in enumerate(LABELS)}

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary containing tokenized text and the corresponding label.
        """
        row = self.df.iloc[idx]
        text = str(row["text"])
        # Use label_str if available, otherwise fallback to label column
        label_str_val = row.get("label_str", row.get("label"))
        # Handle both int (0/1) and string ("fake"/"real") formats
        if isinstance(label_str_val, int):
            label = label_str_val
        else:
            label = self.label_map.get(label_str_val, 0)

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

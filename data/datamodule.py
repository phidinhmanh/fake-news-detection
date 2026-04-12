"""
datamodule.py — PyTorch Lightning DataModule
===============================================
Người A phát triển. Output chính cho Người B dùng.

TODO (Tuần 1-2):
    - [ ] Implement FakeNewsDataset(torch.utils.data.Dataset)
    - [ ] Implement FakeNewsDataModule(lightning.LightningDataModule)
    - [ ] Support cả tiếng Việt và tiếng Anh
    - [ ] Integrate với normalized datasets

TODO (Tuần 3-4):
    - [ ] DataModule v2 với WeightedRandomSampler
    - [ ] Support augmented datasets
"""

from __future__ import annotations

import pathlib
import sys

_project_root = pathlib.Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import DEFAULT_BATCH_SIZE, NORMALIZED_DIR
from data.dataset import FakeNewsDataset


class FakeNewsDataModule(L.LightningDataModule):
    """Lightning DataModule cho Fake News Detection.

    Người B sẽ import class này để training:
        from data.datamodule import FakeNewsDataModule
        dm = FakeNewsDataModule(batch_size=32)
        trainer.fit(model, dm)
    """

    def __init__(self, batch_size=DEFAULT_BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = FakeNewsDataset(split="train")
            self.val_dataset = FakeNewsDataset(split="val")

        if stage == "test" or stage is None:
            self.test_dataset = FakeNewsDataset(split="test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

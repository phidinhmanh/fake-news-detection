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

# import lightning as L
# import torch
# from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# from config import DEFAULT_BATCH_SIZE, NORMALIZED_DIR


# class FakeNewsDataset(Dataset):
#     """Dataset cho Fake News Detection.
#
#     Args:
#         data_dir: Path tới thư mục chứa parquet files.
#         split: 'train', 'val', hoặc 'test'.
#         max_length: Maximum sequence length.
#     """
#
#     def __init__(self, data_dir, split="train", max_length=256):
#         raise NotImplementedError("Người A implement")
#
#     def __len__(self):
#         raise NotImplementedError
#
#     def __getitem__(self, idx):
#         raise NotImplementedError


# class FakeNewsDataModule(L.LightningDataModule):
#     """Lightning DataModule cho Fake News Detection.
#
#     Người B sẽ import class này để training:
#         from data.datamodule import FakeNewsDataModule
#         dm = FakeNewsDataModule(batch_size=32)
#         trainer.fit(model, dm)
#     """
#
#     def __init__(self, batch_size=DEFAULT_BATCH_SIZE):
#         super().__init__()
#         self.batch_size = batch_size
#
#     def setup(self, stage=None):
#         raise NotImplementedError("Người A implement")
#
#     def train_dataloader(self):
#         raise NotImplementedError
#
#     def val_dataloader(self):
#         raise NotImplementedError
#
#     def test_dataloader(self):
#         raise NotImplementedError

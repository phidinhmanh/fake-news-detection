"""
train.py — HuggingFace + Lightning Training Loop
====================================================
Người B phát triển.

TODO (Tuần 1-2):
    - [ ] Setup LightningModule wrapper cho HuggingFace model
    - [ ] Config loading từ config.yaml
    - [ ] Training loop với Lightning Trainer
    - [ ] Metrics logging (F1, accuracy, loss)
    - [ ] Baseline F1 ≥ 0.72

Chạy:
    python model/train.py --config model/config.yaml
"""

from __future__ import annotations

# import lightning as L
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer


# class FakeNewsClassifier(L.LightningModule):
#     """Lightning wrapper cho HuggingFace model.
#
#     Args:
#         model_name: HuggingFace model identifier.
#         num_labels: Số classes (2: fake/real).
#         learning_rate: Learning rate.
#     """
#
#     def __init__(
#         self,
#         model_name="xlm-roberta-base",
#         num_labels=2,
#         learning_rate=2e-5,
#     ):
#         super().__init__()
#         self.save_hyperparameters()
#         raise NotImplementedError("Người B implement")
#
#     def forward(self, **kwargs):
#         raise NotImplementedError
#
#     def training_step(self, batch, batch_idx):
#         raise NotImplementedError
#
#     def validation_step(self, batch, batch_idx):
#         raise NotImplementedError
#
#     def configure_optimizers(self):
#         raise NotImplementedError


# if __name__ == "__main__":
#     # TODO: argparse + config loading
#     pass

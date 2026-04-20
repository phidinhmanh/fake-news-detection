"""
phobert.py — Unified PhoBERT Models & Datasets
==============================================
Consolidates baseline and stylistic feature variants for binary classification.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Models ─────────────────────────────────────────────────────────────

class PhoBERTBaseline(nn.Module):
    """PhoBERT Binary Classifier (Baseline).
    Architecture: PhoBERT-base → [CLS] → Linear(768, 2) → Softmax
    """
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_labels: int = 2,
        dropout: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.class_weights = class_weights
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        
        res = {"logits": logits}
        if labels is not None:
            weight = self.class_weights.to(labels.device) if self.class_weights is not None else None
            loss_fn = nn.CrossEntropyLoss(weight=weight)
            res["loss"] = loss_fn(logits, labels)
        return res

class PhoBERTWithFeatures(nn.Module):
    """PhoBERT + Stylistic Features Classifier.
    Concatenate PhoBERT [CLS] (768-dim) with stylistic features (9-dim).
    """
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_features: int = 9,
        num_labels: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.feature_norm = nn.BatchNorm1d(num_features)
        self.class_weights = class_weights
        
        combined_dim = self.backbone.config.hidden_size + num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, stylistic_features: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        feat_norm = self.feature_norm(stylistic_features.float())
        
        combined = torch.cat([cls_embedding, feat_norm], dim=-1)
        logits = self.classifier(combined)
        
        res = {"logits": logits}
        if labels is not None:
            weight = self.class_weights.to(labels.device) if self.class_weights is not None else None
            loss_fn = nn.CrossEntropyLoss(weight=weight)
            res["loss"] = loss_fn(logits, labels)
        return res

# ── Dataset ────────────────────────────────────────────────────────────

class PhoBERTDataset(torch.utils.data.Dataset):
    """Unified Dataset for PhoBERT training (with or without features)."""
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        features: Optional[np.ndarray] = None,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.features = features
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }
        if self.features is not None:
            item["stylistic_features"] = torch.tensor(self.features[idx], dtype=torch.float32)
        return item

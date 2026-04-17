"""
phobert_with_features.py — PhoBERT + Stylistic Features
=========================================================
Architecture:
    PhoBERT → [CLS] embedding (768-dim)
    ⊕ stylistic_features (9-dim)
    → Linear(777, 256) → ReLU → Dropout → Linear(256, 2) → Softmax

Cải tiến so với baseline: concatenate embedding với features phong cách viết
(số từ, sentiment, tỷ lệ dấu chấm than, v.v.)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PhoBERTWithFeatures(nn.Module):
    """PhoBERT + Stylistic Features Classifier.

    Concatenate PhoBERT [CLS] embedding (768-dim) với stylistic features (9-dim)
    rồi đưa qua 2-layer MLP classifier.
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_features: int = 9,
        num_labels: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_features = num_features

        try:
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained(model_name)
        except Exception as exc:
            logger.error(f"❌ Không thể load PhoBERT: {exc}")
            raise

        hidden_size = self.backbone.config.hidden_size  # 768

        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Feature normalization layer
        self.feature_norm = nn.BatchNorm1d(num_features)

        # Total input: 768 (PhoBERT) + 9 (stylistic) = 777
        combined_dim = hidden_size + num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

        logger.info(
            f"📐 PhoBERT+Features: {hidden_size} (embedding) + "
            f"{num_features} (features) = {combined_dim} → {hidden_dim} → {num_labels}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        stylistic_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len).
            attention_mask: Attention mask, shape (batch, seq_len).
            stylistic_features: Extra features, shape (batch, num_features).
            labels: Optional labels, shape (batch,).

        Returns:
            Dict with 'logits' and optionally 'loss'.
        """
        # PhoBERT embedding
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        # Normalize features
        features_normed = self.feature_norm(stylistic_features.float())  # (batch, 9)

        # Concatenate
        combined = torch.cat([cls_embedding, features_normed], dim=-1)  # (batch, 777)

        logits = self.classifier(combined)  # (batch, num_labels)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        stylistic_features: torch.Tensor,
    ) -> torch.Tensor:
        """Dự đoán xác suất.

        Returns:
            Probabilities shape (batch, num_labels).
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, stylistic_features)
            probs = torch.softmax(outputs["logits"], dim=-1)
        return probs

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        logger.info(f"💾 Model saved: {path}")

    def load_model(self, path: str, device: str = "cpu") -> None:
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        logger.info(f"📂 Model loaded: {path}")


class PhoBERTFeaturesDataset(torch.utils.data.Dataset):
    """Dataset cho PhoBERT + stylistic features.

    Args:
        texts: List of text strings.
        labels: List of integer labels (0=real, 1=fake).
        features: numpy array shape (n, 9) — stylistic features.
        tokenizer: PhoBERT tokenizer.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        features: np.ndarray,
        tokenizer,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length

        assert len(texts) == len(labels) == len(features), (
            f"Length mismatch: texts={len(texts)}, labels={len(labels)}, features={len(features)}"
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        feature_vec = self.features[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "stylistic_features": torch.tensor(feature_vec, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
        }

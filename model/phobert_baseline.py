"""
phobert_baseline.py — Fine-tune PhoBERT cho Binary Classification (Real/Fake)
================================================================================
Architecture: PhoBERT-base → [CLS] → Linear(768, 2) → Softmax
Training: AdamW + Linear Warmup + Early Stopping

Usage:
    from model.phobert_baseline import PhoBERTBaseline
    model = PhoBERTBaseline()
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR

logger = logging.getLogger(__name__)


class PhoBERTBaseline(nn.Module):
    """PhoBERT Binary Classifier.

    Fine-tune vinai/phobert-base cho binary classification (Real=0 / Fake=1).
    Chỉ dùng [CLS] token embedding → FC → Softmax.
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        try:
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained(model_name)
        except Exception as exc:
            logger.error(f"❌ Không thể load PhoBERT: {exc}")
            raise

        hidden_size = self.backbone.config.hidden_size  # 768 for phobert-base

        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("🧊 PhoBERT backbone frozen (chỉ train classifier head)")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len).
            attention_mask: Attention mask, shape (batch, seq_len).
            labels: Optional labels for computing loss, shape (batch,).

        Returns:
            Dict with 'logits' and optionally 'loss'.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Lấy [CLS] token (index 0) từ last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        logits = self.classifier(cls_embedding)  # (batch, num_labels)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Dự đoán xác suất.

        Returns:
            Probabilities shape (batch, num_labels).
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1)
        return probs

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Dự đoán label.

        Returns:
            Predicted labels shape (batch,).
        """
        probs = self.predict_proba(input_ids, attention_mask)
        return probs.argmax(dim=-1)

    @staticmethod
    def get_tokenizer(model_name: str = "vinai/phobert-base"):
        """Load PhoBERT tokenizer.

        Returns:
            AutoTokenizer instance.
        """
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)

    def save_model(self, path: str) -> None:
        """Lưu model weights."""
        torch.save(self.state_dict(), path)
        logger.info(f"💾 Model saved: {path}")

    def load_model(self, path: str, device: str = "cpu") -> None:
        """Load model weights."""
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        logger.info(f"📂 Model loaded: {path}")


class PhoBERTDataset(torch.utils.data.Dataset):
    """PyTorch Dataset cho PhoBERT training.

    Args:
        texts: List of text strings.
        labels: List of integer labels (0=real, 1=fake).
        tokenizer: PhoBERT tokenizer.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])

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
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_optimizer_and_scheduler(
    model: PhoBERTBaseline,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    total_steps: int = 1000,
    weight_decay: float = 0.01,
) -> tuple:
    """Tạo AdamW optimizer + Linear warmup scheduler.

    Returns:
        Tuple of (optimizer, scheduler).
    """
    # Tách parameters: backbone dùng LR thấp, classifier dùng LR cao
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    optimizer_grouped = [
        {
            "params": [
                p for n, p in model.backbone.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
            "lr": learning_rate,
        },
        {
            "params": [
                p for n, p in model.backbone.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
        {
            "params": model.classifier.parameters(),
            "weight_decay": weight_decay,
            "lr": learning_rate * 10,  # Classifier head: higher LR
        },
    ]

    optimizer = AdamW(optimizer_grouped, lr=learning_rate)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    decay_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=max(1, total_steps - warmup_steps),
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps],
    )

    return optimizer, scheduler

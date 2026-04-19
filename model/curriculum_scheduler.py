"""curriculum_scheduler.py — Curriculum Learning Scheduler
==========================================================
Orders samples from easy to hard for curriculum learning.

Strategy: Start with high-confidence samples and gradually introduce
harder samples as training progresses.

Usage:
    scheduler = CurriculumLearningScheduler(
        dataset=train_dataset,
        num_epochs=10,
        initial_ratio=0.5,
        final_ratio=1.0,
    )
    scheduler.compute_difficulties(model)
    indices = scheduler.get_curriculum_indices(epoch=0)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from torch import nn


class CurriculumLearningScheduler:
    """Curriculum Learning scheduler that orders samples from easy to hard.

    Responsibility: Curriculum learning logic only (SRP fix for ARCH-002).
    """

    def __init__(
        self,
        dataset,
        num_epochs: int,
        initial_ratio: float = 0.5,
        final_ratio: float = 1.0,
        difficulty_metric: str = "loss",
    ):
        """Initialize curriculum scheduler.

        Args:
            dataset: The training dataset.
            num_epochs: Total number of training epochs.
            initial_ratio: Ratio of easy samples to use in first epoch.
            final_ratio: Ratio to use in final epoch (typically 1.0).
            difficulty_metric: Metric to determine difficulty ("loss" or "confidence").
        """
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.difficulty_metric = difficulty_metric
        self.sample_difficulties: Optional[np.ndarray] = None

    def compute_difficulties(
        self,
        model: "nn.Module",
        device: str = "cuda",
    ) -> np.ndarray:
        """Compute difficulty scores for all samples in the dataset.

        Args:
            model: The model to use for computing difficulties.
            device: Device to run inference on.

        Returns:
            Array of difficulty scores (higher = harder).
        """
        model.eval()
        difficulties = []

        dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                if self.difficulty_metric == "loss":
                    logits = outputs.logits
                    loss = F.cross_entropy(logits, labels, reduction="none")
                    difficulties.extend(loss.cpu().numpy().tolist())
                elif self.difficulty_metric == "confidence":
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    max_probs = torch.max(probs, dim=-1)[0]
                    difficulty = 1.0 - max_probs
                    difficulties.extend(difficulty.cpu().numpy().tolist())

        model.train()
        self.sample_difficulties = np.array(difficulties)
        return self.sample_difficulties

    def get_curriculum_indices(self, epoch: int) -> np.ndarray:
        """Get sample indices for the current epoch based on curriculum.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Array of sample indices to use for training.
        """
        if self.sample_difficulties is None:
            raise ValueError("Must call compute_difficulties() first")

        ratio = self.initial_ratio + (self.final_ratio - self.initial_ratio) * (
            epoch / max(self.num_epochs - 1, 1)
        )
        num_samples = int(len(self.dataset) * ratio)

        sorted_indices = np.argsort(self.sample_difficulties)

        return sorted_indices[:num_samples]

    def reset_difficulties(self) -> None:
        """Reset computed difficulties."""
        self.sample_difficulties = None
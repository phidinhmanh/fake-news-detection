"""
train.py — HuggingFace + Lightning Training Loop with LoRA & Curriculum Learning
==================================================================================
Người B phát triển.

Features:
    - LoRA fine-tuning using PEFT library
    - Curriculum Learning (Easy → Hard samples)
    - PyTorch Lightning training loop
    - Metrics logging (F1, accuracy, loss)
    - Target F1 ≥ 0.85

Usage:
    python model/train.py --config model/config.yaml
    python model/train.py --config model/config.yaml --curriculum
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_SEQ_LEN,
    MODEL_DIR,
    MODELS_ARTIFACTS_DIR,
    TARGET_LORA_F1,
)
from data.datamodule import FakeNewsDataModule


class FakeNewsClassifier(L.LightningModule):
    """Lightning wrapper for HuggingFace model with LoRA fine-tuning.

    Args:
        model_name: HuggingFace model identifier (e.g., "xlm-roberta-base").
        num_labels: Number of classes (2: fake/real).
        learning_rate: Learning rate for optimizer.
        warmup_ratio: Ratio of training steps for warmup.
        weight_decay: Weight decay for AdamW optimizer.
        lora_config: LoRA configuration dictionary.
        total_training_steps: Total number of training steps for scheduler.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        lora_config: Optional[Dict[str, Any]] = None,
        total_training_steps: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load base model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

        # Apply LoRA if config provided
        if lora_config:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("alpha", 32),
                lora_dropout=lora_config.get("dropout", 0.1),
                target_modules=lora_config.get("target_modules", ["query", "value"]),
                inference_mode=False,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.total_training_steps = total_training_steps

        # Storage for predictions and labels (for epoch-level metrics)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    # ========== Helper method for metrics computation ==========
    def _compute_metrics(self, outputs: list) -> dict:
        """Compute metrics from step outputs. Reused across train/val/test."""
        if not outputs:
            return {}
        all_preds = torch.cat([x["preds"] for x in outputs]).cpu().numpy()
        all_labels = torch.cat([x["labels"] for x in outputs]).cpu().numpy()
        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="binary", zero_division=0),
            "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="binary", zero_division=0),
        }

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through the model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        """Training step for each batch."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )

        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Store outputs for epoch-level metrics
        self.training_step_outputs.append({
            "loss": loss.detach(),
            "preds": preds.detach(),
            "labels": batch["label"].detach(),
        })

        return loss

    def on_train_epoch_end(self):
        """Compute metrics at the end of training epoch."""
        metrics = self._compute_metrics(self.training_step_outputs)
        if metrics:
            self.log_dict({f"train_{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        """Validation step for each batch."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )

        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        # Log loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Store outputs for epoch-level metrics
        self.validation_step_outputs.append({
            "loss": loss.detach(),
            "preds": preds.detach(),
            "labels": batch["label"].detach(),
        })

        return loss

    def on_validation_epoch_end(self):
        """Compute metrics at the end of validation epoch."""
        metrics = self._compute_metrics(self.validation_step_outputs)
        if metrics:
            self.log_dict({f"val_{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step for each batch."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )

        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        # Store outputs for epoch-level metrics
        self.test_step_outputs.append({
            "loss": loss.detach(),
            "preds": preds.detach(),
            "labels": batch["label"].detach(),
        })

        return loss

    def on_test_epoch_end(self):
        """Compute metrics at the end of test epoch."""
        metrics = self._compute_metrics(self.test_step_outputs)
        if metrics:
            self.log_dict({f"test_{k}": v for k, v in metrics.items()})
            # Print results
            print(f"\n{'='*60}")
            print(f"Test Results:")
            for k, v in metrics.items():
                print(f"  {k.capitalize()}: {v:.4f}")
            print(f"{'='*60}\n")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        if self.total_training_steps:
            num_warmup_steps = int(self.total_training_steps * self.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_training_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer


class CurriculumLearningScheduler:
    """Curriculum Learning scheduler that orders samples from easy to hard.

    Strategy: Start with high-confidence samples and gradually introduce
    harder samples as training progresses.
    """

    def __init__(
        self,
        dataset,
        num_epochs: int,
        initial_ratio: float = 0.5,
        final_ratio: float = 1.0,
        difficulty_metric: str = "loss",
    ):
        """
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
        self.sample_difficulties = None

    def compute_difficulties(self, model, device="cuda"):
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
                    # Use per-sample loss as difficulty
                    logits = outputs.logits
                    loss = F.cross_entropy(logits, labels, reduction="none")
                    difficulties.extend(loss.cpu().numpy().tolist())
                elif self.difficulty_metric == "confidence":
                    # Use (1 - max_prob) as difficulty
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

        # Linear interpolation of difficulty ratio
        ratio = self.initial_ratio + (self.final_ratio - self.initial_ratio) * (epoch / max(self.num_epochs - 1, 1))
        num_samples = int(len(self.dataset) * ratio)

        # Sort samples by difficulty (easy first)
        sorted_indices = np.argsort(self.sample_difficulties)

        # Return the easiest samples up to the current ratio
        return sorted_indices[:num_samples]


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def train_with_curriculum(
    config: Dict[str, Any],
    datamodule: FakeNewsDataModule,
    use_curriculum: bool = True,
):
    """Train model with optional curriculum learning.

    Args:
        config: Training configuration dictionary.
        datamodule: Lightning DataModule.
        use_curriculum: Whether to use curriculum learning.
    """
    # Setup datamodule
    datamodule.setup("fit")

    # Calculate total training steps
    num_epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    train_dataset_size = len(datamodule.train_dataset)
    steps_per_epoch = train_dataset_size // batch_size
    total_training_steps = steps_per_epoch * num_epochs

    # Initialize model
    model = FakeNewsClassifier(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        learning_rate=config["training"]["learning_rate"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        lora_config=config.get("lora"),
        total_training_steps=total_training_steps,
    )

    # Setup callbacks
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=MODELS_ARTIFACTS_DIR / "checkpoints",
        filename="fake-news-{epoch:02d}-{val_f1:.4f}",
        monitor=config["logging"]["monitor"],
        mode=config["logging"]["mode"],
        save_top_k=config["logging"]["save_top_k"],
        save_last=True,
    )

    early_stopping_callback = L.pytorch.callbacks.EarlyStopping(
        monitor=config["logging"]["monitor"],
        mode=config["logging"]["mode"],
        patience=3,
        verbose=True,
    )

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        callbacks=[checkpoint_callback, early_stopping_callback],
        gradient_clip_val=1.0,
        deterministic=False,
    )

    if use_curriculum:
        print("\n" + "="*60)
        print("Training with Curriculum Learning")
        print("="*60 + "\n")

        # Initialize curriculum scheduler
        curriculum = CurriculumLearningScheduler(
            dataset=datamodule.train_dataset,
            num_epochs=num_epochs,
            initial_ratio=0.5,
            final_ratio=1.0,
            difficulty_metric="loss",
        )

        # Compute initial difficulties (using a small model or random initialization)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print("Computing sample difficulties...")
        curriculum.compute_difficulties(model.model, device=device)
        print("Difficulty computation complete.\n")

        # Train with curriculum
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

            # Get curriculum indices for this epoch
            indices = curriculum.get_curriculum_indices(epoch)
            print(f"Training on {len(indices)}/{len(datamodule.train_dataset)} samples")

            # Create subset dataloader
            train_subset = Subset(datamodule.train_dataset, indices)
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )

            # Train for one epoch
            trainer.fit_loop.max_epochs = epoch + 1
            if epoch == 0:
                trainer.fit(model, train_loader, datamodule.val_dataloader())
            else:
                trainer.fit_loop.epoch_progress.current.completed = epoch
                trainer.fit(model, train_loader, datamodule.val_dataloader(), ckpt_path="last")

            # Recompute difficulties after each epoch
            if epoch < num_epochs - 1:
                model = model.to(device)
                curriculum.compute_difficulties(model.model, device=device)

    else:
        print("\n" + "="*60)
        print("Training without Curriculum Learning")
        print("="*60 + "\n")

        # Standard training
        trainer.fit(model, datamodule)

    # Test the model
    print("\n" + "="*60)
    print("Testing model on test set")
    print("="*60 + "\n")
    trainer.test(model, datamodule)

    # Save final model
    final_model_path = MODELS_ARTIFACTS_DIR / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Save LoRA adapter if using LoRA
    if config.get("lora"):
        lora_adapter_path = MODELS_ARTIFACTS_DIR / "lora_adapter"
        model.model.save_pretrained(lora_adapter_path)
        print(f"LoRA adapter saved to: {lora_adapter_path}")

    return model, trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Fake News Classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="model/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)

    # Override config with command line arguments
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs

    # Print configuration
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Model: {config['model']['name']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Max Seq Length: {config['training']['max_seq_length']}")
    if config.get("lora"):
        print(f"LoRA r: {config['lora']['r']}")
        print(f"LoRA alpha: {config['lora']['alpha']}")
        print(f"LoRA dropout: {config['lora']['dropout']}")
    print(f"Curriculum Learning: {args.curriculum}")
    print(f"Target F1: {TARGET_LORA_F1}")
    print("="*60 + "\n")

    # Create artifacts directory
    MODELS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_ARTIFACTS_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Initialize datamodule
    datamodule = FakeNewsDataModule(batch_size=config["training"]["batch_size"])

    # Train model
    model, trainer = train_with_curriculum(
        config=config,
        datamodule=datamodule,
        use_curriculum=args.curriculum,
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60 + "\n")


def train(
    config_path: str = "model/config.yaml",
    use_curriculum: bool = False,
    epochs: int | None = None,
    batch_size: int | None = None,
):
    """Train model — gọi trực tiếp từ notebook.

    Args:
        config_path: Path to config.yaml
        use_curriculum: Enable curriculum learning (default False)
        epochs: Override epochs from config
        batch_size: Override batch_size from config
    """
    config = load_config(Path(config_path))

    if epochs:
        config["training"]["epochs"] = epochs
    if batch_size:
        config["training"]["batch_size"] = batch_size

    # Ensure output directories
    MODELS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_ARTIFACTS_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Init datamodule
    datamodule = FakeNewsDataModule(batch_size=config["training"]["batch_size"])

    # Train
    model, trainer = train_with_curriculum(
        config=config,
        datamodule=datamodule,
        use_curriculum=use_curriculum,
    )

    return model, trainer


if __name__ == "__main__":
    main()

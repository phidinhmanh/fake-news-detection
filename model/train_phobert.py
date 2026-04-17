"""
train_phobert.py — Training Script cho PhoBERT models
======================================================
Hỗ trợ 2 variants:
  1. PhoBERT baseline (chỉ dùng text)
  2. PhoBERT + stylistic features (text + 9 features)

Chạy:
    python model/train_phobert.py --variant baseline
    python model/train_phobert.py --variant features
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATASET_PROCESSED_DIR,
    MODELS_ARTIFACTS_DIR,
    PHOBERT_BATCH_SIZE,
    PHOBERT_EPOCHS,
    PHOBERT_LEARNING_RATE,
    PHOBERT_MAX_SEQ_LEN,
    PHOBERT_MODEL_NAME,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data(split: str, with_features: bool = False) -> pd.DataFrame:
    """Load split data from processed directory.

    Args:
        split: 'train', 'val', or 'test'.
        with_features: Load version with stylistic features.

    Returns:
        DataFrame.
    """
    if with_features:
        path = DATASET_PROCESSED_DIR / f"{split}_with_features.csv"
        if not path.exists():
            path = DATASET_PROCESSED_DIR / f"{split}.csv"
            logger.warning(f"⚠️ Features file not found, using {path}")
    else:
        path = DATASET_PROCESSED_DIR / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"📂 Loaded {split}: {len(df)} samples")
    return df


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    with_features: bool = False,
) -> float:
    """Train 1 epoch.

    Returns:
        Average loss.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if with_features:
            features = batch["stylistic_features"].to(device)
            outputs = model(input_ids, attention_mask, features, labels=labels)
        else:
            outputs = model(input_ids, attention_mask, labels=labels)

        loss = outputs["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    with_features: bool = False,
) -> dict:
    """Evaluate model on dataloader.

    Returns:
        Dict with metrics: accuracy, f1, precision, recall, auc, loss.
    """
    model.eval()
    all_preds: list = []
    all_labels: list = []
    all_probs: list = []
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if with_features:
            features = batch["stylistic_features"].to(device)
            outputs = model(input_ids, attention_mask, features, labels=labels)
        else:
            outputs = model(input_ids, attention_mask, labels=labels)

        total_loss += outputs["loss"].item()

        probs = torch.softmax(outputs["logits"], dim=-1)
        preds = probs.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # P(fake)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
        "precision": float(precision_score(y_true, y_pred, average="binary")),
        "recall": float(recall_score(y_true, y_pred, average="binary")),
        "loss": total_loss / len(dataloader),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = 0.0

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhoBERT for Fake News Detection")
    parser.add_argument("--variant", choices=["baseline", "features"], default="baseline")
    parser.add_argument("--epochs", type=int, default=PHOBERT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=PHOBERT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=PHOBERT_LEARNING_RATE)
    parser.add_argument("--max-seq-len", type=int, default=PHOBERT_MAX_SEQ_LEN)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    with_features = args.variant == "features"

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"🖥️ Device: {device}")

    # Load tokenizer
    from model.phobert_baseline import PhoBERTBaseline, PhoBERTDataset, create_optimizer_and_scheduler

    tokenizer = PhoBERTBaseline.get_tokenizer(PHOBERT_MODEL_NAME)

    # Load data
    df_train = load_data("train", with_features=with_features)
    df_val = load_data("val", with_features=with_features)

    text_col = "text_final" if "text_final" in df_train.columns else "text"
    label_col = "label_binary"

    if label_col not in df_train.columns:
        logger.error(f"❌ Column '{label_col}' not found. Run merge_datasets.py first.")
        return

    # Create datasets
    if with_features:
        from model.phobert_with_features import PhoBERTWithFeatures, PhoBERTFeaturesDataset
        from dataset.feature_extraction import FEATURE_NAMES, extract_features_batch

        # Extract features if not in dataframe
        if not all(f in df_train.columns for f in FEATURE_NAMES):
            logger.info("📊 Extracting features on-the-fly...")
            train_features = extract_features_batch(df_train[text_col].tolist())
            val_features = extract_features_batch(df_val[text_col].tolist())
        else:
            train_features = df_train[FEATURE_NAMES].values.astype(np.float32)
            val_features = df_val[FEATURE_NAMES].values.astype(np.float32)

        train_dataset = PhoBERTFeaturesDataset(
            texts=df_train[text_col].tolist(),
            labels=df_train[label_col].tolist(),
            features=train_features,
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
        )
        val_dataset = PhoBERTFeaturesDataset(
            texts=df_val[text_col].tolist(),
            labels=df_val[label_col].tolist(),
            features=val_features,
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
        )

        model = PhoBERTWithFeatures(model_name=PHOBERT_MODEL_NAME).to(device)
    else:
        train_dataset = PhoBERTDataset(
            texts=df_train[text_col].tolist(),
            labels=df_train[label_col].tolist(),
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
        )
        val_dataset = PhoBERTDataset(
            texts=df_val[text_col].tolist(),
            labels=df_val[label_col].tolist(),
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
        )

        model = PhoBERTBaseline(model_name=PHOBERT_MODEL_NAME).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Optimizer & scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, learning_rate=args.lr, warmup_steps=warmup_steps, total_steps=total_steps
    )

    # Training loop
    logger.info(f"\n{'=' * 60}")
    logger.info(f"🚀 TRAINING PhoBERT {'+ Features' if with_features else 'Baseline'}")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")

    best_f1 = 0.0
    best_epoch = 0
    history: list[dict] = []

    MODELS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = f"phobert_{'features' if with_features else 'baseline'}"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, with_features)
        val_metrics = evaluate(model, val_loader, device, with_features)

        elapsed = time.time() - t0

        logger.info(
            f"  Epoch {epoch:>2d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        # Save best
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            save_path = MODELS_ARTIFACTS_DIR / f"{model_name}_best.pt"
            model.save_model(str(save_path))
            logger.info(f"  ⭐ New best F1: {best_f1:.4f} (epoch {epoch})")

    # Save final
    final_path = MODELS_ARTIFACTS_DIR / f"{model_name}_final.pt"
    model.save_model(str(final_path))

    # Save training history
    history_path = MODELS_ARTIFACTS_DIR / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"✅ TRAINING HOÀN TẤT")
    logger.info(f"  Best F1: {best_f1:.4f} (epoch {best_epoch})")
    logger.info(f"  Model: {final_path}")
    logger.info(f"  History: {history_path}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()

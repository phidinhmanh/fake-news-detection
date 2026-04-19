"""
train_phobert.py — Cập nhật dụng thư viện Hugging Face Trainer API
===================================================================
Hỗ trợ 2 variants:
  1. PhoBERT baseline (Sử dụng AutoModelForSequenceClassification)
  2. PhoBERT + stylistic features (Sử dụng PhoBERTWithFeatures custom model)

Chạy:
    python model/train_phobert.py --variant baseline
    python model/train_phobert.py --variant features
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import sys
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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import transformers
    transformers.set_seed(seed)
    logger.info(f"🌱 Seed set to {seed}")


def load_data(split: str, with_features: bool = False) -> pd.DataFrame:
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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Check if predictions is a tuple (e.g. from custom model outputs)
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=1)

    precision = precision_score(labels, predictions, average="binary", zero_division=0)
    recall = recall_score(labels, predictions, average="binary", zero_division=0)
    f1 = f1_score(labels, predictions, average="binary", zero_division=0)
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


class WeightedTrainer(Trainer):
    """Custom Trainer để apply class weights cho CrossEntropyLoss."""
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)

        # HF AutoModelForSequenceClassification returns an object with logits and loss
        # Phượng BERTWithFeatures returns a dictionary with logits and loss.
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        else:
            logits = outputs.logits

        if labels is not None:
            device = getattr(model, "device", next(model.parameters()).device)
            weight = self.class_weights.to(device) if self.class_weights is not None else None
            # Use label smoothing from TrainingArguments and class weights
            loss_fct = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.args.label_smoothing_factor)
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        else:
            if isinstance(outputs, dict):
                loss = outputs.get("loss")
            else:
                loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhoBERT using HF Trainer")
    parser.add_argument("--variant", choices=["baseline", "features"], default="baseline")
    parser.add_argument("--epochs", type=int, default=PHOBERT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=PHOBERT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=PHOBERT_LEARNING_RATE)
    parser.add_argument("--max-seq-len", type=int, default=PHOBERT_MAX_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default=PHOBERT_MODEL_NAME, help="Model from HuggingFace")
    parser.add_argument("--lr-scheduler-type", type=str, default="linear", 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau"],
                        help="The scheduler type to use.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW if we apply some.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Linear warmup over warmup_ratio fraction of total steps.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability for hidden layers and attention.")
    parser.add_argument("--label-smoothing-factor", type=float, default=0.0, help="Label smoothing factor (e.g. 0.1).")
    
    args = parser.parse_args()
    set_seed(args.seed)
    with_features = args.variant == "features"

    logger.info(f"🖥️ Training variant: {args.variant} using {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load data
    df_train = load_data("train", with_features=with_features)
    df_val = load_data("val", with_features=with_features)

    text_col = "text_final" if "text_final" in df_train.columns else "text"
    label_col = "label_binary"

    if label_col not in df_train.columns:
        logger.error(f"❌ Column '{label_col}' not found. Run merge_datasets.py first.")
        return

    # Tính toán class weights (giúp cân bằng dữ liệu nếu bị lệch)
    classes = np.unique(df_train[label_col])
    class_w = compute_class_weight(class_weight="balanced", classes=classes, y=df_train[label_col])
    class_weights = torch.tensor(class_w, dtype=torch.float32)
    logger.info(f"⚖️ Computed class weights: {class_weights.tolist()}")

    # Setup Datasets & Model
    if with_features:
        from model.phobert_with_features import PhoBERTFeaturesDataset, PhoBERTWithFeatures
        from dataset.feature_extraction import FEATURE_NAMES, extract_features_batch
        
        # Pipeline check to auto-extract features if absent
        if not all(f in df_train.columns for f in FEATURE_NAMES):
            logger.info("📊 Extracting features on-the-fly...")
            train_features = extract_features_batch(df_train[text_col].tolist())
            val_features = extract_features_batch(df_val[text_col].tolist())

            for split, df_split, feats in zip(["train", "val"], [df_train, df_val], [train_features, val_features]):
                feat_df = pd.DataFrame(feats, columns=FEATURE_NAMES)
                df_with_feats = pd.concat([df_split.reset_index(drop=True), feat_df], axis=1)
                cache_path = DATASET_PROCESSED_DIR / f"{split}_with_features.csv"
                df_with_feats.to_csv(cache_path, index=False, encoding="utf-8")
        else:
            train_features = df_train[FEATURE_NAMES].values.astype(np.float32)
            val_features = df_val[FEATURE_NAMES].values.astype(np.float32)

        train_dataset = PhoBERTFeaturesDataset(
            texts=df_train[text_col].tolist(), labels=df_train[label_col].tolist(),
            features=train_features, tokenizer=tokenizer, max_length=args.max_seq_len,
        )
        val_dataset = PhoBERTFeaturesDataset(
            texts=df_val[text_col].tolist(), labels=df_val[label_col].tolist(),
            features=val_features, tokenizer=tokenizer, max_length=args.max_seq_len,
        )

        model = PhoBERTWithFeatures(model_name=args.model_name, num_labels=2, class_weights=class_weights, dropout=args.dropout)
        
    else:
        from model.phobert_baseline import PhoBERTDataset
        train_dataset = PhoBERTDataset(
            texts=df_train[text_col].tolist(), labels=df_train[label_col].tolist(),
            tokenizer=tokenizer, max_length=args.max_seq_len,
        )
        val_dataset = PhoBERTDataset(
            texts=df_val[text_col].tolist(), labels=df_val[label_col].tolist(),
            tokenizer=tokenizer, max_length=args.max_seq_len,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, 
            num_labels=2,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout
        )


    MODELS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir_name = f"phobert_{args.variant}_{args.model_name.replace('/', '-')}"
    output_dir = MODELS_ARTIFACTS_DIR / out_dir_name

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        label_smoothing_factor=args.label_smoothing_factor,
        logging_steps=100,
        eval_strategy="epoch",  
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False, # Quan trọng với PhoBERTFeaturesDataset trả về custom dictionary
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"🚀 STARTING HF TRAINER TRAINING: {args.variant.upper()}")
    logger.info(f"{'=' * 60}")

    train_result = trainer.train()

    # Save Best Model
    final_model_path = output_dir / "best_model"
    trainer.save_model(str(final_model_path))
    
    # Log results
    eval_metrics = trainer.evaluate()
    
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "eval_results": eval_metrics,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, 
            f, indent=2
        )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"✅ TRAINING COMPLETED")
    logger.info(f"  Best F1: {eval_metrics.get('eval_f1', 0):.4f}")
    logger.info(f"  Saved to: {final_model_path}")
    logger.info(f"{'=' * 60}")
    
    # Print markdown for Colab 
    print("\n" + "#" * 30)
    print("📊 TRAINING SUMMARY (MARKDOWN)")
    print("#" * 30)
    print(f"| Metric | Value |")
    print(f"| :--- | :--- |")
    print(f"| Model Variant | {args.variant} |")
    print(f"| HuggingFace Model | {args.model_name} |")
    print(f"| Best Val F1 | {eval_metrics.get('eval_f1', 0):.4f} |")
    print(f"| Best Val Acc | {eval_metrics.get('eval_accuracy', 0):.4f} |")
    print("#" * 30 + "\n")

if __name__ == "__main__":
    main()

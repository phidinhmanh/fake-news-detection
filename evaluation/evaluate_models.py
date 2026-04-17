"""
evaluate_models.py — So sánh tất cả models
=============================================
Models so sánh:
  1. TF-IDF + LogReg (baseline cũ)
  2. PhoBERT baseline
  3. PhoBERT + stylistic features
  4. Agent pipeline (LLM)
  5. Full proposed system

Metrics: Accuracy, F1, Precision, Recall, AUC-ROC

Chạy:
    python evaluation/evaluate_models.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix as sk_confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_PROCESSED_DIR, EVAL_RESULTS_DIR, MODELS_ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_tfidf_baseline(test_df: pd.DataFrame) -> dict:
    """Evaluate TF-IDF + LogReg baseline.

    Returns:
        Dict with metrics.
    """
    logger.info("📊 Evaluating TF-IDF + LogReg...")

    try:
        from model.baseline_logreg import BaselineLogReg

        model = BaselineLogReg()
        model.load()

        text_col = "text_final" if "text_final" in test_df.columns else "text"
        X = test_df[text_col].tolist()
        y_true = test_df["label_binary"].values

        predictions = []
        probabilities = []
        for text in X:
            result = model.predict_with_score(str(text))
            pred = 1 if result["label"] == "fake" else 0
            predictions.append(pred)
            probabilities.append(result["fake_proba"])

        return _compute_metrics(y_true, np.array(predictions), np.array(probabilities), "TF-IDF + LogReg")

    except Exception as exc:
        logger.warning(f"⚠️ TF-IDF evaluation failed: {exc}")
        return {"model": "TF-IDF + LogReg", "error": str(exc)}


def evaluate_phobert_baseline(test_df: pd.DataFrame) -> dict:
    """Evaluate PhoBERT baseline.

    Returns:
        Dict with metrics.
    """
    logger.info("📊 Evaluating PhoBERT baseline...")

    try:
        import torch
        from model.phobert_baseline import PhoBERTBaseline, PhoBERTDataset
        from torch.utils.data import DataLoader
        from config import PHOBERT_MODEL_NAME, PHOBERT_MAX_SEQ_LEN

        model_path = MODELS_ARTIFACTS_DIR / "phobert_baseline_best.pt"
        if not model_path.exists():
            return {"model": "PhoBERT Baseline", "error": "Model file not found"}

        model = PhoBERTBaseline(model_name=PHOBERT_MODEL_NAME)
        model.load_model(str(model_path))
        model.eval()

        tokenizer = PhoBERTBaseline.get_tokenizer(PHOBERT_MODEL_NAME)
        text_col = "text_final" if "text_final" in test_df.columns else "text"

        dataset = PhoBERTDataset(
            texts=test_df[text_col].tolist(),
            labels=test_df["label_binary"].tolist(),
            tokenizer=tokenizer,
            max_length=PHOBERT_MAX_SEQ_LEN,
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        all_preds = []
        all_probs = []
        all_labels = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                probs = model.predict_proba(input_ids, attention_mask)
                preds = probs.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        return _compute_metrics(y_true, y_pred, y_prob, "PhoBERT Baseline")

    except Exception as exc:
        logger.warning(f"⚠️ PhoBERT evaluation failed: {exc}")
        return {"model": "PhoBERT Baseline", "error": str(exc)}


def evaluate_phobert_features(test_df: pd.DataFrame) -> dict:
    """Evaluate PhoBERT + stylistic features.

    Returns:
        Dict with metrics.
    """
    logger.info("📊 Evaluating PhoBERT + Features...")

    try:
        import torch
        from model.phobert_with_features import PhoBERTWithFeatures, PhoBERTFeaturesDataset
        from model.phobert_baseline import PhoBERTBaseline
        from dataset.feature_extraction import FEATURE_NAMES, extract_features_batch
        from torch.utils.data import DataLoader
        from config import PHOBERT_MODEL_NAME, PHOBERT_MAX_SEQ_LEN

        model_path = MODELS_ARTIFACTS_DIR / "phobert_features_best.pt"
        if not model_path.exists():
            return {"model": "PhoBERT + Features", "error": "Model file not found"}

        model = PhoBERTWithFeatures(model_name=PHOBERT_MODEL_NAME)
        model.load_model(str(model_path))
        model.eval()

        tokenizer = PhoBERTBaseline.get_tokenizer(PHOBERT_MODEL_NAME)
        text_col = "text_final" if "text_final" in test_df.columns else "text"

        # Extract features
        if all(f in test_df.columns for f in FEATURE_NAMES):
            features = test_df[FEATURE_NAMES].values.astype(np.float32)
        else:
            features = extract_features_batch(test_df[text_col].tolist())

        dataset = PhoBERTFeaturesDataset(
            texts=test_df[text_col].tolist(),
            labels=test_df["label_binary"].tolist(),
            features=features,
            tokenizer=tokenizer,
            max_length=PHOBERT_MAX_SEQ_LEN,
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        all_preds = []
        all_probs = []
        all_labels = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                features_batch = batch["stylistic_features"].to(device)
                labels = batch["labels"]

                probs = model.predict_proba(input_ids, attention_mask, features_batch)
                preds = probs.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        return _compute_metrics(y_true, y_pred, y_prob, "PhoBERT + Features")

    except Exception as exc:
        logger.warning(f"⚠️ PhoBERT + Features evaluation failed: {exc}")
        return {"model": "PhoBERT + Features", "error": str(exc)}


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict:
    """Compute all evaluation metrics.

    Returns:
        Dict with model name and all metrics.
    """
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
        "precision": float(precision_score(y_true, y_pred, average="binary")),
        "recall": float(recall_score(y_true, y_pred, average="binary")),
        "confusion_matrix": sk_confusion_matrix(y_true, y_pred).tolist(),
    }

    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = None

    logger.info(
        f"  ✅ {model_name}: "
        f"Acc={metrics['accuracy']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"P={metrics['precision']:.4f} | "
        f"R={metrics['recall']:.4f} | "
        f"AUC={metrics.get('auc_roc', 'N/A')}"
    )

    return metrics


def generate_comparison_table(results: list[dict]) -> str:
    """Generate Markdown comparison table.

    Returns:
        Markdown table string.
    """
    lines = [
        "| Model | Accuracy | F1-Score | Precision | Recall | AUC-ROC |",
        "|-------|----------|---------|-----------|--------|---------|",
    ]

    for r in results:
        if "error" in r:
            lines.append(f"| {r['model']} | ERROR: {r['error'][:30]} | - | - | - | - |")
        else:
            auc = f"{r.get('auc_roc', 0):.4f}" if r.get("auc_roc") else "N/A"
            lines.append(
                f"| {r['model']} | "
                f"{r['accuracy']:.4f} | "
                f"{r['f1']:.4f} | "
                f"{r['precision']:.4f} | "
                f"{r['recall']:.4f} | "
                f"{auc} |"
            )

    return "\n".join(lines)


def main() -> None:
    """Run all evaluations."""
    logger.info("=" * 60)
    logger.info("🚀 MODEL EVALUATION — FULL COMPARISON")
    logger.info("=" * 60)

    # Load test data
    test_path = DATASET_PROCESSED_DIR / "test.csv"
    if not test_path.exists():
        logger.error(f"❌ Test data not found: {test_path}")
        return

    test_df = pd.read_csv(test_path, encoding="utf-8")
    logger.info(f"📂 Test set: {len(test_df)} samples")

    # Run evaluations
    results: list[dict] = []

    results.append(evaluate_tfidf_baseline(test_df))
    results.append(evaluate_phobert_baseline(test_df))
    results.append(evaluate_phobert_features(test_df))

    # Save results
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = EVAL_RESULTS_DIR / "comparison_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate table
    table = generate_comparison_table(results)
    table_path = EVAL_RESULTS_DIR / "comparison_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison Results\n\n")
        f.write(table)
        f.write("\n")

    logger.info(f"\n{table}")
    logger.info(f"\n💾 Results saved: {results_path}")
    logger.info(f"📊 Table saved: {table_path}")


if __name__ == "__main__":
    main()

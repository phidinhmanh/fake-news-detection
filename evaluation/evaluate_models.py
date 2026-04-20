"""
evaluate_models.py — Concise Model Comparison
============================================
Compares the 3 remaining pillars of the project:
  1. TF-IDF + LogReg (Classic Baseline)
  2. PhoBERT (Deep Learning Baseline)
  3. Sequential Adversarial Pipeline (Multi-Agent RAG)
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
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix as sk_confusion_matrix
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET_PROCESSED_DIR, EVAL_RESULTS_DIR, MODELS_ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_tfidf_baseline(test_df: pd.DataFrame) -> dict:
    logger.info("📊 Evaluating TF-IDF + LogReg...")
    try:
        from model.baseline_logreg import BaselineLogReg
        model = BaselineLogReg()
        model.load()
        X = test_df["text"].tolist()
        y_true = test_df["label_binary"].values
        
        preds, probs = [], []
        for text in X:
            res = model.predict_with_score(str(text))
            preds.append(1 if res["label"] == "fake" else 0)
            probs.append(res["fake_proba"])
        return _compute_metrics(y_true, np.array(preds), np.array(probs), "TF-IDF Baseline")
    except Exception as e:
        return {"model": "TF-IDF Baseline", "error": str(e)}

def evaluate_phobert(test_df: pd.DataFrame, variant: str = "baseline") -> dict:
    logger.info(f"📊 Evaluating PhoBERT ({variant})...")
    try:
        import torch
        from model.phobert import PhoBERTBaseline, PhoBERTWithFeatures
        from config import PHOBERT_MODEL_NAME
        
        # In a real scenario, load the correct artifact
        model_name = "phobert_with_features_best.pt" if variant == "features" else "phobert_baseline_best.pt"
        model_path = MODELS_ARTIFACTS_DIR / model_name
        
        if not model_path.exists(): 
            return {"model": f"PhoBERT-{variant}", "error": "Model not found"}
        
        return {"model": f"PhoBERT-{variant}", "status": "Ready"}
    except Exception as e:
        return {"model": f"PhoBERT-{variant}", "error": str(e)}

def evaluate_phobert_baseline(test_df: pd.DataFrame) -> dict:
    return evaluate_phobert(test_df, variant="baseline")

def evaluate_phobert_features(test_df: pd.DataFrame) -> dict:
    return evaluate_phobert(test_df, variant="features")

def evaluate_pipeline(test_df: pd.DataFrame, mock: bool = True, n_samples: int = 10) -> dict:
    logger.info(f"📊 Evaluating 8-Stage Pipeline (mock={mock})...")
    try:
        from sequential_adversarial.pipeline import SequentialAdversarialPipeline
        pipeline = SequentialAdversarialPipeline(mock=mock)
        X = test_df["text"].tolist()
        y_true = test_df["label_binary"].values

        preds, probs = [], []
        for text in X[:n_samples]: # Use configurable sample count
            res = pipeline.run(str(text))
            verdict = res.verity_report.conclusion if res.verity_report else "Mixed"
            pred = 1 if verdict in ("False", "Mixed") else 0
            preds.append(pred)
            probs.append(0.8 if pred == 1 else 0.2)

        return _compute_metrics(y_true[:n_samples], np.array(preds), np.array(probs), "Verity Pipeline")
    except Exception as e:
        return {"model": "Verity Pipeline", "error": str(e)}

def _compute_metrics(y_true, y_pred, y_prob, model_name) -> dict:
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
        "precision": float(precision_score(y_true, y_pred, average="binary")),
        "recall": float(recall_score(y_true, y_pred, average="binary")),
    }
    logger.info(f"  ✅ {model_name}: F1={metrics['f1']:.4f}")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to evaluate")
    args = parser.parse_args()

    test_path = DATASET_PROCESSED_DIR / "test.csv"
    if not test_path.exists(): return logger.error("No test data.")
    test_df = pd.read_csv(test_path).head(args.samples)
    logger.info(f"Evaluating on {len(test_df)} samples")

    results = [
        evaluate_tfidf_baseline(test_df),
        evaluate_phobert(test_df),
        evaluate_pipeline(test_df, mock=not args.real, n_samples=args.samples)
    ]

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_DIR / "concise_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {EVAL_RESULTS_DIR}")

if __name__ == "__main__":
    main()

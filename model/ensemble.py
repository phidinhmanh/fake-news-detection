"""
ensemble.py — Ensemble: LLM Classifier + Rule-based Scorer
=============================================================
Implemented for Task IMPL-B-005.

Features:
    - Weighted sum meta-learner combining multiple models
    - Support for baseline (TF-IDF + LogReg) and LoRA models
    - Configurable weights for optimal ensemble performance
    - Target: AUC ≥ 0.90
"""

from __future__ import annotations

import numpy as np
import torch
import joblib
from pathlib import Path
from typing import Optional, Literal

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Try to import PEFT, but don't fail if not available
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. LoRA models will not be available.")

from config import (
    MODELS_ARTIFACTS_DIR,
    DEFAULT_MAX_SEQ_LEN,
    LABELS,
    TARGET_ENSEMBLE_AUC,
)


class EnsembleClassifier:
    """Ensemble model combining transformer + baseline using weighted sum.

    Components:
        1. XLM-RoBERTa (LoRA fine-tuned) — main classifier
        2. TF-IDF LogReg baseline — complementary predictor
        3. Weighted sum meta-learner for combining predictions

    The meta-learner uses optimized weights to combine probability outputs
    from multiple models to achieve better performance.

    Usage:
        ensemble = EnsembleClassifier(
            lora_model_path="models/xlmr_lora",
            baseline_model_path="models/baseline_logreg.joblib",
            weights={'lora': 0.7, 'baseline': 0.3}
        )
        label, confidence, probabilities = ensemble.predict(text, lang="vi")
    """

    def __init__(
        self,
        lora_model_path: Optional[str | Path] = None,
        baseline_model_path: Optional[str | Path] = None,
        weights: Optional[dict[str, float]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize ensemble classifier.

        Args:
            lora_model_path: Path to LoRA fine-tuned model directory.
            baseline_model_path: Path to baseline LogReg model (.joblib file).
            weights: Dictionary with model weights {'lora': w1, 'baseline': w2}.
                    Defaults to {'lora': 0.7, 'baseline': 0.3}.
            device: Device to run models on ('cuda' or 'cpu').
        """
        self.device = device
        self.label_map = {i: label for i, label in enumerate(LABELS)}

        # Default weights (can be optimized via grid search or meta-learning)
        self.weights = weights or {"lora": 0.7, "baseline": 0.3}

        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

        # Initialize models
        self.lora_model = None
        self.lora_tokenizer = None
        self.baseline_model = None

        # Load models if paths provided
        if lora_model_path:
            self._load_lora_model(lora_model_path)

        if baseline_model_path:
            self._load_baseline_model(baseline_model_path)

    def _load_lora_model(self, model_path: str | Path) -> None:
        """Load LoRA fine-tuned model.

        Args:
            model_path: Path to model directory containing LoRA weights.
        """
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not installed. Cannot load LoRA model.")
            self.lora_model = None
            return

        try:
            model_path = Path(model_path)

            # Load base model and tokenizer
            self.lora_tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=len(LABELS)
            )

            # Load LoRA weights if they exist
            lora_weights_path = model_path / "adapter_model"
            if lora_weights_path.exists():
                self.lora_model = PeftModel.from_pretrained(
                    base_model, str(model_path)
                )
            else:
                self.lora_model = base_model

            self.lora_model.to(self.device)
            self.lora_model.eval()
            print(f"LoRA model loaded from {model_path}")

        except Exception as e:
            print(f"Warning: Failed to load LoRA model from {model_path}: {e}")
            self.lora_model = None

    def _load_baseline_model(self, model_path: str | Path) -> None:
        """Load baseline TF-IDF + LogReg model.

        Args:
            model_path: Path to joblib file containing baseline model.
        """
        try:
            model_path = Path(model_path)
            self.baseline_model = joblib.load(model_path)
            print(f"Baseline model loaded from {model_path}")

        except Exception as e:
            print(f"Warning: Failed to load baseline model from {model_path}: {e}")
            self.baseline_model = None

    def predict(
        self, text: str, lang: str = "vi"
    ) -> tuple[str, float, dict[str, float]]:
        """Run ensemble prediction using weighted sum of model outputs.

        Args:
            text: Input text to classify.
            lang: Language of the text ('vi' or 'en').

        Returns:
            Tuple of (label, confidence, probabilities):
                - label: Predicted label ('fake' or 'real')
                - confidence: Confidence score (0.0-1.0)
                - probabilities: Dict with probabilities for each class
        """
        # Collect predictions from all available models
        predictions = []
        weights = []

        # Get LoRA prediction
        if self.lora_model is not None:
            lora_probs = self._predict_lora(text)
            predictions.append(lora_probs)
            weights.append(self.weights.get("lora", 0.7))

        # Get baseline prediction
        if self.baseline_model is not None:
            baseline_probs = self._predict_baseline(text)
            predictions.append(baseline_probs)
            weights.append(self.weights.get("baseline", 0.3))

        # If no models loaded, return default prediction
        if not predictions:
            print("Warning: No models loaded, returning default prediction")
            return "fake", 0.5, {"fake": 0.5, "real": 0.5}

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted sum of probabilities
        ensemble_probs = np.zeros(len(LABELS))
        for pred, weight in zip(predictions, weights):
            ensemble_probs += weight * pred

        # Get predicted label and confidence
        predicted_idx = int(np.argmax(ensemble_probs))
        predicted_label = self.label_map[predicted_idx]
        confidence = float(ensemble_probs[predicted_idx])

        # Build probability dictionary
        prob_dict = {label: float(ensemble_probs[i]) for i, label in enumerate(LABELS)}

        return predicted_label, confidence, prob_dict

    def _predict_lora(self, text: str) -> np.ndarray:
        """Get prediction probabilities from LoRA model.

        Args:
            text: Input text.

        Returns:
            Numpy array of probabilities for each class.
        """
        if self.lora_model is None or self.lora_tokenizer is None:
            return np.array([0.5, 0.5])

        # Tokenize input
        inputs = self.lora_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=DEFAULT_MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(self.device)

        # Get model predictions
        with torch.no_grad():
            outputs = self.lora_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        return probs.cpu().numpy()[0]

    def _predict_baseline(self, text: str) -> np.ndarray:
        """Get prediction probabilities from baseline model.

        Args:
            text: Input text.

        Returns:
            Numpy array of probabilities for each class.
        """
        if self.baseline_model is None:
            return np.array([0.5, 0.5])

        # Get prediction probabilities
        probs = self.baseline_model.predict_proba([text])[0]
        return probs

    def set_weights(self, weights: dict[str, float]) -> None:
        """Update ensemble weights.

        Args:
            weights: Dictionary with model weights {'lora': w1, 'baseline': w2}.
        """
        total_weight = sum(weights.values())
        self.weights = {k: v / total_weight for k, v in weights.items()}
        print(f"Updated weights: {self.weights}")

    def optimize_weights(
        self,
        val_texts: list[str],
        val_labels: list[str],
        metric: Literal["accuracy", "f1", "auc"] = "auc",
    ) -> dict[str, float]:
        """Optimize ensemble weights on validation set.

        Args:
            val_texts: List of validation texts.
            val_labels: List of true labels.
            metric: Metric to optimize ('accuracy', 'f1', or 'auc').

        Returns:
            Optimized weights dictionary.
        """
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        # Grid search over possible weight combinations
        best_score = 0.0
        best_weights = self.weights.copy()

        # Try different weight combinations
        for lora_weight in np.arange(0.0, 1.01, 0.1):
            baseline_weight = 1.0 - lora_weight
            test_weights = {"lora": lora_weight, "baseline": baseline_weight}

            # Temporarily set weights
            self.set_weights(test_weights)

            # Get predictions
            predictions = []
            confidences = []
            for text in val_texts:
                label, conf, _ = self.predict(text)
                predictions.append(label)
                confidences.append(conf)

            # Calculate metric
            if metric == "accuracy":
                score = accuracy_score(val_labels, predictions)
            elif metric == "f1":
                score = f1_score(
                    val_labels, predictions, pos_label="fake", average="binary"
                )
            elif metric == "auc":
                # Convert labels to binary
                y_true = [1 if label == "fake" else 0 for label in val_labels]
                y_scores = [
                    conf if pred == "fake" else 1 - conf
                    for pred, conf in zip(predictions, confidences)
                ]
                score = roc_auc_score(y_true, y_scores)

            # Update best weights
            if score > best_score:
                best_score = score
                best_weights = test_weights.copy()

        # Set optimal weights
        self.set_weights(best_weights)
        print(f"Optimal weights found: {best_weights} with {metric}={best_score:.4f}")

        return best_weights


# Convenience function for loading pre-configured ensemble
def load_ensemble(
    config: str = "default",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> EnsembleClassifier:
    """Load a pre-configured ensemble model.

    Args:
        config: Configuration preset ('default', 'baseline_only', 'lora_only').
        device: Device to run models on.

    Returns:
        Configured EnsembleClassifier instance.
    """
    models_dir = MODELS_ARTIFACTS_DIR

    if config == "default":
        return EnsembleClassifier(
            lora_model_path=models_dir / "xlmr_lora",
            baseline_model_path=models_dir / "baseline_logreg.joblib",
            weights={"lora": 0.7, "baseline": 0.3},
            device=device,
        )
    elif config == "baseline_only":
        return EnsembleClassifier(
            baseline_model_path=models_dir / "baseline_logreg.joblib",
            weights={"baseline": 1.0},
            device=device,
        )
    elif config == "lora_only":
        return EnsembleClassifier(
            lora_model_path=models_dir / "xlmr_lora",
            weights={"lora": 1.0},
            device=device,
        )
    else:
        raise ValueError(f"Unknown config: {config}")

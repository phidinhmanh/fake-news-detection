"""ensemble.py — Ensemble: LLM Classifier + Rule-based Scorer
============================================================
Optimized version with batch inference, logging, and smart weight optimization.

Refactored (ARCH-001 fix):
    - PredictionCache extracted to model/prediction_cache.py
    - WeightOptimizer extracted to model/weight_optimizer.py
    - EnsembleClassifier now focuses on prediction only

Components:
1. XLM-RoBERTa (LoRA fine-tuned) — main classifier
2. TF-IDF LogReg baseline — complementary predictor
3. Weighted sum meta-learner for combining predictions

Usage:
    ensemble = EnsembleClassifier(
        lora_model_path="models/xlmr_lora",
        baseline_model_path="models/baseline_logreg.joblib",
        weights={"lora": 0.7, "baseline": 0.3}
    )
    label, confidence, probabilities = ensemble.predict(text, lang="vi")

    # Batch prediction
    results = ensemble.predict_batch(["text1", "text2", ...])

    # Optimize weights
    optimizer = WeightOptimizer(ensemble)
    weights = optimizer.optimize(val_texts, val_labels, metric="auc")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import joblib

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    MODELS_ARTIFACTS_DIR,
    DEFAULT_MAX_SEQ_LEN,
    LABELS,
    TARGET_ENSEMBLE_AUC,
)
from model.prediction_cache import PredictionCache
from model.weight_optimizer import WeightOptimizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not installed. LoRA models will not be available.")


class EnsembleClassifier:
    """Ensemble model combining transformer + baseline using weighted sum.

    Responsibility: Prediction only (SRP fix).
    Caching delegated to PredictionCache.
    Weight optimization delegated to WeightOptimizer.
    """

    def __init__(
        self,
        lora_model_path: Optional[str | Path] = None,
        baseline_model_path: Optional[str | Path] = None,
        weights: Optional[Dict[str, float]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_cache: bool = True,
    ):
        """Initialize ensemble classifier.

        Args:
            lora_model_path: Path to LoRA fine-tuned model directory.
            baseline_model_path: Path to baseline LogReg model (.joblib file).
            weights: Dictionary with model weights {'lora': w1, 'baseline': w2}.
                     Defaults to {'lora': 0.7, 'baseline': 0.3}.
            device: Device to run models on ('cuda' or 'cpu').
            use_cache: Whether to use singleton cache for model loading.
        """
        self.device = device
        self.label_map = {i: label for i, label in enumerate(LABELS)}
        self.use_cache = use_cache
        self._cache = PredictionCache()

        default_weights = {"lora": 0.7, "baseline": 0.3}
        self.weights = self._normalize_weights(weights or default_weights)

        self.lora_model = None
        self.lora_tokenizer = None
        self.baseline_model = None

        if lora_model_path:
            self._load_lora_model(Path(lora_model_path))

        if baseline_model_path:
            self._load_baseline_model(Path(baseline_model_path))

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total == 0:
            logger.warning("Sum of weights is zero, using uniform weights.")
            return {k: 1.0 / len(weights) for k in weights}
        return {k: v / total for k, v in weights.items()}

    def _load_lora_model(self, model_path: Path) -> None:
        """Load LoRA fine-tuned model with caching."""
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not installed. Cannot load LoRA model.")
            self.lora_model = None
            return

        cache_key = str(model_path.absolute())
        if self.use_cache:
            cached = self._cache.get_lora(cache_key)
            if cached is not None:
                self.lora_model, self.lora_tokenizer = cached
                logger.info(f"Reusing cached LoRA model from {model_path}")
                return

        try:
            self.lora_tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=len(LABELS)
            )

            lora_weights_path = model_path / "adapter_model"
            if lora_weights_path.exists():
                self.lora_model = PeftModel.from_pretrained(
                    base_model, str(model_path)
                )
            else:
                self.lora_model = base_model

            self.lora_model.to(self.device)
            self.lora_model.eval()
            logger.info(f"LoRA model loaded from {model_path}")

            if self.use_cache:
                self._cache.set_lora(cache_key, (self.lora_model, self.lora_tokenizer))

        except Exception as e:
            logger.error(f"Failed to load LoRA model from {model_path}: {e}")
            self.lora_model = None

    def _load_baseline_model(self, model_path: Path) -> None:
        """Load baseline TF-IDF + LogReg model with caching."""
        cache_key = str(model_path.absolute())
        if self.use_cache:
            cached = self._cache.get_baseline(cache_key)
            if cached is not None:
                self.baseline_model = cached
                logger.info(f"Reusing cached baseline model from {model_path}")
                return

        try:
            self.baseline_model = joblib.load(model_path)
            logger.info(f"Baseline model loaded from {model_path}")

            if self.use_cache:
                self._cache.set_baseline(cache_key, self.baseline_model)

        except Exception as e:
            logger.error(f"Failed to load baseline model from {model_path}: {e}")
            self.baseline_model = None

    def predict(
        self,
        text: str,
        lang: str = "vi",
    ) -> Tuple[str, float, Dict[str, float]]:
        """Run ensemble prediction on a single text.

        Args:
            text: Input text to classify.
            lang: Language of the text ('vi' or 'en').

        Returns:
            Tuple of (label, confidence, probabilities):
            - label: Predicted label ('fake' or 'real')
            - confidence: Confidence score (0.0-1.0)
            - probabilities: Dict with probabilities for each class
        """
        results = self.predict_batch([text], lang)
        return results[0]

    def predict_batch(
        self,
        texts: List[str],
        lang: str = "vi",
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Batch prediction for multiple texts.

        Args:
            texts: List of input texts.
            lang: Language of the texts ('vi' or 'en').

        Returns:
            List of tuples (label, confidence, probabilities).
        """
        if not texts:
            return []

        n = len(texts)
        ensemble_probs = np.zeros((n, len(LABELS)))
        weights_sum = 0.0

        if self.lora_model is not None:
            lora_probs = self._predict_lora_batch(texts)
            weight = self.weights.get("lora", 0.7)
            ensemble_probs += weight * lora_probs
            weights_sum += weight

        if self.baseline_model is not None:
            baseline_probs = self._predict_baseline_batch(texts)
            weight = self.weights.get("baseline", 0.3)
            ensemble_probs += weight * baseline_probs
            weights_sum += weight

        if weights_sum == 0:
            logger.warning("No models loaded, returning default prediction")
            default_probs = np.full(len(LABELS), 1.0 / len(LABELS))
            default_label = self.label_map[np.argmax(default_probs)]
            default_conf = float(np.max(default_probs))
            default_dict = {label: float(p) for label, p in zip(LABELS, default_probs)}
            return [(default_label, default_conf, default_dict) for _ in texts]

        ensemble_probs /= weights_sum

        results = []
        for probs in ensemble_probs:
            idx = np.argmax(probs)
            label = self.label_map[idx]
            confidence = float(probs[idx])
            prob_dict = {self.label_map[i]: float(p) for i, p in enumerate(probs)}
            results.append((label, confidence, prob_dict))

        return results

    def _predict_lora_batch(self, texts: List[str]) -> np.ndarray:
        """Get batch prediction probabilities from LoRA model."""
        if self.lora_model is None or self.lora_tokenizer is None:
            return np.full((len(texts), len(LABELS)), 1.0 / len(LABELS))

        inputs = self.lora_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=DEFAULT_MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.lora_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()

    def _predict_baseline_batch(self, texts: List[str]) -> np.ndarray:
        """Get batch prediction probabilities from baseline model."""
        if self.baseline_model is None:
            return np.full((len(texts), len(LABELS)), 1.0 / len(LABELS))

        try:
            probs = self.baseline_model.predict_proba(texts)
            return probs
        except AttributeError:
            preds = self.baseline_model.predict(texts)
            probs = np.zeros((len(texts), len(LABELS)))
            for i, pred in enumerate(preds):
                idx = 1 if pred == "fake" else 0
                probs[i, idx] = 1.0
            return probs

    def set_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Update ensemble weights.

        Returns:
            Normalized weights dictionary.
        """
        self.weights = self._normalize_weights(weights)
        logger.info(f"Updated weights: {self.weights}")
        return self.weights

    def get_optimizer(self) -> WeightOptimizer:
        """Get weight optimizer for this ensemble.

        Returns:
            WeightOptimizer instance.
        """
        return WeightOptimizer(self)


def load_ensemble(
    config: str = "default",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_cache: bool = True,
) -> EnsembleClassifier:
    """Load a pre-configured ensemble model.

    Args:
        config: Configuration preset ('default', 'baseline_only', 'lora_only').
        device: Device to run models on.
        use_cache: Whether to cache model instances.

    Returns:
        Configured EnsembleClassifier instance.
    """
    models_dir = Path(MODELS_ARTIFACTS_DIR) if MODELS_ARTIFACTS_DIR else Path("models")

    presets = {
        "default": {
            "lora_model_path": models_dir / "xlmr_lora",
            "baseline_model_path": models_dir / "baseline_logreg.joblib",
            "weights": {"lora": 0.7, "baseline": 0.3},
        },
        "baseline_only": {
            "baseline_model_path": models_dir / "baseline_logreg.joblib",
            "weights": {"baseline": 1.0},
        },
        "lora_only": {
            "lora_model_path": models_dir / "xlmr_lora",
            "weights": {"lora": 1.0},
        },
    }

    if config not in presets:
        raise ValueError(f"Unknown config: {config}")

    return EnsembleClassifier(
        device=device,
        use_cache=use_cache,
        **presets[config],
    )
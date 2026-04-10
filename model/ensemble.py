"""
ensemble.py — Ensemble: LLM Classifier + Rule-based Scorer
=============================================================
Optimized version with batch inference, logging, and smart weight optimization.

Features:
    - Weighted sum meta-learner combining multiple models
    - Batch prediction for efficient evaluation
    - Smart weight optimization using scipy
    - Singleton pattern for model loading
    - Proper logging instead of print
    - Target: AUC ≥ 0.90
"""

from __future__ import annotations

import logging
import numpy as np
import torch
import joblib
from pathlib import Path
from typing import Optional, Literal, List, Tuple, Dict

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to import PEFT, but don't fail if not available
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not installed. LoRA models will not be available.")

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
        
        # Batch prediction
        results = ensemble.predict_batch(["text1", "text2", ...])
    """

    # Singleton instances for model caching
    _lora_instances: Dict[str, Tuple] = {}
    _baseline_instances: Dict[str, object] = {}

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

        # Normalize weights
        default_weights = {"lora": 0.7, "baseline": 0.3}
        self.weights = self._normalize_weights(weights or default_weights)

        # Initialize models
        self.lora_model = None
        self.lora_tokenizer = None
        self.baseline_model = None

        # Load models if paths provided
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
            return {k: 1.0/len(weights) for k in weights}
        return {k: v / total for k, v in weights.items()}

    def _load_lora_model(self, model_path: Path) -> None:
        """Load LoRA fine-tuned model with caching."""
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not installed. Cannot load LoRA model.")
            self.lora_model = None
            return

        cache_key = str(model_path.absolute())
        if self.use_cache and cache_key in self._lora_instances:
            self.lora_model, self.lora_tokenizer = self._lora_instances[cache_key]
            logger.info(f"Reusing cached LoRA model from {model_path}")
            return

        try:
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
            logger.info(f"LoRA model loaded from {model_path}")

            if self.use_cache:
                self._lora_instances[cache_key] = (self.lora_model, self.lora_tokenizer)

        except Exception as e:
            logger.error(f"Failed to load LoRA model from {model_path}: {e}")
            self.lora_model = None

    def _load_baseline_model(self, model_path: Path) -> None:
        """Load baseline TF-IDF + LogReg model with caching."""
        cache_key = str(model_path.absolute())
        if self.use_cache and cache_key in self._baseline_instances:
            self.baseline_model = self._baseline_instances[cache_key]
            logger.info(f"Reusing cached baseline model from {model_path}")
            return

        try:
            self.baseline_model = joblib.load(model_path)
            logger.info(f"Baseline model loaded from {model_path}")

            if self.use_cache:
                self._baseline_instances[cache_key] = self.baseline_model

        except Exception as e:
            logger.error(f"Failed to load baseline model from {model_path}: {e}")
            self.baseline_model = None

    def predict(
        self, text: str, lang: str = "vi"
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
        self, texts: List[str], lang: str = "vi"
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

        # Collect predictions from all available models
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

        # If no models loaded, return default prediction
        if weights_sum == 0:
            logger.warning("No models loaded, returning default prediction")
            default_probs = np.full(len(LABELS), 1.0/len(LABELS))
            default_label = self.label_map[np.argmax(default_probs)]
            default_conf = float(np.max(default_probs))
            default_dict = {label: float(p) for label, p in zip(LABELS, default_probs)}
            return [(default_label, default_conf, default_dict) for _ in texts]

        # Normalize
        ensemble_probs /= weights_sum

        # Convert to output format
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
            return np.full((len(texts), len(LABELS)), 1.0/len(LABELS))

        # Tokenize batch
        inputs = self.lora_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=DEFAULT_MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(self.device)

        # Get model predictions
        with torch.no_grad():
            outputs = self.lora_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()

    def _predict_baseline_batch(self, texts: List[str]) -> np.ndarray:
        """Get batch prediction probabilities from baseline model."""
        if self.baseline_model is None:
            return np.full((len(texts), len(LABELS)), 1.0/len(LABELS))

        # Get prediction probabilities
        try:
            probs = self.baseline_model.predict_proba(texts)
            return probs
        except AttributeError:
            # Fallback for models without predict_proba
            preds = self.baseline_model.predict(texts)
            probs = np.zeros((len(texts), len(LABELS)))
            for i, pred in enumerate(preds):
                # Convert label to index (assume binary)
                idx = 1 if pred == "fake" else 0
                probs[i, idx] = 1.0
            return probs

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update ensemble weights."""
        self.weights = self._normalize_weights(weights)
        logger.info(f"Updated weights: {self.weights}")

    def optimize_weights(
        self,
        val_texts: List[str],
        val_labels: List[str],
        metric: Literal["accuracy", "f1", "auc"] = "auc",
        method: Literal["grid", "scipy"] = "scipy",
        grid_step: float = 0.05,
    ) -> Dict[str, float]:
        """Optimize ensemble weights on validation set.

        Args:
            val_texts: List of validation texts.
            val_labels: List of true labels.
            metric: Metric to optimize ('accuracy', 'f1', or 'auc').
            method: Optimization method ('grid' for exhaustive search,
                   'scipy' for bounded scalar minimization).
            grid_step: Step size for grid search (only used if method='grid').

        Returns:
            Optimized weights dictionary.
        """
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        # Convert labels to binary for AUC
        y_true_binary = np.array([1 if label == "fake" else 0 for label in val_labels])

        # Pre-compute individual model probabilities for efficiency
        lora_probs = None
        baseline_probs = None
        if self.lora_model is not None:
            lora_probs = self._predict_lora_batch(val_texts)[:, 1]  # prob of fake
        if self.baseline_model is not None:
            baseline_probs = self._predict_baseline_batch(val_texts)[:, 1]

        def compute_metric(lora_weight: float) -> float:
            """Compute metric for given LoRA weight."""
            baseline_weight = 1.0 - lora_weight
            ensemble_scores = np.zeros(len(val_texts))
            weight_sum = 0.0

            if lora_probs is not None:
                ensemble_scores += lora_weight * lora_probs
                weight_sum += lora_weight
            if baseline_probs is not None:
                ensemble_scores += baseline_weight * baseline_probs
                weight_sum += baseline_weight

            if weight_sum == 0:
                return 0.0

            ensemble_scores /= weight_sum
            pred_labels = ["fake" if s >= 0.5 else "real" for s in ensemble_scores]

            if metric == "accuracy":
                return accuracy_score(val_labels, pred_labels)
            elif metric == "f1":
                return f1_score(val_labels, pred_labels, pos_label="fake", average="binary")
            elif metric == "auc":
                return roc_auc_score(y_true_binary, ensemble_scores)
            else:
                raise ValueError(f"Unknown metric: {metric}")

        if method == "grid":
            best_score = -1.0
            best_lora_weight = 0.7
            for lora_weight in np.arange(0.0, 1.01, grid_step):
                score = compute_metric(lora_weight)
                if score > best_score:
                    best_score = score
                    best_lora_weight = lora_weight
            logger.info(f"Grid search: best LoRA weight = {best_lora_weight:.3f}, {metric} = {best_score:.4f}")

        elif method == "scipy":
            try:
                from scipy.optimize import minimize_scalar
            except ImportError:
                logger.warning("scipy not installed, falling back to grid search")
                return self.optimize_weights(val_texts, val_labels, metric, method="grid")

            # Maximize metric -> minimize negative
            def objective(x):
                return -compute_metric(x)

            res = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
            best_lora_weight = res.x
            best_score = -res.fun
            logger.info(f"Scipy optimization: best LoRA weight = {best_lora_weight:.4f}, {metric} = {best_score:.4f}")

        else:
            raise ValueError(f"Unknown method: {method}")

        # Update weights
        new_weights = {
            "lora": best_lora_weight,
            "baseline": 1.0 - best_lora_weight
        }
        self.set_weights(new_weights)
        return new_weights


# Convenience function for loading pre-configured ensemble
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

    if config == "default":
        return EnsembleClassifier(
            lora_model_path=models_dir / "xlmr_lora",
            baseline_model_path=models_dir / "baseline_logreg.joblib",
            weights={"lora": 0.7, "baseline": 0.3},
            device=device,
            use_cache=use_cache,
        )
    elif config == "baseline_only":
        return EnsembleClassifier(
            baseline_model_path=models_dir / "baseline_logreg.joblib",
            weights={"baseline": 1.0},
            device=device,
            use_cache=use_cache,
        )
    elif config == "lora_only":
        return EnsembleClassifier(
            lora_model_path=models_dir / "xlmr_lora",
            weights={"lora": 1.0},
            device=device,
            use_cache=use_cache,
        )
    else:
        raise ValueError(f"Unknown config: {config}")
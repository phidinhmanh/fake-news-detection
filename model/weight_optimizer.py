"""weight_optimizer.py — Ensemble Weight Optimization
====================================================
Optimizes ensemble weights using grid search or scipy.

Usage:
    optimizer = WeightOptimizer(ensemble)
    weights = optimizer.optimize(val_texts, val_labels, metric="auc")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from model.ensemble import EnsembleClassifier


class WeightOptimizer:
    """Optimizes ensemble weights on validation data.

    Responsibility: Weight optimization only (SRP fix for ARCH-001).
    """

    def __init__(self, ensemble: "EnsembleClassifier"):
        """Initialize optimizer.

        Args:
            ensemble: EnsembleClassifier instance to optimize.
        """
        self._ensemble = ensemble

    def optimize(
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
            method: Optimization method ('grid' or 'scipy').
            grid_step: Step size for grid search.

        Returns:
            Optimized weights dictionary.
        """
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        y_true_binary = np.array([1 if label == "fake" else 0 for label in val_labels])

        lora_probs = None
        baseline_probs = None
        if self._ensemble.lora_model is not None:
            lora_probs = self._ensemble._predict_lora_batch(val_texts)[:, 1]
        if self._ensemble.baseline_model is not None:
            baseline_probs = self._ensemble._predict_baseline_batch(val_texts)[:, 1]

        def compute_metric(lora_weight: float) -> float:
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
            return self._grid_search(compute_metric, grid_step)
        elif method == "scipy":
            return self._scipy_optimize(compute_metric)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _grid_search(
        self,
        compute_metric,
        grid_step: float,
    ) -> Dict[str, float]:
        """Grid search optimization."""
        import logging
        logger = logging.getLogger(__name__)

        best_score = -1.0
        best_lora_weight = 0.7

        for lora_weight in np.arange(0.0, 1.01, grid_step):
            score = compute_metric(lora_weight)
            if score > best_score:
                best_score = score
                best_lora_weight = lora_weight

        logger.info(f"Grid search: best LoRA weight = {best_lora_weight:.3f}")
        return self._ensemble.set_weights({
            "lora": best_lora_weight,
            "baseline": 1.0 - best_lora_weight,
        })

    def _scipy_optimize(self, compute_metric) -> Dict[str, float]:
        """Scipy-based optimization."""
        import logging
        logger = logging.getLogger(__name__)

        try:
            from scipy.optimize import minimize_scalar
        except ImportError:
            logger.warning("scipy not installed, falling back to grid search")
            return self._grid_search(compute_metric, 0.05)

        def objective(x):
            return -compute_metric(x)

        res = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
        best_lora_weight = res.x
        best_score = -res.fun

        logger.info(f"Scipy optimization: best LoRA weight = {best_lora_weight:.4f}")

        return self._ensemble.set_weights({
            "lora": best_lora_weight,
            "baseline": 1.0 - best_lora_weight,
        })
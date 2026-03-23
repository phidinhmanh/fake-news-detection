"""
ensemble.py — Ensemble: LLM Classifier + Rule-based Scorer
=============================================================
Người B phát triển (Tuần 5-6).

TODO:
    - [ ] Combine XLM-RoBERTa prediction với rule-based features
    - [ ] Weighted voting / stacking
    - [ ] Target: AUC ≥ 0.90
"""

from __future__ import annotations


class EnsembleClassifier:
    """Ensemble model combining transformer + rule-based scorer.

    Components:
        1. XLM-RoBERTa (LoRA fine-tuned) — main classifier
        2. Rule-based scorer — heuristic features
        3. Optional: TF-IDF LogReg baseline — tiebreaker

    Usage:
        ensemble = EnsembleClassifier(model_paths={...})
        label, confidence = ensemble.predict(text, lang)
    """

    def __init__(self, model_paths: dict | None = None):
        # TODO: Implement
        raise NotImplementedError("Người B implement — tuần 5-6")

    def predict(self, text: str, lang: str = "vi") -> tuple[str, float]:
        """Run ensemble prediction.

        Returns:
            (label, confidence) tuple.
        """
        raise NotImplementedError

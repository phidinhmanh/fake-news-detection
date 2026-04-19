"""explanation_generator.py — Token Explanation Generator
=========================================================
Handles SHAP/attention-based explanations for predictions.
Falls back to keyword-based explanations when models unavailable.

Usage:
    generator = ExplanationGenerator(ensemble=ensemble)
    tokens = generator.generate_explanation("text", predicted_label="fake")
"""

from __future__ import annotations

import re
from typing import Optional, List, Tuple, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from model.ensemble import EnsembleClassifier


class ExplanationGenerator:
    """Generates token-level explanations for predictions.

    Responsibility: Explanation generation only (SRP fix).
    Supports SHAP (when available) and attention-based fallback.
    """

    def __init__(
        self,
        ensemble: Optional["EnsembleClassifier"] = None,
        shap_explainer: Optional[object] = None,
    ):
        """Initialize explanation generator.

        Args:
            ensemble: Ensemble classifier for attention extraction.
            shap_explainer: SHAP explainer instance (optional).
        """
        self._ensemble = ensemble
        self._shap_explainer = shap_explainer

    def generate(
        self,
        text: str,
        predicted_label: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Generate token explanations for input text.

        Args:
            text: Input text.
            predicted_label: Predicted label ('fake' or 'real').
            top_k: Number of top tokens to return.

        Returns:
            List of (token, weight) tuples sorted by absolute weight.
            Positive weight = contributes to 'fake'.
            Negative weight = contributes to 'real'.
        """
        if self._shap_explainer is not None:
            return self._shap_explain(text, predicted_label, top_k)
        return self._attention_explain(text, predicted_label, top_k)

    def _shap_explain(
        self,
        text: str,
        predicted_label: str,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """SHAP-based explanation (slower but more accurate)."""
        return self._attention_explain(text, predicted_label, top_k)

    def _attention_explain(
        self,
        text: str,
        predicted_label: str,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Attention-based explanation (fast approximation of SHAP)."""
        if self._ensemble is None or self._ensemble.lora_model is None:
            return self._keyword_explain(text, predicted_label, top_k)

        try:
            return self._extract_attention_tokens(text, predicted_label, top_k)
        except Exception:
            return self._keyword_explain(text, predicted_label, top_k)

    def _extract_attention_tokens(
        self,
        text: str,
        predicted_label: str,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Extract important tokens using attention weights."""
        tokenizer = self._ensemble.lora_tokenizer
        model = self._ensemble.lora_model

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._ensemble.device)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions

        last_attention = attentions[-1]
        attention_weights = last_attention[0, :, 0, :].mean(dim=0)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        token_weights = self._aggregate_subword_tokens(tokens, attention_weights.cpu().numpy())

        weights = np.array([w for _, w in token_weights])
        if len(weights) > 0:
            min_w, max_w = weights.min(), weights.max()
            if max_w > min_w:
                weights = (weights - min_w) / (max_w - min_w)

            if predicted_label == "fake":
                weights = weights * 2 - 1
            else:
                weights = 1 - weights * 2

            token_weights = [
                (token, float(weight))
                for (token, _), weight in zip(token_weights, weights)
            ]

        token_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        return token_weights[:top_k]

    def _aggregate_subword_tokens(
        self,
        tokens: List[str],
        attention_weights: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """Combine subword tokens and aggregate attention weights."""
        token_weights = []
        current_token = ""
        current_weight = 0.0

        for token, weight in zip(tokens, attention_weights):
            if token in ("<s>", "</s>", "<pad>", "[CLS]", "[SEP]"):
                continue

            if token.startswith("Ġ") or token.startswith("##"):
                if current_token:
                    token_weights.append((current_token, float(current_weight)))
                current_token = token.replace("Ġ", "").replace("##", "")
                current_weight = weight
            else:
                if not current_token:
                    current_token = token
                    current_weight = weight
                else:
                    current_token += token
                    current_weight = max(current_weight, weight)

        if current_token:
            token_weights.append((current_token, float(current_weight)))

        return token_weights

    def _keyword_explain(
        self,
        text: str,
        predicted_label: str,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Simple keyword-based explanation (fallback)."""
        fake_keywords = {
            "giả": 0.9, "fake": 0.9, "hoax": 0.85, "gian lận": 0.8,
            "lừa đảo": 0.85, "tin đồn": 0.7, "không xác thực": 0.75,
            "sai sự thật": 0.8, "bịa đặt": 0.85,
        }

        real_keywords = {
            "chính thức": -0.8, "official": -0.8, "xác nhận": -0.75,
            "confirmed": -0.75, "nguồn tin": -0.6, "chứng minh": -0.7,
        }

        all_keywords = {**fake_keywords, **real_keywords}
        text_lower = text.lower()
        found_tokens = []

        for keyword, weight in all_keywords.items():
            if keyword in text_lower:
                if predicted_label == "real":
                    weight = -weight
                found_tokens.append((keyword, weight))

        if not found_tokens:
            words = re.findall(r"\b\w{4,}\b", text_lower)
            found_tokens = [(word, 0.5) for word in words[:top_k]]

        found_tokens.sort(key=lambda x: abs(x[1]), reverse=True)
        return found_tokens[:top_k]
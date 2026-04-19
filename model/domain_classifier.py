"""domain_classifier.py — Domain Classification Service
=======================================================
Handles domain classification (politics/health/finance/social).
Uses trained classifier if available, falls back to keyword-based.

Usage:
    classifier = DomainClassifier(trained_model_path="models/domain_router.pkl")
    domain = classifier.classify("text about politics")
"""

from pathlib import Path
from typing import Optional, Literal

import joblib

from config import DOMAINS, DOMAIN_KEYWORDS


class DomainClassifier:
    """Domain classification with keyword fallback.

    Responsibility: Domain classification only (SRP fix).
    """

    def __init__(
        self,
        trained_model_path: Optional[str | Path] = None,
    ):
        """Initialize domain classifier.

        Args:
            trained_model_path: Path to trained domain classifier.
                                None = keyword-based only.
        """
        self._trained_model = None
        self._use_trained = False

        if trained_model_path is not None:
            model_path = Path(trained_model_path)
            if model_path.exists():
                try:
                    self._trained_model = joblib.load(model_path)
                    self._use_trained = True
                except Exception:
                    pass

    def classify(self, text: str) -> Literal["politics", "health", "finance", "social"]:
        """Classify domain for input text.

        Args:
            text: Input text to classify.

        Returns:
            Predicted domain from DOMAINS.
        """
        if self._use_trained:
            return self._classify_trained(text)
        return self._classify_keywords(text)

    def _classify_trained(self, text: str) -> Literal["politics", "health", "finance", "social"]:
        """Use trained classifier for domain prediction."""
        try:
            domain = self._trained_model.predict([text])[0]
            if domain in DOMAINS:
                return domain
        except Exception:
            pass
        return self._classify_keywords(text)

    def _classify_keywords(self, text: str) -> Literal["politics", "health", "finance", "social"]:
        """Keyword-based domain classification (fallback)."""
        text_lower = text.lower()

        domain_scores = {domain: 0 for domain in DOMAINS}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            domain_scores[domain] = sum(1 for kw in keywords if kw in text_lower)

        max_score = max(domain_scores.values())
        if max_score == 0:
            return "social"

        for domain, score in domain_scores.items():
            if score == max_score:
                return domain

        return "social"

    def is_using_trained_model(self) -> bool:
        """Check if using trained model vs keyword fallback."""
        return self._use_trained
"""mock_predictor.py — Mock Prediction for Testing Without Models
================================================================
Provides realistic mock predictions when trained models are unavailable.

Usage:
    mock = MockPredictor()
    response = mock.predict(PredictRequest(text="...", lang="vi"))
"""

from api.schemas import PredictRequest, PredictResponse


class MockPredictor:
    """Mock prediction for testing without trained models.

    Responsibility: Mock prediction only (SRP fix).
    """

    _FAKE_KEYWORDS = ["giả", "fake", "hoax", "gian lận", "lừa đảo"]
    _REAL_KEYWORDS = ["chính thức", "official", "confirmed", "xác nhận"]

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Generate mock prediction for input.

        Args:
            request: PredictRequest with text and lang.

        Returns:
            Mock PredictResponse with realistic values.
        """
        text_lower = request.text.lower()

        fake_score = sum(1 for kw in self._FAKE_KEYWORDS if kw in text_lower)
        real_score = sum(1 for kw in self._REAL_KEYWORDS if kw in text_lower)

        if fake_score > real_score:
            label = "fake"
            confidence = 0.65 + min(fake_score * 0.1, 0.25)
        elif real_score > fake_score:
            label = "real"
            confidence = 0.65 + min(real_score * 0.1, 0.25)
        else:
            label = "fake"
            confidence = 0.55

        mock_tokens = [
            (word, 0.8 - i * 0.1)
            for i, word in enumerate(text_lower.split()[:5])
            if len(word) > 3
        ]

        return PredictResponse(
            label=label,
            confidence=confidence,
            domain="social",
            shap_tokens=mock_tokens[:3] if mock_tokens else [("mock", 0.5)],
            source_score=0.5,
        )

    def is_mock_mode(self) -> bool:
        """Always returns True for MockPredictor."""
        return True
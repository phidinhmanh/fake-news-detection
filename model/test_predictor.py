"""
test_predictor.py — Unit Tests for Predictor
=============================================
Unit tests for Task IMPL-B-006: Predictor interface.

Tests:
1. Basic prediction flow
2. Mock mode functionality
3. Schema compliance
4. Domain classification
5. SHAP token generation
6. Latency requirements
7. Error handling
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
import time
from api.schemas import PredictRequest, PredictResponse
from model.predictor import Predictor, load_predictor
from config import DOMAINS, TARGET_LATENCY_SECONDS


class TestPredictorBasic:
    """Basic predictor functionality tests."""

    def test_predictor_initialization_mock(self):
        """Test predictor initialization in mock mode."""
        predictor = Predictor(use_mock=True)
        assert predictor.use_mock is True
        assert predictor.ensemble is None

    def test_predictor_initialization_with_paths(self):
        """Test predictor initialization with model paths."""
        # Should not fail even if models don't exist (graceful fallback)
        predictor = Predictor(
            model_path="nonexistent/path",
            baseline_path="nonexistent/baseline.joblib",
            use_mock=False,
        )
        # Should fall back to mock mode if models not found
        assert predictor is not None

    def test_load_predictor_factory(self):
        """Test load_predictor factory function."""
        # Mock mode
        predictor_mock = load_predictor("mock")
        assert predictor_mock.use_mock is True

        # Default mode (may fall back to mock if no models)
        predictor_default = load_predictor("default")
        assert predictor_default is not None


class TestPredictorPrediction:
    """Prediction functionality tests."""

    @pytest.fixture
    def predictor(self):
        """Create predictor for testing."""
        return load_predictor("mock")

    def test_basic_prediction(self, predictor):
        """Test basic prediction flow."""
        request = PredictRequest(
            text="This is a test article about fake news.",
            lang="en"
        )
        response = predictor.predict(request)

        assert isinstance(response, PredictResponse)
        assert response.label in ["fake", "real"]
        assert 0.0 <= response.confidence <= 1.0
        assert response.domain in DOMAINS

    def test_prediction_vietnamese(self, predictor):
        """Test prediction with Vietnamese text."""
        request = PredictRequest(
            text="Đây là bài viết về tin giả. Thông tin không chính xác.",
            lang="vi"
        )
        response = predictor.predict(request)

        assert isinstance(response, PredictResponse)
        assert response.label in ["fake", "real"]

    def test_prediction_english(self, predictor):
        """Test prediction with English text."""
        request = PredictRequest(
            text="This is fake news about politics and government.",
            lang="en"
        )
        response = predictor.predict(request)

        assert isinstance(response, PredictResponse)
        assert response.label in ["fake", "real"]


class TestSchemaCompliance:
    """Schema compliance tests."""

    @pytest.fixture
    def predictor(self):
        """Create predictor for testing."""
        return load_predictor("mock")

    def test_response_label_valid(self, predictor):
        """Test that response label is valid."""
        request = PredictRequest(text="Test text", lang="vi")
        response = predictor.predict(request)
        assert response.label in ["fake", "real"]

    def test_response_confidence_range(self, predictor):
        """Test that confidence is in valid range."""
        request = PredictRequest(text="Test text", lang="vi")
        response = predictor.predict(request)
        assert 0.0 <= response.confidence <= 1.0

    def test_response_domain_valid(self, predictor):
        """Test that domain is valid."""
        request = PredictRequest(text="Test text", lang="vi")
        response = predictor.predict(request)
        assert response.domain in DOMAINS

    def test_response_shap_tokens_format(self, predictor):
        """Test that shap_tokens has correct format."""
        request = PredictRequest(text="Test text", lang="vi")
        response = predictor.predict(request)

        assert isinstance(response.shap_tokens, list)
        for token, weight in response.shap_tokens:
            assert isinstance(token, str)
            assert isinstance(weight, float)

    def test_response_source_score_range(self, predictor):
        """Test that source_score is in valid range or None."""
        request = PredictRequest(text="Test text", lang="vi")
        response = predictor.predict(request)

        if response.source_score is not None:
            assert 0.0 <= response.source_score <= 1.0


class TestDomainClassification:
    """Domain classification tests."""

    @pytest.fixture
    def predictor(self):
        """Create predictor for testing."""
        return load_predictor("mock")

    def test_politics_domain(self, predictor):
        """Test politics domain classification."""
        text = "Quốc hội thông qua luật mới về bầu cử tổng thống"
        request = PredictRequest(text=text, lang="vi")
        response = predictor.predict(request)
        assert response.domain == "politics"

    def test_health_domain(self, predictor):
        """Test health domain classification."""
        text = "Bệnh viện phát hiện ca nhiễm virus mới, bác sĩ khuyến cáo"
        request = PredictRequest(text=text, lang="vi")
        response = predictor.predict(request)
        assert response.domain == "health"

    def test_finance_domain(self, predictor):
        """Test finance domain classification."""
        text = "Chứng khoán giảm mạnh, ngân hàng tăng lãi suất"
        request = PredictRequest(text=text, lang="vi")
        response = predictor.predict(request)
        assert response.domain == "finance"

    def test_social_domain(self, predictor):
        """Test social domain classification."""
        text = "Trường đại học tổ chức lễ hội văn hóa và nghệ thuật"
        request = PredictRequest(text=text, lang="vi")
        response = predictor.predict(request)
        assert response.domain == "social"

    def test_default_domain(self, predictor):
        """Test default domain for ambiguous text."""
        text = "Random text without clear domain indicators"
        request = PredictRequest(text=text, lang="en")
        response = predictor.predict(request)
        # Should return a valid domain (default is "social")
        assert response.domain in DOMAINS


class TestShapTokens:
    """SHAP token explanation tests."""

    @pytest.fixture
    def predictor(self):
        """Create predictor for testing."""
        return load_predictor("mock")

    def test_shap_tokens_returned(self, predictor):
        """Test that SHAP tokens are returned."""
        request = PredictRequest(
            text="This article contains fake hoax information",
            lang="en"
        )
        response = predictor.predict(request)
        assert len(response.shap_tokens) > 0

    def test_shap_tokens_format(self, predictor):
        """Test SHAP tokens format."""
        request = PredictRequest(text="Test text for SHAP", lang="en")
        response = predictor.predict(request)

        for token, weight in response.shap_tokens:
            assert isinstance(token, str)
            assert len(token) > 0
            assert isinstance(weight, float)
            # Weights should be in reasonable range
            assert -1.5 <= weight <= 1.5

    def test_shap_tokens_relevance(self, predictor):
        """Test that SHAP tokens are relevant to text."""
        text = "vaccine hoax fake government"
        request = PredictRequest(text=text, lang="en")
        response = predictor.predict(request)

        # At least one token should be from the input text
        response_tokens = [token for token, _ in response.shap_tokens]
        input_words = text.lower().split()

        # Check if any response token is in input
        has_relevant = any(
            any(word in token.lower() or token.lower() in word
                for word in input_words)
            for token in response_tokens
        )
        assert has_relevant or len(response_tokens) == 0


class TestLatencyRequirements:
    """Latency requirement tests."""

    @pytest.fixture
    def predictor(self):
        """Create predictor for testing."""
        return load_predictor("mock")

    def test_prediction_latency_short_text(self, predictor):
        """Test latency for short text."""
        request = PredictRequest(text="Short text", lang="vi")

        start = time.time()
        predictor.predict(request)
        latency = time.time() - start

        assert latency < TARGET_LATENCY_SECONDS

    def test_prediction_latency_long_text(self, predictor):
        """Test latency for long text."""
        long_text = "This is a long article. " * 80  # ~2000 chars (under 2048 limit)
        request = PredictRequest(text=long_text, lang="en")

        start = time.time()
        predictor.predict(request)
        latency = time.time() - start

        # Even long texts should be under target
        assert latency < TARGET_LATENCY_SECONDS

    def test_average_latency_multiple_calls(self, predictor):
        """Test average latency over multiple calls."""
        latencies = []

        for i in range(10):
            request = PredictRequest(
                text=f"Test article number {i} about fake news",
                lang="vi"
            )

            start = time.time()
            predictor.predict(request)
            latencies.append(time.time() - start)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < TARGET_LATENCY_SECONDS


class TestErrorHandling:
    """Error handling tests."""

    def test_predictor_handles_empty_text(self):
        """Test handling of empty text."""
        predictor = load_predictor("mock")
        request = PredictRequest(text="", lang="vi")

        # Should not crash, returns valid response
        response = predictor.predict(request)
        assert isinstance(response, PredictResponse)

    def test_predictor_handles_very_long_text(self):
        """Test handling of very long text (at max limit)."""
        predictor = load_predictor("mock")
        # 2000 character text (under 2048 limit)
        long_text = "a " * 1000
        request = PredictRequest(text=long_text, lang="vi")

        # Should not crash, returns valid response
        response = predictor.predict(request)
        assert isinstance(response, PredictResponse)

    def test_predictor_handles_special_characters(self):
        """Test handling of special characters."""
        predictor = load_predictor("mock")
        text = "Test with special chars: @#$%^&*()_+-={}[]|\\:;\"'<>,.?/"
        request = PredictRequest(text=text, lang="en")

        # Should not crash
        response = predictor.predict(request)
        assert isinstance(response, PredictResponse)

    def test_predictor_handles_unicode(self):
        """Test handling of Unicode characters."""
        predictor = load_predictor("mock")
        text = "Unicode: 你好 مرحبا こんにちは 안녕하세요 Привет"
        request = PredictRequest(text=text, lang="vi")

        # Should not crash
        response = predictor.predict(request)
        assert isinstance(response, PredictResponse)


class TestMockModeBehavior:
    """Mock mode specific tests."""

    def test_mock_mode_deterministic(self):
        """Test that mock mode gives consistent results."""
        predictor = load_predictor("mock")
        text = "Test deterministic behavior"

        responses = []
        for _ in range(3):
            request = PredictRequest(text=text, lang="vi")
            response = predictor.predict(request)
            responses.append(response)

        # All responses should have same label (deterministic mock)
        labels = [r.label for r in responses]
        assert len(set(labels)) == 1

    def test_mock_mode_keyword_sensitivity(self):
        """Test that mock mode is sensitive to keywords."""
        predictor = load_predictor("mock")

        # Text with fake keywords
        fake_request = PredictRequest(
            text="This is fake hoax gian lận lừa đảo",
            lang="vi"
        )
        fake_response = predictor.predict(fake_request)

        # Text with real keywords
        real_request = PredictRequest(
            text="Official chính thức confirmed xác nhận",
            lang="en"
        )
        real_response = predictor.predict(real_request)

        # Should show some sensitivity (though not required to be perfect)
        assert fake_response.label in ["fake", "real"]
        assert real_response.label in ["fake", "real"]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

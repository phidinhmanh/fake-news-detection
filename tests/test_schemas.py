"""
test_schemas.py — Validate API Contract
=========================================
Test schemas.py để đảm bảo contract không bị break.
Chạy: pytest tests/test_schemas.py -v
"""

import pytest
from api.schemas import PredictRequest, PredictResponse


class TestPredictRequest:
    """Test PredictRequest validation."""

    def test_valid_request_vi(self):
        req = PredictRequest(text="Tin tức về vaccine COVID-19", lang="vi")
        assert req.text == "Tin tức về vaccine COVID-19"
        assert req.lang == "vi"

    def test_valid_request_en(self):
        req = PredictRequest(text="News about vaccine", lang="en")
        assert req.lang == "en"

    def test_default_lang_is_vi(self):
        req = PredictRequest(text="Nội dung bài viết")
        assert req.lang == "vi"

    def test_text_max_length(self):
        """Text không được vượt quá 2048 ký tự."""
        long_text = "a" * 2049
        with pytest.raises(Exception):
            PredictRequest(text=long_text)

    def test_text_at_max_length(self):
        """Text đúng 2048 ký tự phải hợp lệ."""
        text = "a" * 2048
        req = PredictRequest(text=text)
        assert len(req.text) == 2048

    def test_invalid_lang(self):
        """Lang chỉ chấp nhận 'vi' hoặc 'en'."""
        with pytest.raises(Exception):
            PredictRequest(text="test", lang="fr")

    def test_empty_text_rejected(self):
        """Text rỗng vẫn hợp lệ (không có min_length)."""
        req = PredictRequest(text="")
        assert req.text == ""


class TestPredictResponse:
    """Test PredictResponse validation."""

    def test_valid_response_fake(self):
        resp = PredictResponse(
            label="fake",
            confidence=0.87,
            domain="politics",
            shap_tokens=[("vaccine", 0.9), ("hoax", 0.75)],
            source_score=0.3,
        )
        assert resp.label == "fake"
        assert resp.confidence == 0.87
        assert resp.domain == "politics"
        assert len(resp.shap_tokens) == 2
        assert resp.source_score == 0.3

    def test_valid_response_real(self):
        resp = PredictResponse(
            label="real",
            confidence=0.95,
            domain="health",
            shap_tokens=[],
            source_score=None,
        )
        assert resp.label == "real"
        assert resp.source_score is None

    def test_all_domains(self):
        """Tất cả 4 domain phải hợp lệ."""
        for domain in ("politics", "health", "finance", "social"):
            resp = PredictResponse(
                label="fake",
                confidence=0.5,
                domain=domain,
            )
            assert resp.domain == domain

    def test_invalid_label(self):
        with pytest.raises(Exception):
            PredictResponse(label="unknown", confidence=0.5, domain="politics")

    def test_confidence_bounds(self):
        """Confidence phải trong [0.0, 1.0]."""
        with pytest.raises(Exception):
            PredictResponse(label="fake", confidence=1.5, domain="politics")
        with pytest.raises(Exception):
            PredictResponse(label="fake", confidence=-0.1, domain="politics")

    def test_source_score_optional(self):
        """source_score mặc định là None."""
        resp = PredictResponse(label="fake", confidence=0.5, domain="politics")
        assert resp.source_score is None

    def test_shap_tokens_default_empty(self):
        """shap_tokens mặc định là list rỗng."""
        resp = PredictResponse(label="fake", confidence=0.5, domain="politics")
        assert resp.shap_tokens == []

    def test_response_serialization(self):
        """Response phải serialize được thành JSON."""
        resp = PredictResponse(
            label="fake",
            confidence=0.87,
            domain="politics",
            shap_tokens=[("vaccine", 0.9)],
            source_score=0.3,
        )
        data = resp.model_dump()
        assert isinstance(data, dict)
        assert data["label"] == "fake"
        assert isinstance(data["shap_tokens"], list)

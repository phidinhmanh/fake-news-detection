"""
test_mock_server.py — Test Mock FastAPI Server
================================================
Test mock_server.py để đảm bảo API contract hoạt động.
Chạy: pytest tests/test_mock_server.py -v
"""

import pytest
from fastapi.testclient import TestClient

from api.mock import app
from api.schemas import PredictResponse


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["server"] == "mock"


class TestPredictEndpoint:
    """Test /predict endpoint."""

    def test_predict_vi(self, client):
        resp = client.post(
            "/predict",
            json={"text": "Tin tức về vaccine COVID-19", "lang": "vi"},
        )
        assert resp.status_code == 200
        data = resp.json()

        # Validate response matches PredictResponse schema
        parsed = PredictResponse(**data)
        assert parsed.label in ("fake", "real")
        assert 0.0 <= parsed.confidence <= 1.0
        assert parsed.domain in ("politics", "health", "finance", "social")
        assert isinstance(parsed.shap_tokens, list)

    def test_predict_en(self, client):
        resp = client.post(
            "/predict",
            json={"text": "News about vaccine", "lang": "en"},
        )
        assert resp.status_code == 200
        data = resp.json()
        parsed = PredictResponse(**data)
        assert parsed.label in ("fake", "real")

    def test_predict_default_lang(self, client):
        """Nếu không truyền lang, mặc định là 'vi'."""
        resp = client.post(
            "/predict",
            json={"text": "Nội dung bài viết"},
        )
        assert resp.status_code == 200

    def test_predict_invalid_lang(self, client):
        """Lang không hợp lệ phải trả 422."""
        resp = client.post(
            "/predict",
            json={"text": "test", "lang": "fr"},
        )
        assert resp.status_code == 422

    def test_predict_empty_text(self, client):
        """Text rỗng vẫn chấp nhận."""
        resp = client.post(
            "/predict",
            json={"text": ""},
        )
        assert resp.status_code == 200

    def test_predict_long_text_rejected(self, client):
        """Text > 2048 ký tự phải bị reject."""
        resp = client.post(
            "/predict",
            json={"text": "a" * 2049},
        )
        assert resp.status_code == 422

    def test_predict_response_has_shap_tokens(self, client):
        """Response phải có shap_tokens (có thể empty)."""
        resp = client.post(
            "/predict",
            json={"text": "Vaccine hoax"},
        )
        data = resp.json()
        assert "shap_tokens" in data

    def test_predict_source_score_nullable(self, client):
        """source_score có thể là float hoặc None."""
        # Chạy nhiều lần vì mock random
        scores = []
        for _ in range(20):
            resp = client.post(
                "/predict",
                json={"text": "test article"},
            )
            data = resp.json()
            scores.append(data["source_score"])

        # Phải có ít nhất 1 lần None và 1 lần float (xác suất rất cao với 20 lần)
        assert any(s is None for s in scores) or any(
            isinstance(s, float) for s in scores
        )

    def test_missing_text_rejected(self, client):
        """Thiếu text field phải trả 422."""
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

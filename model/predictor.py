"""
predictor.py — Prediction Interface (khớp schemas.py)
========================================================
Người B phát triển. Người C gọi qua API.

Interface contract:
    predictor = Predictor()
    response: PredictResponse = predictor.predict(request: PredictRequest)

TODO (Tuần 1-2):
    - [ ] predictor.py v1 — wrapper đơn giản, dùng baseline model
TODO (Tuần 5-6):
    - [ ] predictor.py v2 — ensemble + SHAP integration
"""

from __future__ import annotations

from schemas import PredictRequest, PredictResponse


class Predictor:
    """Prediction interface — entry point cho API server.

    Người C (UI) và mock_server.py đều gọi class này.
    Interface KHÔNG THAY ĐỔI sau tuần 2.

    Usage:
        predictor = Predictor(model_path="models/xlmr_lora/")
        response = predictor.predict(PredictRequest(text="...", lang="vi"))
    """

    def __init__(self, model_path: str | None = None):
        """Load model từ disk.

        Args:
            model_path: Path tới model directory. None = dùng mock.
        """
        self.model_path = model_path
        self._model = None
        # TODO: Load model khi không phải mock
        # if model_path:
        #     self._load_model(model_path)

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Predict fake/real cho input text.

        Args:
            request: PredictRequest với text và lang.

        Returns:
            PredictResponse với label, confidence, domain, shap_tokens, source_score.
        """
        # TODO: Implement thật — hiện tại trả mock
        return PredictResponse(
            label="fake",
            confidence=0.87,
            domain="politics",
            shap_tokens=[("vaccine", 0.9), ("hoax", 0.75)],
            source_score=0.3,
        )

    def _load_model(self, model_path: str) -> None:
        """Load trained model từ checkpoint."""
        # TODO: Người B implement
        raise NotImplementedError("Người B implement — tuần 5-6")

    def _predict_ensemble(self, text: str, lang: str) -> dict:
        """Run ensemble prediction (LLM + rule-based)."""
        # TODO: Người B implement
        raise NotImplementedError("Người B implement — tuần 5-6")

    def _get_shap_tokens(self, text: str) -> list[tuple[str, float]]:
        """Generate SHAP token explanations."""
        # TODO: Người B implement
        raise NotImplementedError("Người B implement — tuần 5-6")

    def _classify_domain(self, text: str) -> str:
        """Classify domain (politics/health/finance/social)."""
        # TODO: Liên kết với domain_router.pkl từ Người A
        raise NotImplementedError("Người A + B implement — tuần 5-6")

"""interfaces.py — Prediction Abstraction (DIP: High-level depends on abstraction)
================================================================================
Defines the contract that all predictor implementations must follow.
This allows Predictor to depend on abstraction, not concrete classes.
"""

from abc import ABC, abstractmethod
from typing import Optional, Literal, List, Tuple

from api.schemas import PredictRequest, PredictResponse


class PredictionInterface(ABC):
    """Abstract interface for prediction services.

    All concrete implementations (EnsemblePredictor, MockPredictor, etc.)
    must conform to this contract.

    Usage:
        predictor: PredictionInterface = EnsemblePredictor()
        response = predictor.predict(request)
    """

    @abstractmethod
    def predict(self, request: PredictRequest) -> PredictResponse:
        """Run prediction on input text.

        Args:
            request: PredictRequest with text and lang.

        Returns:
            PredictResponse with label, confidence, domain, shap_tokens, source_score.
        """
        ...

    @abstractmethod
    def is_mock_mode(self) -> bool:
        """Check if running in mock mode (no trained models).

        Returns:
            True if mock mode, False otherwise.
        """
        ...

    def predict_batch(
        self, requests: List[PredictRequest]
    ) -> List[PredictResponse]:
        """Batch prediction for multiple texts.

        Default implementation calls predict() for each item.
        Override for optimized batch processing.

        Args:
            requests: List of PredictRequest.

        Returns:
            List of PredictResponse.
        """
        return [self.predict(req) for req in requests]
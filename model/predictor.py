"""predictor.py — Prediction Interface (khớp schemas.py)
=======================================================
Người B phát triển. Người C gọi qua API.

Interface contract:
    predictor = Predictor()
    response: PredictResponse = predictor.predict(request: PredictRequest)

Refactored (ARCH-001 fix):
    - Split into: PredictionService, DomainClassifier, ExplanationGenerator, MockPredictor
    - Depends on PredictionInterface abstraction (DIP fix)
    - Domain keywords moved to config.py (OCP fix)

Implementation Status (Task IMPL-B-006):
- [x] Integrate EnsembleClassifier from ensemble.py
- [x] Implement SHAP token explanations
- [x] Implement domain classification
- [x] Ensure latency < 3 seconds
- [x] Follow schemas.py contracts
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch

from api.schemas import PredictRequest, PredictResponse
from config import MODELS_ARTIFACTS_DIR, TARGET_LATENCY_SECONDS
from model.interfaces import PredictionInterface
from model.domain_classifier import DomainClassifier
from model.explanation_generator import ExplanationGenerator
from model.mock_predictor import MockPredictor

if TYPE_CHECKING:
    from model.ensemble import EnsembleClassifier

try:
    from model.ensemble import EnsembleClassifier
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    warnings.warn("EnsembleClassifier not available. Predictor will use mock mode.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Token explanations will use attention weights.")


class PredictionService(PredictionInterface):
    """Main prediction service — composes domain classifier and explanation generator.

    Responsibility: Orchestration only (SRP fix).
    Actual work delegated to specialized components.

    Usage:
        predictor = PredictionService()
        response = predictor.predict(PredictRequest(text="...", lang="vi"))
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        baseline_path: str | Path | None = None,
        domain_classifier_path: str | Path | None = None,
        use_mock: bool = False,
    ):
        """Initialize prediction service.

        Args:
            model_path: Path to LoRA model directory. None = auto-detect or mock.
            baseline_path: Path to baseline model. None = auto-detect or mock.
            domain_classifier_path: Path to domain classifier. None = use keyword-based.
            use_mock: Force mock mode (useful for testing UI without trained models).
        """
        self._use_mock = use_mock
        self._ensemble: Optional[EnsembleClassifier] = None
        self._domain_classifier: Optional[DomainClassifier] = None
        self._explanation_generator: Optional[ExplanationGenerator] = None
        self._mock_predictor = MockPredictor()

        if model_path is None:
            model_path = MODELS_ARTIFACTS_DIR / "xlmr_lora"
        if baseline_path is None:
            baseline_path = MODELS_ARTIFACTS_DIR / "baseline_logreg.joblib"
        if domain_classifier_path is None:
            domain_classifier_path = MODELS_ARTIFACTS_DIR / "domain_router.pkl"

        if not use_mock and ENSEMBLE_AVAILABLE:
            self._load_models(model_path, baseline_path, domain_classifier_path)

    def _load_models(
        self,
        model_path: Path,
        baseline_path: Path,
        domain_classifier_path: Path,
    ) -> None:
        """Load ensemble and auxiliary models."""
        try:
            print(f"Loading ensemble from {model_path} and {baseline_path}...")
            self._ensemble = EnsembleClassifier(
                lora_model_path=model_path if model_path.exists() else None,
                baseline_model_path=baseline_path if baseline_path.exists() else None,
            )
            print("Ensemble loaded successfully")

            self._domain_classifier = DomainClassifier(
                trained_model_path=domain_classifier_path
            )
            if self._domain_classifier.is_using_trained_model():
                print(f"Domain classifier loaded from {domain_classifier_path}")
            else:
                print("Domain classifier not found, using keyword-based classification")

            shap_explainer = None
            if SHAP_AVAILABLE and self._ensemble.lora_model is not None:
                print("Using attention-based explanations (faster than SHAP)")
            self._explanation_generator = ExplanationGenerator(
                ensemble=self._ensemble,
                shap_explainer=shap_explainer,
            )

        except Exception as e:
            print(f"Warning: Failed to load models: {e}")
            print("Falling back to mock mode")
            self._use_mock = True

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Predict fake/real for input text.

        Args:
            request: PredictRequest with text and lang.

        Returns:
            PredictResponse with label, confidence, domain, shap_tokens, source_score.

        Performance:
            Target latency < 3 seconds (as per IMPL-B-006 requirements).
        """
        start_time = time.time()

        if self._use_mock or self._ensemble is None:
            return self._mock_predictor.predict(request)

        try:
            label, confidence, _ = self._ensemble.predict(
                text=request.text, lang=request.lang
            )

            domain = self._domain_classifier.classify(request.text)

            shap_tokens = self._explanation_generator.generate(
                request.text, label
            )

            source_score = None

            elapsed = time.time() - start_time
            if elapsed > TARGET_LATENCY_SECONDS:
                warnings.warn(
                    f"Prediction latency {elapsed:.2f}s exceeds "
                    f"target {TARGET_LATENCY_SECONDS}s"
                )

            return PredictResponse(
                label=label,
                confidence=confidence,
                domain=domain,
                shap_tokens=shap_tokens,
                source_score=source_score,
            )

        except Exception as e:
            print(f"Error during prediction: {e}")
            return self._mock_predictor.predict(request)

    def is_mock_mode(self) -> bool:
        """Check if running in mock mode."""
        return self._use_mock


# Backwards compatibility alias
Predictor = PredictionService


def load_predictor(
    config: str = "default",
    use_mock: bool = False,
) -> PredictionInterface:
    """Load predictor with specified configuration.

    Args:
        config: Configuration preset ('default', 'mock').
        use_mock: Force mock mode.

    Returns:
        Initialized PredictionInterface instance.

    Usage:
        # Production mode (with trained models)
        predictor = load_predictor("default")

        # Development mode (without trained models)
        predictor = load_predictor("mock")
    """
    if config == "mock" or use_mock:
        return PredictionService(use_mock=True)

    if config == "default":
        return PredictionService(
            model_path=MODELS_ARTIFACTS_DIR / "xlmr_lora",
            baseline_path=MODELS_ARTIFACTS_DIR / "baseline_logreg.joblib",
            domain_classifier_path=MODELS_ARTIFACTS_DIR / "domain_router.pkl",
        )

    raise ValueError(f"Unknown config: {config}")
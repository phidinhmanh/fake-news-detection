"""
server.py — Production FastAPI Server
======================================
Production API server using the Predictor interface.

Usage:
    # Development mode (mock predictor)
    uvicorn server:app --port 8000 --reload

    # Production mode (with trained models)
    PREDICTOR_MODE=production uvicorn server:app --port 8000

Test:
    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"text": "Tin giả về vaccine", "lang": "vi"}'
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import PredictRequest, PredictResponse
from model.predictor import load_predictor

# Determine mode from environment
PREDICTOR_MODE = os.getenv("PREDICTOR_MODE", "mock")  # "mock" or "production"

# Initialize FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="Production API for fake news detection using ensemble models.",
    version="1.0.0",
)

# CORS middleware for Streamlit UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance (initialized once at startup)
predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize predictor on server startup."""
    global predictor

    print(f"Initializing predictor in {PREDICTOR_MODE} mode...")

    if PREDICTOR_MODE == "mock":
        predictor = load_predictor("mock")
        print("✓ Predictor initialized in MOCK mode (no trained models)")
    else:
        predictor = load_predictor("default")
        print("✓ Predictor initialized in PRODUCTION mode")


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest) -> PredictResponse:
    """Main prediction endpoint.

    Args:
        request: PredictRequest with text and language.

    Returns:
        PredictResponse with label, confidence, domain, shap_tokens, source_score.

    Raises:
        HTTPException: If prediction fails.
    """
    if predictor is None:
        raise HTTPException(
            status_code=500,
            detail="Predictor not initialized. Server may be starting up.",
        )

    try:
        response = predictor.predict(request)
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.get("/health")
def health_check():
    """Health check endpoint.

    Returns:
        Server status and configuration info.
    """
    return {
        "status": "ok",
        "server": "production",
        "version": "1.0.0",
        "mode": PREDICTOR_MODE,
        "predictor_loaded": predictor is not None,
    }


@app.get("/info")
def info():
    """Server information endpoint.

    Returns:
        Detailed server configuration.
    """
    return {
        "name": "Fake News Detection API",
        "version": "1.0.0",
        "mode": PREDICTOR_MODE,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "info": "/info",
        },
        "supported_languages": ["vi", "en"],
        "domains": ["politics", "health", "finance", "social"],
        "features": {
            "ensemble_prediction": True,
            "shap_explanations": True,
            "domain_classification": True,
            "source_scoring": False,  # Future feature
        },
    }


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )

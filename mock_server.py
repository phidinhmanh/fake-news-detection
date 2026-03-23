"""
mock_server.py — Mock FastAPI Server
=====================================
Người A setup, Người C dùng ngay từ ngày 1.

Chạy:
    uvicorn mock_server:app --port 8000 --reload

Test:
    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"text": "Tin giả về vaccine", "lang": "vi"}'
"""

import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="Fake News Detection API (Mock)",
    description="Mock server cho phát triển UI. Trả về dữ liệu giả lập.",
    version="0.1.0",
)

# Cho phép Streamlit connect từ localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data pools
MOCK_DOMAINS = ["politics", "health", "finance", "social"]
MOCK_SHAP_TOKENS_VI = [
    ("vaccine", 0.92),
    ("hoax", 0.85),
    ("chính phủ", 0.78),
    ("nguồn tin", -0.65),
    ("theo", -0.12),
    ("cho biết", -0.08),
    ("giả mạo", 0.88),
    ("sự thật", -0.72),
    ("nghiên cứu", -0.55),
    ("tuyên bố", 0.45),
]
MOCK_SHAP_TOKENS_EN = [
    ("vaccine", 0.90),
    ("hoax", 0.75),
    ("government", 0.68),
    ("source", -0.55),
    ("according", -0.10),
    ("fake", 0.82),
    ("truth", -0.70),
    ("study", -0.50),
    ("claims", 0.40),
    ("breaking", 0.35),
]


@app.post("/predict", response_model=PredictResponse)
def mock_predict(req: PredictRequest) -> PredictResponse:
    """Mock prediction endpoint.

    Trả về dữ liệu giả lập với một chút randomness
    để UI test được nhiều trường hợp.
    """
    # Random label với bias nhẹ về 'fake' để test highlight
    confidence = round(random.uniform(0.55, 0.98), 2)
    label = "fake" if confidence > 0.6 else "real"

    # Chọn domain ngẫu nhiên
    domain = random.choice(MOCK_DOMAINS)

    # Chọn SHAP tokens theo ngôn ngữ
    token_pool = MOCK_SHAP_TOKENS_VI if req.lang == "vi" else MOCK_SHAP_TOKENS_EN
    num_tokens = min(len(token_pool), random.randint(3, 7))
    shap_tokens = random.sample(token_pool, num_tokens)

    # Source score: 30% chance là None
    source_score = round(random.uniform(0.1, 0.9), 2) if random.random() > 0.3 else None

    return PredictResponse(
        label=label,
        confidence=confidence,
        domain=domain,
        shap_tokens=shap_tokens,
        source_score=source_score,
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "server": "mock", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

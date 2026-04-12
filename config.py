"""
config.py — Shared Configuration
==================================
Cấu hình chung cho cả 3 người. Import từ đây thay vì hardcode paths.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASETS_DIR = DATA_DIR / "datasets"
NORMALIZED_DIR = DATASETS_DIR / "normalized"
AUGMENTED_DIR = DATASETS_DIR / "augmented"

MODEL_DIR = PROJECT_ROOT / "model"
MODELS_ARTIFACTS_DIR = PROJECT_ROOT / "saved_models"  # Trained model artifacts

UI_DIR = PROJECT_ROOT / "ui"

TESTS_DIR = PROJECT_ROOT / "tests"

# ── API ────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_URL = f"http://localhost:{API_PORT}"
PREDICT_ENDPOINT = f"{API_URL}/predict"

# ── Model ──────────────────────────────────────────────
MAX_TEXT_LENGTH = 2048
SUPPORTED_LANGUAGES = ("vi", "en")
DOMAINS = ("politics", "health", "finance", "social")
LABELS = ("fake", "real")

# ── Training defaults ──────────────────────────────────
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_EPOCHS = 10
DEFAULT_MAX_SEQ_LEN = 256

# ── Target metrics ─────────────────────────────────────
TARGET_BASELINE_F1 = 0.72
TARGET_LORA_F1 = 0.85
TARGET_ENSEMBLE_AUC = 0.90
TARGET_LATENCY_SECONDS = 3.0

# ── Sequential Adversarial Pipeline ────────────────────────────────────────────
SA_DIR = PROJECT_ROOT / "sequential_adversarial"

# Gemini model to use (can be overridden by GOOGLE_GEMINI_MODEL env var)
import os as _os
SA_MODEL_NAME = _os.getenv("GOOGLE_GEMINI_MODEL", "gemini-1.5-flash")

# SQLite database for persistence (Stage 7)
SA_DB_PATH = SA_DIR / "data" / "verity_reports.db"

# Output directory for Mermaid visual flowcharts (Stage 6)
SA_VISUAL_DIR = SA_DIR / "data" / "visuals"

# Max text length fed to the LLM pipeline
SA_MAX_TEXT_CHARS = 8_000

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

# ── New Dataset Pipeline (Vietnamese) ──────────────────
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_RAW_DIR = DATASET_DIR / "raw"
DATASET_PROCESSED_DIR = DATASET_DIR / "processed"
DATASET_STATS_DIR = DATASET_DIR / "statistics"
VIFACTCHECK_DIR = DATASET_RAW_DIR / "vifactcheck"
REINTEL_DIR = DATASET_RAW_DIR / "reintel"
COLLECTED_DIR = DATASET_RAW_DIR / "collected"

MODEL_DIR = PROJECT_ROOT / "model"
MODELS_ARTIFACTS_DIR = PROJECT_ROOT / "saved_models"  # Trained model artifacts

# ── Agents ─────────────────────────────────────────────
AGENTS_DIR = PROJECT_ROOT / "agents"
KB_DIR = DATASET_PROCESSED_DIR / "knowledge_base"

# ── Evaluation ─────────────────────────────────────────
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
EVAL_RESULTS_DIR = EVALUATION_DIR / "results"

# ── Docs ───────────────────────────────────────────────
DOCS_DIR = PROJECT_ROOT / "docs"

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

# ── PhoBERT Configuration ─────────────────────────────
PHOBERT_MODEL_NAME = "vinai/phobert-base"
PHOBERT_MAX_SEQ_LEN = 256
PHOBERT_BATCH_SIZE = 16
PHOBERT_LEARNING_RATE = 1e-5
PHOBERT_EPOCHS = 10
PHOBERT_WARMUP_RATIO = 0.1
PHOBERT_DROPOUT = 0.3
NUM_STYLISTIC_FEATURES = 9   # word_count, sentiment, etc.

# ── Agent Configuration ────────────────────────────────
AGENT_TIMEOUT_SECONDS = 30
AGENT_TOP_K_EVIDENCE = 5
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LANCEDB_TABLE_NAME = "vifactcheck_evidence"

# ── Multi-LLM Provider Configuration ───────────────────────────────────────────
import os as _os
LLM_PROVIDER = _os.getenv("LLM_PROVIDER", "gemini")  # gemini/gemma/qwen/grok/nvidia/openai
OLLAMA_BASE_URL = _os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QWEN_API_KEY = _os.getenv("QWEN_API_KEY", "")
GROK_API_KEY = _os.getenv("GROK_API_KEY", "")
NVIDIA_API_KEY = _os.getenv("NVIDIA_API_KEY", "")
OPENAI_API_KEY = _os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = _os.getenv("OPENAI_BASE_URL", "")

# ── Search API Configuration ────────────────────────────────────────────────────
SEARCH_PROVIDER = _os.getenv("SEARCH_PROVIDER", "wikipedia")  # wikipedia/serper
SERPER_API_KEY = _os.getenv("SERPER_API_KEY", "")

# ── Target metrics ─────────────────────────────────────
TARGET_BASELINE_F1 = 0.72
TARGET_PHOBERT_F1 = 0.80
TARGET_PROPOSED_F1 = 0.85
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

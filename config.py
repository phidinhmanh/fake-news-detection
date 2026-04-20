"""
config.py — Slimmed Shared Configuration
========================================
Central source for all paths and parameters focusing on Vietnamese Fake News detection.
"""
from pathlib import Path
import os as _os

# Load .env at import to ensure API keys are available
from dotenv import load_dotenv
load_dotenv()

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# Vietnamese Dataset Pipeline
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_RAW_DIR = DATASET_DIR / "raw"
DATASET_PROCESSED_DIR = DATASET_DIR / "processed"
VIFACTCHECK_DIR = DATASET_RAW_DIR / "vifactcheck"

# Models & Agents
MODELS_ARTIFACTS_DIR = PROJECT_ROOT / "saved_models"
AGENTS_DIR = PROJECT_ROOT / "agents"
KB_DIR = DATASET_PROCESSED_DIR / "knowledge_base"

# Evaluation
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
EVAL_RESULTS_DIR = EVALUATION_DIR / "results"

# ── Model Configuration ────────────────────────────────
LABELS = ("fake", "real")
MAX_TEXT_LENGTH = 2048

# PhoBERT
PHOBERT_MODEL_NAME = "vinai/phobert-base-v2"
PHOBERT_MAX_SEQ_LEN = 512
PHOBERT_BATCH_SIZE = 16
PHOBERT_EPOCHS = 5
PHOBERT_LEARNING_RATE = 2e-5
NUM_STYLISTIC_FEATURES = 9

# ── Agent & RAG ────────────────────────────────────────
AGENT_TIMEOUT_SECONDS = 30
AGENT_TOP_K_EVIDENCE = 5
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LANCEDB_TABLE_NAME = "vifactcheck_evidence"

# ── LLM Providers ──────────────────────────────────────
LLM_PROVIDER = _os.getenv("LLM_PROVIDER", "nvidia")
SA_MODEL_NAME = _os.getenv("SA_MODEL_NAME", "meta/llama-3.1-70b-instruct")

# API Keys
NVIDIA_API_KEY = _os.getenv("NVIDIA_API_KEY", "")
OPENAI_API_KEY = _os.getenv("OPENAI_API_KEY", "")
QWEN_API_KEY = _os.getenv("QWEN_API_KEY", "")
GROK_API_KEY = _os.getenv("GROK_API_KEY", "")
SERPER_API_KEY = _os.getenv("SERPER_API_KEY", "")

# ── Targets ────────────────────────────────────────────
TARGET_PHOBERT_F1 = 0.80
TARGET_PROPOSED_F1 = 0.85
TARGET_BASELINE_F1 = 0.72

# ── TF-IDF baseline ─────────────────────────────────────
TFIDF_MAX_FEATURES = 5000

# ── Sequential Adversarial Pipeline ─────────────────────
SA_DIR = PROJECT_ROOT / "sequential_adversarial"
SA_VISUAL_DIR = PROJECT_ROOT / "visuals"
SA_DB_PATH = SA_DIR / "data" / "verity_reports.db"
SA_MAX_TEXT_CHARS = 8_000
SA_CHECKPOINT_DIR = SA_DIR / "checkpoints"

# ── Watchdog (NFR-8.8) ─────────────────────────────────
STAGE_TIMEOUT_SECONDS = int(_os.getenv("STAGE_TIMEOUT_SECONDS", "120"))

# ── API Keys & Base URLs ────────────────────────────────
OPENAI_BASE_URL = _os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ── Timeouts (NFR-8.4) ──────────────────────────────────
REQUEST_TIMEOUT = int(_os.getenv("REQUEST_TIMEOUT", "30"))  # seconds
LLM_TIMEOUT = int(_os.getenv("LLM_TIMEOUT", "60"))  # seconds for LLM API calls
DB_TIMEOUT = int(_os.getenv("DB_TIMEOUT", "10"))  # seconds for database operations
SEARCH_TIMEOUT = int(_os.getenv("SEARCH_TIMEOUT", "15"))  # seconds for search APIs

# ── Rate Limiting ────────────────────────────────────────
RATE_LIMIT_REQUESTS_PER_MINUTE = 30
RATE_LIMIT_BURST = 10

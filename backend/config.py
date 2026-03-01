"""
PharmaLLM Backend Configuration
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_BASE   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_WEIGHTS = os.path.join(ROOT_DIR, "training", "fine_tuned_model", "checkpoint-200")
DATA_DIR     = os.path.join(ROOT_DIR, "data")

# ── Inference ──────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS  = 256
TEMPERATURE     = 0.3
TOP_P           = 0.9
REPETITION_PENALTY = 1.2

# ── Server ─────────────────────────────────────────────────────────────────────
DEBUG    = os.getenv("FLASK_DEBUG", "true").lower() == "true"
HOST     = os.getenv("FLASK_HOST", "0.0.0.0")
PORT     = int(os.getenv("FLASK_PORT", 5000))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173")

# ── Feature flags ──────────────────────────────────────────────────────────────
ENABLE_TTS         = True
ENABLE_STT         = True
ENABLE_TRANSLATION = True

# ── TTS ────────────────────────────────────────────────────────────────────────
TTS_LANG = "en"

# ── STT ────────────────────────────────────────────────────────────────────────
WHISPER_MODEL = "base"   # tiny | base | small | medium | large

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_TOKEN = os.getenv("TOKEN_TELEGRAM")

# OpenRouter settings
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
LLM_MODEL = "google/gemini-3-flash-preview"  # Default model

# Available LLM models for users to choose from
AVAILABLE_MODELS = {
    "google/gemini-3-flash-preview": "Gemini 3 Flash ",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5 "
}

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SPEAKERS_DIR = BASE_DIR / "speakers"
SESSIONS_DIR = BASE_DIR / "sessions"
PODCASTS_DIR = Path(__file__).parent.parent / "podcasts"

# RAG settings
CHUNK_MAX_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 150
SEARCH_TOP_K = 5

# Sentiment Analysis Settings
SENTIMENT_MAX_CONCURRENT_REQUESTS = int(os.getenv("SENTIMENT_MAX_CONCURRENT", "10"))
SENTIMENT_REQUEST_TIMEOUT = float(os.getenv("SENTIMENT_TIMEOUT", "90.0"))
SENTIMENT_INCLUDE_FULL_TEXT = os.getenv("SENTIMENT_FULL_TEXT", "true").lower() == "true"
SENTIMENT_TOP_K = int(os.getenv("SENTIMENT_TOP_K", "5000"))  # Max chunks to search for mentions
SENTIMENT_MAX_TEXT_LENGTH = int(os.getenv("SENTIMENT_MAX_TEXT_LENGTH", "30000"))  # Max chars per podcast for LLM

# Logging settings
LOGS_DIR = BASE_DIR / "logs"
LOG_ENABLED = os.getenv("LOG_ENABLED", "true").lower() == "true"
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "90"))

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
if LOG_ENABLED:
    LOGS_DIR.mkdir(exist_ok=True)

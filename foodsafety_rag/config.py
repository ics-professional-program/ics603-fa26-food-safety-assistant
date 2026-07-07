"""Settings and constants. The Gemini key is read ONLY from the environment."""

import os
from dataclasses import dataclass

GEMINI_MODEL = "gemini-flash-latest"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K = 4
SIMILARITY_THRESHOLD = 0.35
DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/foodsafety"


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    database_url: str
    replay: bool


def get_settings() -> Settings:
    """Read settings from the environment on every call (test-friendly)."""
    return Settings(
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
        database_url=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
        replay=os.environ.get("REPLAY", "").strip().lower() in ("1", "true", "yes"),
    )

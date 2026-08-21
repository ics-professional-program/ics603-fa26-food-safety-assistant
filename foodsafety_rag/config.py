"""Settings and constants. The LLM endpoint is read ONLY from the environment."""

import os
from dataclasses import dataclass

# Any OpenAI-compatible server works: the course-provided endpoint, a local
# model server (LM Studio, Ollama, vLLM), or api.openai.com itself.
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_LLM_PROVIDER = "course"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K = 4
SIMILARITY_THRESHOLD = 0.35
DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/foodsafety"


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    database_url: str
    replay: bool


def get_settings() -> Settings:
    """Read current environment settings (the module-level agent snapshots them at import)."""
    return Settings(
        llm_provider=(os.environ.get("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER).strip().lower(),
        llm_api_key=os.environ.get("LLM_API_KEY") or None,
        llm_base_url=os.environ.get("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL,
        llm_model=os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL,
        database_url=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
        replay=os.environ.get("REPLAY", "").strip().lower() in ("1", "true", "yes"),
    )

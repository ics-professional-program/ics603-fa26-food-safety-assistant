from foodsafety_rag import config


def test_constants():
    assert config.DEFAULT_LLM_BASE_URL == "https://api.openai.com/v1"
    assert config.DEFAULT_LLM_MODEL == "gpt-4o-mini"
    assert config.EMBED_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.EMBED_DIM == 384
    assert config.TOP_K == 4
    assert 0.0 < config.SIMILARITY_THRESHOLD < 1.0


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_MODEL", "some-local-model")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@example:5432/db")
    monkeypatch.setenv("REPLAY", "1")
    s = config.get_settings()
    assert s.llm_api_key == "test-key"
    assert s.llm_base_url == "http://localhost:1234/v1"
    assert s.llm_model == "some-local-model"
    assert s.database_url == "postgresql://u:p@example:5432/db"
    assert s.replay is True


def test_settings_defaults(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "DATABASE_URL", "REPLAY"):
        monkeypatch.delenv(name, raising=False)
    s = config.get_settings()
    assert s.llm_api_key is None
    assert s.llm_base_url == config.DEFAULT_LLM_BASE_URL
    assert s.llm_model == config.DEFAULT_LLM_MODEL
    assert s.database_url == config.DEFAULT_DATABASE_URL
    assert s.replay is False

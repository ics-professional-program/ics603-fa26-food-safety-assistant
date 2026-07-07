from foodsafety_rag import config


def test_constants():
    assert config.GEMINI_MODEL == "gemini-flash-latest"
    assert config.EMBED_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.EMBED_DIM == 384
    assert config.TOP_K == 4
    assert 0.0 < config.SIMILARITY_THRESHOLD < 1.0


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@example:5432/db")
    monkeypatch.setenv("REPLAY", "1")
    s = config.get_settings()
    assert s.gemini_api_key == "test-key"
    assert s.database_url == "postgresql://u:p@example:5432/db"
    assert s.replay is True


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REPLAY", raising=False)
    s = config.get_settings()
    assert s.gemini_api_key is None
    assert s.database_url == config.DEFAULT_DATABASE_URL
    assert s.replay is False

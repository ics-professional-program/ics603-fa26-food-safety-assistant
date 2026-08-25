import pytest

from foodsafety_rag.agent import build_model, check_credentials
from foodsafety_rag.config import Settings


def _settings(provider: str, *, key: str | None = None, base_url: str = "http://localhost:1234/v1"):
    return Settings(
        llm_provider=provider,
        llm_api_key=key,
        llm_base_url=base_url,
        llm_model="test-model",
        database_url="postgresql://example",
        replay=False,
    )


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai", "openai-chat:test-model"),
        ("anthropic", "anthropic:test-model"),
        ("google-cloud", "google-cloud:test-model"),
        ("vertex", "google-cloud:test-model"),
    ],
)
def test_hosted_provider_identifiers_are_explicit(provider, expected):
    assert build_model(_settings(provider)) == expected


def test_course_adapter_accepts_a_local_server_without_a_key():
    check_credentials(_settings("course"))
    assert build_model(_settings("course")).model_name == "test-model"


def test_course_adapter_requires_a_key_for_a_hosted_server():
    settings = _settings("course", base_url="https://example.invalid/v1")
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        check_credentials(settings)


def test_openai_and_anthropic_use_their_own_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        check_credentials(_settings("openai"))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        check_credentials(_settings("anthropic"))


def test_unknown_provider_fails_before_the_first_request():
    with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
        check_credentials(_settings("unknown"))


def _passage(doc, heading):
    from foodsafety_rag.schemas import Passage
    return Passage(doc=doc, heading=heading, text="body", score=0.9)


def _citation(doc, heading):
    from foodsafety_rag.schemas import Citation
    return Citation(doc=doc, heading=heading, snippet="s")


def test_citation_matches_tolerates_formatting_variance():
    from foodsafety_rag.agent import citation_matches
    passages = [_passage("FDA Food Code §3-401 — Cooking Temperatures",
                         "Poultry and stuffed foods")]
    # models often reproduce the em dash as a hyphen and drop the section sign
    assert citation_matches(
        _citation("FDA Food Code 3-401 - Cooking Temperatures",
                  "Poultry and stuffed foods"), passages)


def test_citation_matches_tolerates_a_completed_truncated_title():
    from foodsafety_rag.agent import citation_matches
    # generated bulk titles can be cut at a line break; the model cites the
    # full section name
    passages = [_passage(
        "FDA Food Code 2022 3-501.16 Time/Temperature Control for Safety Food, Hot and Cold",
        "3-501.16")]
    assert citation_matches(
        _citation("FDA Food Code 2022 3-501.16 Time/Temperature Control for Safety Food, Hot and Cold Holding",
                  "3-501.16"), passages)


def test_citation_matches_rejects_fabrications():
    from foodsafety_rag.agent import citation_matches
    passages = [_passage("FDA Food Code §3-401 — Cooking Temperatures",
                         "Poultry and stuffed foods")]
    assert not citation_matches(
        _citation("USDA Handbook of Poultry", "Poultry and stuffed foods"),
        passages)
    assert not citation_matches(
        _citation("FDA Food Code §3-401 — Cooking Temperatures", "Ground meats"),
        passages)

import pytest

from foodsafety_rag import generate
from foodsafety_rag.schemas import Answer, Citation, Passage


class StubResponse:
    def __init__(self, text=None, parsed=None):
        self.text = text
        self.parsed = parsed


class StubModels:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.last_kwargs = None

    def generate_content(self, **kwargs):
        self.last_kwargs = kwargs
        if self._error:
            raise self._error
        return self._response


class StubClient:
    def __init__(self, response=None, error=None):
        self.models = StubModels(response, error)


PASSAGES = [
    Passage(doc="FDA Food Code §3-401 — Cooking Temperatures",
            heading="Poultry and stuffed foods",
            text="Raw poultry must reach 165°F (74°C) for 15 seconds.",
            score=0.82),
]


def test_get_client_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        generate.get_client()


def test_ask_model_returns_text():
    client = StubClient(response=StubResponse(text="RAG retrieves passages first."))
    assert generate.ask_model("What is RAG?", client=client) == "RAG retrieves passages first."
    assert client.models.last_kwargs["model"] == "gemini-flash-latest"


def test_answer_question_builds_grounded_answer():
    parsed = generate.ModelAnswer(
        answer="Poultry must reach 165°F (74°C) for 15 seconds.",
        supported=True,
        citations=[Citation(doc="FDA Food Code §3-401 — Cooking Temperatures",
                            heading="Poultry and stuffed foods",
                            snippet="...165°F (74°C)...")],
    )
    client = StubClient(response=StubResponse(parsed=parsed))
    answer = generate.answer_question(
        "What is the minimum internal temperature for poultry?", PASSAGES, client=client)
    assert isinstance(answer, Answer)
    assert answer.grounded is True
    assert answer.passages == PASSAGES
    assert answer.citations[0].heading == "Poultry and stuffed foods"
    # the grounded prompt must contain the passage text and the question
    prompt = client.models.last_kwargs["contents"]
    assert "165°F (74°C) for 15 seconds" in prompt
    assert "minimum internal temperature for poultry" in prompt
    assert "only" in prompt.lower()  # answer ONLY from passages instruction


def test_gemini_failure_raises_generation_error():
    client = StubClient(error=ValueError("boom"))
    with pytest.raises(generate.GenerationError):
        generate.ask_model("hi", client=client)
    with pytest.raises(generate.GenerationError):
        generate.answer_question("hi", PASSAGES, client=client)

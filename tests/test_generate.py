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


from types import SimpleNamespace

from foodsafety_rag.generate import (
    SOURCES_MARKER,
    StreamChunk,
    stream_grounded,
)


class StubStreamModels:
    def __init__(self, pieces, error=None):
        self._pieces = pieces
        self._error = error
        self.last_kwargs = None

    def generate_content_stream(self, **kwargs):
        self.last_kwargs = kwargs
        if self._error:
            raise self._error
        for p in self._pieces:
            yield SimpleNamespace(text=p)


class StubStreamClient:
    def __init__(self, pieces=None, error=None):
        self.models = StubStreamModels(pieces or [], error)


def _run(pieces=None, error=None):
    client = StubStreamClient(pieces, error)
    chunks = list(stream_grounded("q?", PASSAGES, client=client))
    return client, chunks


def test_stream_grounded_yields_prose_tokens_then_citations():
    pieces = [
        "Poultry must reach ",
        "165°F (74°C).\n",
        SOURCES_MARKER + "\n",
        '[{"doc": "FDA Food Code §3-401 — Cooking Temperatures", '
        '"heading": "Poultry and stuffed foods", "snippet": "...165°F..."}]',
    ]
    client, chunks = _run(pieces)
    tokens = [c for c in chunks if c.kind == "token"]
    cites = [c for c in chunks if c.kind == "citations"]
    prose = "".join(c.text for c in tokens)
    assert "Poultry must reach 165°F (74°C)." in prose
    assert SOURCES_MARKER not in prose          # marker never leaks into prose
    assert "doc" not in prose                   # citation JSON never leaks into prose
    assert len(cites) == 1                       # exactly one citations chunk, last
    assert chunks[-1].kind == "citations"
    assert cites[0].citations[0].heading == "Poultry and stuffed foods"
    assert client.models.last_kwargs["model"] == "gemini-flash-latest"


def test_stream_grounded_malformed_tail_yields_empty_citations():
    client, chunks = _run(["Some answer.\n", SOURCES_MARKER + "\nnot valid json"])
    cites = [c for c in chunks if c.kind == "citations"]
    assert cites[0].citations == []             # graceful: caller falls back to passages
    prose = "".join(c.text for c in chunks if c.kind == "token")
    assert "Some answer." in prose


def test_stream_grounded_no_marker_emits_all_prose():
    client, chunks = _run(["Just prose, ", "no marker here."])
    prose = "".join(c.text for c in chunks if c.kind == "token")
    assert prose.strip() == "Just prose, no marker here."
    assert chunks[-1].kind == "citations" and chunks[-1].citations == []


def test_stream_grounded_wraps_sdk_error():
    import pytest
    from foodsafety_rag.generate import GenerationError
    with pytest.raises(GenerationError):
        list(stream_grounded("q?", PASSAGES, client=StubStreamClient(error=ValueError("boom"))))

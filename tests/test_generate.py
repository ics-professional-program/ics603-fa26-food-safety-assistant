from types import SimpleNamespace

import pytest
from pydantic_ai import ModelAPIError, UnexpectedModelBehavior, models
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from foodsafety_rag import generate
from foodsafety_rag.agent import ModelAnswer, food_safety_agent
from foodsafety_rag.generate import (
    SOURCES_MARKER,
    GenerationError,
    stream_grounded,
)
from foodsafety_rag.schemas import Answer, Citation, Passage

HOSTED_URL = "https://api.openai.com/v1"
LOCAL_URL = "http://localhost:1234/v1"


def _completion(content):
    """The shape of a non-streaming chat completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _delta(content):
    """The shape of one streamed chunk."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


class StubCompletions:
    def __init__(self, content=None, pieces=None, error=None):
        self._content = content
        self._pieces = pieces
        self._error = error
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._error:
            raise self._error
        if kwargs.get("stream"):
            return (_delta(p) for p in (self._pieces or []))
        return _completion(self._content)


class StubClient:
    def __init__(self, content=None, pieces=None, error=None):
        self.completions = StubCompletions(content, pieces, error)
        self.chat = SimpleNamespace(completions=self.completions)


class StubAgent:
    def __init__(self, output=None, *, usage=None, error=None):
        self.output = output
        self.run_usage = usage or RunUsage()
        self.error = error
        self.prompt = None
        self.deps = None

    def run_sync(self, prompt, *, deps):
        self.prompt = prompt
        self.deps = deps
        if self.error:
            raise self.error
        return SimpleNamespace(output=self.output, usage=self.run_usage)


PASSAGES = [
    Passage(doc="FDA Food Code §3-401 — Cooking Temperatures",
            heading="Poultry and stuffed foods",
            text="Raw poultry must reach 165°F (74°C) for 15 seconds.",
            score=0.82),
]


@pytest.fixture(autouse=True)
def _endpoint(monkeypatch):
    """Every test runs against a known endpoint, never the developer's own."""
    monkeypatch.setenv("LLM_BASE_URL", LOCAL_URL)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_PROVIDER", "course")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with models.override_allow_model_requests(False):
        yield


def test_hosted_endpoint_requires_key(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", HOSTED_URL)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        generate.get_client()


def test_local_endpoint_needs_no_key():
    client = generate.get_client()          # no LLM_API_KEY set - must not raise
    assert str(client.base_url).startswith(LOCAL_URL)


def test_client_is_fast_fail():
    client = generate.get_client()
    assert client.timeout == generate.REQUEST_TIMEOUT_S
    assert client.timeout <= 120           # bounded: a stuck call can't hang forever
    assert client.max_retries == generate.REQUEST_MAX_RETRIES
    assert 0 <= generate.REQUEST_MAX_RETRIES <= 2   # cap retries under a 503 storm


def test_ask_model_returns_text():
    client = StubClient(content="RAG retrieves passages first.")
    assert generate.ask_model("What is RAG?", client=client) == "RAG retrieves passages first."
    kwargs = client.completions.last_kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"][0]["content"] == "What is RAG?"


def test_answer_question_builds_grounded_answer():
    agent = StubAgent(ModelAnswer(
        answer="Poultry must reach 165°F (74°C) for 15 seconds.",
        supported=True,
        citations=[Citation(doc="FDA Food Code §3-401 — Cooking Temperatures",
                            heading="Poultry and stuffed foods",
                            snippet="...165°F (74°C)...")],
    ))
    answer = generate.answer_question(
        "What is the minimum internal temperature for poultry?", PASSAGES, agent=agent)
    assert isinstance(answer, Answer)
    assert answer.grounded is True
    assert answer.passages == PASSAGES
    assert answer.citations[0].heading == "Poultry and stuffed foods"
    # the grounded prompt must contain the passage text and the question
    prompt = agent.prompt
    assert "165°F (74°C) for 15 seconds" in prompt
    assert "minimum internal temperature for poultry" in prompt
    assert "only" in prompt.lower()  # answer ONLY from passages instruction
    assert agent.deps == PASSAGES


def test_answer_question_reports_token_cost():
    agent = StubAgent(
        ModelAnswer(answer="165°F.", supported=True, citations=[]),
        usage=RunUsage(input_tokens=812, output_tokens=64),
    )
    answer = generate.answer_question("q?", PASSAGES, agent=agent)
    assert answer.usage.prompt_tokens == 812
    assert answer.usage.completion_tokens == 64
    assert answer.usage.total_tokens == 876
    assert answer.usage.latency_ms >= 0


def test_usage_is_none_when_the_server_reports_none():
    """Not every OpenAI-compatible server returns a usage block."""
    agent = StubAgent(
        ModelAnswer(answer="a", supported=True, citations=[]),
        usage=RunUsage(requests=1),
    )
    assert generate.answer_question("q?", PASSAGES, agent=agent).usage is None


def test_stream_grounded_yields_usage_last():
    client = StubClient(pieces=["Some answer."])
    client.completions.create = lambda **kw: iter([
        _delta("Some answer."),
        SimpleNamespace(choices=[], usage=SimpleNamespace(
            prompt_tokens=800, completion_tokens=40)),
    ])
    chunks = list(stream_grounded("q?", PASSAGES, client=client))
    assert chunks[-1].kind == "usage"
    assert chunks[-1].usage.prompt_tokens == 800
    assert chunks[-1].usage.completion_tokens == 40


def test_stream_grounded_requests_usage_from_the_server():
    client, _ = _run(["prose"])
    assert client.completions.last_kwargs["stream_options"] == {"include_usage": True}


def test_agent_override_exercises_the_real_output_contract():
    output = (
        '{"answer":"165°F.","supported":true,"citations":[{'
        f'"doc":"{PASSAGES[0].doc}","heading":"{PASSAGES[0].heading}",'
        '"snippet":"165°F"}]}'
    )
    with food_safety_agent.override(model=TestModel(
        profile={"supports_json_schema_output": True}, custom_output_text=output
    )):
        result = food_safety_agent.run_sync("test", deps=PASSAGES)

    assert isinstance(result.output, ModelAnswer)
    assert result.output.citations[0].heading == PASSAGES[0].heading


def test_agent_rejects_a_citation_that_was_not_retrieved():
    output = (
        '{"answer":"invented","supported":true,"citations":['
        '{"doc":"missing","heading":"missing","snippet":"x"}]}'
    )
    with food_safety_agent.override(model=TestModel(
        profile={"supports_json_schema_output": True}, custom_output_text=output
    )):
        with pytest.raises(UnexpectedModelBehavior):
            food_safety_agent.run_sync("test", deps=PASSAGES)


def test_llm_failure_raises_generation_error():
    client = StubClient(error=ValueError("boom"))
    with pytest.raises(GenerationError):
        generate.ask_model("hi", client=client)
    agent = StubAgent(error=UnexpectedModelBehavior("bad output"))
    with pytest.raises(GenerationError):
        generate.answer_question("hi", PASSAGES, agent=agent)


def test_connection_failure_raises_generation_error():
    agent = StubAgent(error=ModelAPIError("test-model", "connection failed"))
    with pytest.raises(GenerationError):
        generate.answer_question("hi", PASSAGES, agent=agent)


def _run(pieces=None, error=None):
    client = StubClient(pieces=pieces, error=error)
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
    assert len(cites) == 1                       # exactly one citations chunk
    assert [c.kind for c in chunks[-2:]] == ["citations", "usage"]
    assert cites[0].citations[0].heading == "Poultry and stuffed foods"
    assert client.completions.last_kwargs["model"] == "test-model"
    assert client.completions.last_kwargs["stream"] is True


def test_stream_grounded_malformed_tail_yields_empty_citations():
    client, chunks = _run(["Some answer.\n", SOURCES_MARKER + "\nnot valid json"])
    cites = [c for c in chunks if c.kind == "citations"]
    assert cites[0].citations == []             # graceful: caller falls back to passages
    assert chunks[-1].kind == "usage"
    prose = "".join(c.text for c in chunks if c.kind == "token")
    assert "Some answer." in prose


def test_stream_grounded_no_marker_emits_all_prose():
    client, chunks = _run(["Just prose, ", "no marker here."])
    prose = "".join(c.text for c in chunks if c.kind == "token")
    assert prose.strip() == "Just prose, no marker here."
    assert [c.kind for c in chunks[-2:]] == ["citations", "usage"]
    assert chunks[-2].citations == []


def test_stream_grounded_tolerates_chunks_without_content():
    """Some servers end a stream with a usage-only chunk carrying no choices."""
    client = StubClient(pieces=["Answer text."])
    client.completions.create = lambda **kw: iter(
        [_delta("Answer "), SimpleNamespace(choices=[]), _delta("text.")])
    prose = "".join(c.text for c in stream_grounded("q?", PASSAGES, client=client)
                    if c.kind == "token")
    assert prose.strip() == "Answer text."


def test_stream_grounded_wraps_sdk_error():
    with pytest.raises(GenerationError):
        list(stream_grounded("q?", PASSAGES, client=StubClient(error=ValueError("boom"))))

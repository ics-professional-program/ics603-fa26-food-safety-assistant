from foodsafety_rag import stream
from foodsafety_rag.config import SIMILARITY_THRESHOLD
from foodsafety_rag.generate import GenerationError, StreamChunk
from foodsafety_rag.schemas import Answer, Citation


def _passage_row(score, heading="Poultry"):
    return {"doc": "FDA Food Code §3-401 — Cooking Temperatures",
            "heading": heading, "text": "Raw poultry must reach 165°F (74°C).",
            "score": score}


def _patch(monkeypatch, rows, stream_chunks=None):
    calls = {"logged": [], "generated": 0, "outcomes": []}
    monkeypatch.setattr(stream.embed, "embed_text", lambda q: [0.1] * 384)
    monkeypatch.setattr(stream.store, "log_query",
                        lambda conn, q: calls["logged"].append(q) or 1)
    monkeypatch.setattr(stream.store, "log_outcome",
                        lambda conn, qid, **kw: calls["outcomes"].append(kw))
    monkeypatch.setattr(stream.store, "similarity_search",
                        lambda conn, vec, top_k: rows)

    def fake_stream_grounded(question, passages, *, client=None):
        calls["generated"] += 1
        yield from (stream_chunks or [])

    monkeypatch.setattr(stream, "stream_grounded", fake_stream_grounded)
    return calls


def test_grounded_path_event_order(monkeypatch):
    rows = [_passage_row(SIMILARITY_THRESHOLD + 0.3)]
    chunks = [StreamChunk(kind="token", text="165°F for 15 s."),
              StreamChunk(kind="citations",
                          citations=[Citation(doc="d", heading="h", snippet="s")])]
    calls = _patch(monkeypatch, rows, chunks)
    events = list(stream.stream_events("q?", conn=object()))
    steps = [(e["type"], e.get("step")) for e in events]
    assert steps[:3] == [("stage", "embed"), ("stage", "retrieve"), ("stage", "guard")]
    assert ("stage", "generate") in steps
    assert ("token", None) in steps
    assert events[-1]["type"] == "answer"
    ans = events[-1]
    assert ans["grounded"] is True
    assert ans["answer"] == "165°F for 15 s."
    assert ans["citations"][0]["heading"] == "h"
    assert calls["generated"] == 1 and calls["logged"] == ["q?"]


def test_guard_event_reports_score_and_threshold(monkeypatch):
    _patch(monkeypatch, [_passage_row(0.71)],
           [StreamChunk(kind="citations", citations=[])])
    guard = next(e for e in stream.stream_events("q?", conn=object())
                 if e.get("step") == "guard")
    assert guard["grounded"] is True
    assert guard["threshold"] == SIMILARITY_THRESHOLD
    assert abs(guard["top_score"] - 0.71) < 1e-9


def test_declined_path_skips_generation(monkeypatch):
    calls = _patch(monkeypatch, [_passage_row(SIMILARITY_THRESHOLD - 0.1)])
    events = list(stream.stream_events("Can I bring my dog in?", conn=object()))
    assert calls["generated"] == 0                        # NO model call
    assert not any(e.get("step") == "generate" for e in events)
    assert any(e["type"] == "token" for e in events)      # decline text still streams
    ans = events[-1]
    assert ans["type"] == "answer" and ans["grounded"] is False
    assert ans["citations"] == []
    assert len(ans["passages"]) == 1                       # weak passages still shown


def test_grounded_empty_citations_fall_back_to_passages(monkeypatch):
    rows = [_passage_row(0.7)]
    _patch(monkeypatch, rows, [StreamChunk(kind="token", text="ans"),
                               StreamChunk(kind="citations", citations=[])])
    ans = list(stream.stream_events("q?", conn=object()))[-1]
    assert ans["citations"], "empty model citations must fall back to retrieved passages"
    assert ans["citations"][0]["heading"] == "Poultry"


def test_optional_provider_uses_typed_answer_path(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    rows = [_passage_row(0.7)]
    calls = _patch(monkeypatch, rows)
    typed_calls = []

    def fake_answer(question, passages, *, agent=None):
        typed_calls.append((question, passages))
        return Answer(
            question=question,
            answer="Poultry must reach 165°F.",
            grounded=True,
            citations=[Citation(doc=passages[0].doc,
                                heading=passages[0].heading,
                                snippet="165°F")],
            passages=passages,
        )

    monkeypatch.setattr(stream, "answer_question", fake_answer)
    events = list(stream.stream_events("q?", conn=object()))

    assert typed_calls and calls["generated"] == 0
    assert any(event["type"] == "token" for event in events)
    assert events[-1]["type"] == "answer"
    assert events[-1]["answer"] == "Poultry must reach 165°F."


def test_generation_error_yields_error_event(monkeypatch):
    rows = [_passage_row(0.7)]  # high score to take grounded branch
    calls = {"logged": []}

    # Set up the mocks for embed, log, search
    monkeypatch.setattr(stream.embed, "embed_text", lambda q: [0.1] * 384)
    monkeypatch.setattr(stream.store, "log_query",
                        lambda conn, q: calls["logged"].append(q) or 1)
    monkeypatch.setattr(stream.store, "log_outcome", lambda conn, qid, **kw: None)
    monkeypatch.setattr(stream.store, "similarity_search",
                        lambda conn, vec, top_k: rows)

    # Make stream_grounded raise GenerationError
    def fake_stream_grounded_raises(question, passages, *, client=None):
        raise GenerationError("boom")

    monkeypatch.setattr(stream, "stream_grounded", fake_stream_grounded_raises)

    # Collect all events
    events = list(stream.stream_events("q?", conn=object()))

    # Check that there's an error event
    error_event = next((e for e in events if e["type"] == "error"), None)
    assert error_event is not None, "Should have an error event"
    assert error_event["detail"], "Error event should have non-empty detail"

    # Check that there's NO answer event
    answer_events = [e for e in events if e["type"] == "answer"]
    assert not answer_events, "Should have no answer event when generation fails"

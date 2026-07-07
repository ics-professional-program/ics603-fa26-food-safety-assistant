from foodsafety_rag import pipeline
from foodsafety_rag.config import SIMILARITY_THRESHOLD
from foodsafety_rag.schemas import Answer, Passage


def _passage(score):
    return Passage(doc="FDA Food Code §3-401 — Cooking Temperatures",
                   heading="Poultry and stuffed foods",
                   text="Raw poultry must reach 165°F (74°C).", score=score)


def _patch(monkeypatch, passages, generated=None):
    calls = {"generate": 0, "logged": []}
    monkeypatch.setattr(pipeline, "retrieve",
                        lambda question, *, conn, top_k=4: passages)
    monkeypatch.setattr(pipeline.store, "log_query",
                        lambda conn, q: calls["logged"].append(q))

    def fake_answer_question(question, psgs, *, client=None):
        calls["generate"] += 1
        return generated

    monkeypatch.setattr(pipeline, "answer_question", fake_answer_question)
    return calls


def test_good_retrieval_calls_gemini(monkeypatch):
    passages = [_passage(SIMILARITY_THRESHOLD + 0.3)]
    generated = Answer(question="q", answer="165°F for 15 seconds.",
                       grounded=True, passages=passages)
    calls = _patch(monkeypatch, passages, generated)
    result = pipeline.grounded_answer("q", conn=object())
    assert result is generated
    assert calls["generate"] == 1
    assert calls["logged"] == ["q"]


def test_low_similarity_short_circuits_without_gemini(monkeypatch):
    passages = [_passage(SIMILARITY_THRESHOLD - 0.1)]
    calls = _patch(monkeypatch, passages)
    result = pipeline.grounded_answer("Can I bring my dog into the kitchen?",
                                      conn=object())
    assert result.grounded is False
    assert result.answer == pipeline.NOT_FOUND_ANSWER
    assert result.citations == []
    assert result.passages == passages  # UI still shows what was (weakly) retrieved
    assert calls["generate"] == 0       # the guard: no fabricated answer


def test_empty_retrieval_short_circuits(monkeypatch):
    calls = _patch(monkeypatch, [])
    result = pipeline.grounded_answer("anything", conn=object())
    assert result.grounded is False
    assert calls["generate"] == 0

import pytest
from pydantic import ValidationError

from foodsafety_rag.schemas import Answer, AskRequest, Citation, Passage


def test_answer_round_trip():
    answer = Answer(
        question="What is the minimum internal temperature for poultry?",
        answer="Poultry must reach 165°F (74°C) for 15 seconds.",
        grounded=True,
        citations=[
            Citation(
                doc="FDA Food Code §3-401 — Cooking Temperatures",
                heading="Poultry and stuffed foods",
                snippet="...165°F (74°C) for at least 15 seconds...",
            )
        ],
        passages=[
            Passage(
                doc="FDA Food Code §3-401 — Cooking Temperatures",
                heading="Poultry and stuffed foods",
                text="Raw poultry ... 165°F (74°C) for at least 15 seconds.",
                score=0.82,
            )
        ],
    )
    data = answer.model_dump()
    assert data["grounded"] is True
    assert data["citations"][0]["heading"] == "Poultry and stuffed foods"
    assert Answer.model_validate(data) == answer


def test_answer_defaults_empty_lists():
    a = Answer(question="q", answer="not found", grounded=False)
    assert a.citations == [] and a.passages == []


def test_ask_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(question="")


def test_ask_request_rejects_overlong_question():
    with pytest.raises(ValidationError):
        AskRequest(question="x" * 501)

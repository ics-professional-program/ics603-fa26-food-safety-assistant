"""Scoring for the labeled eval set (course session 13.0).

Retrieval and generation are scored SEPARATELY, because a RAG pipeline can
fail on either side independently: the right passage retrieved and a poor
answer written from it, or a fluent answer built on the wrong passage.

Deterministic generation rule: an answer passes when every temperature/time
value extracted from the reference answer also appears in the answer, after
unit normalization (F, C, sec, min, hr). Extra values in the answer are
reported but do not fail the check; missing required values do. Values are
extracted with VALUE_RE: a number followed by a degree/time unit, so "165°F
(74°C) for 15 seconds" yields {(165.0, "F"), (74.0, "C"), (15.0, "sec")}.

For an unanswerable case (``answerable: false``) the deterministic rule is
instead: the pipeline must DECLINE (``grounded == False``). Declining is the
correct behavior, not a failure.
"""

import re

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelAPIError, NativeOutput, UnexpectedModelBehavior

from foodsafety_rag.agent import build_model
from foodsafety_rag.generate import GenerationError
from foodsafety_rag.pipeline import grounded_answer
from foodsafety_rag.schemas import Answer, Passage

VALUE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:°\s*)?"
    r"(F|C|seconds?|secs?|minutes?|mins?|hours?|hrs?)\b",
    re.IGNORECASE)

_UNIT = {"f": "F", "c": "C",
         "second": "sec", "seconds": "sec", "sec": "sec", "secs": "sec",
         "minute": "min", "minutes": "min", "min": "min", "mins": "min",
         "hour": "hr", "hours": "hr", "hr": "hr", "hrs": "hr"}


def extract_values(text: str) -> set[tuple[float, str]]:
    """All (number, unit) pairs in the text, units normalized."""
    return {(float(num), _UNIT[unit.lower()])
            for num, unit in VALUE_RE.findall(text)}


def deterministic_pass(answer_text: str,
                       reference_text: str) -> tuple[bool, set]:
    """(passed, missing values). See the module docstring for the rule."""
    missing = extract_values(reference_text) - extract_values(answer_text)
    return (not missing, missing)


def match_rank(passages: list[Passage], doc_title: str,
               heading: str) -> int | None:
    """1-based rank of the labeled chunk among retrieved passages, or None."""
    for i, passage in enumerate(passages, start=1):
        if passage.doc == doc_title and passage.heading == heading:
            return i
    return None


def title_for_path(conn, path: str) -> str | None:
    """Retrieved passages carry the document TITLE; the eval set labels the
    document PATH. The documents table maps one to the other."""
    row = conn.execute(
        "SELECT title FROM documents WHERE path = %s", (path,)).fetchone()
    return row[0] if row else None


JUDGE_RUBRIC = """\
You grade one answer from a food-safety assistant. Score each axis 0-2
(0 = fails, 1 = partial, 2 = meets it):
- faithful: every claim is supported by the supplied passages; nothing is
  invented.
- answers_question: it addresses exactly what was asked.
- cites_sources: it names the documents its claims come from.
- handles_conflict: if the supplied passages disagree with each other, the
  answer presents both sides and says which applies; score 2 if no conflict
  is present and none is claimed.
The reference answer describes the expected content, including whether the
correct behavior is to decline."""


class JudgeVerdict(BaseModel):
    faithful: int = Field(ge=0, le=2)
    answers_question: int = Field(ge=0, le=2)
    cites_sources: int = Field(ge=0, le=2)
    handles_conflict: int = Field(ge=0, le=2)

    @property
    def total(self) -> int:
        return (self.faithful + self.answers_question
                + self.cites_sources + self.handles_conflict)


judge_agent = Agent(
    build_model(),
    output_type=NativeOutput(JudgeVerdict),
    instructions=JUDGE_RUBRIC,
    defer_model_check=True,
)


def _judge_prompt(case: dict, answer: Answer) -> str:
    passages = "\n\n".join(
        f"[{i + 1}] {p.doc} — {p.heading}\n{p.text}"
        for i, p in enumerate(answer.passages)) or "(none retrieved)"
    return (f"Question: {case['question']}\n\n"
            f"Reference answer (expected content):\n{case['reference_answer']}\n\n"
            f"Supplied passages:\n{passages}\n\n"
            f"Answer under review:\n{answer.answer}")


def run_judge(case: dict, answer: Answer, *, agent: Agent = judge_agent) -> JudgeVerdict:
    return agent.run_sync(_judge_prompt(case, answer)).output


def evaluate_question(conn, case: dict, *, judge: bool = True,
                      judge_agent_override: Agent | None = None) -> dict:
    """Run one eval case through the real pipeline and score it.

    A pipeline error on a question gets ONE recorded retry (the ``attempts``
    field says which answers took two); a second failure scores the question
    as failed (an ``error`` field carries the reason) instead of aborting the
    run - an eval that dies on its first flaky call measures nothing.
    """
    attempts = 1
    try:
        answer = grounded_answer(case["question"], conn=conn)
    except GenerationError:
        attempts = 2
        try:
            answer = grounded_answer(case["question"], conn=conn)
        except GenerationError as exc:
            return {
                "id": case["id"], "retrieval_hit": None,
                "retrieval_rank": None, "det_pass": False, "det_missing": [],
                "judge_total": None, "judge": None, "grounded": None,
                "answer": "", "attempts": attempts, "error": str(exc),
            }

    retrieval_hit = None
    retrieval_rank = None
    if case.get("expected_chunk"):
        title = title_for_path(conn, case["expected_chunk"]["path"])
        retrieval_rank = (match_rank(answer.passages, title,
                                     case["expected_chunk"]["heading"])
                          if title else None)
        retrieval_hit = retrieval_rank is not None

    if case["answerable"]:
        det_pass, det_missing = deterministic_pass(
            answer.answer, case["reference_answer"])
        if not answer.grounded:
            det_pass = False
    else:
        det_pass, det_missing = (not answer.grounded), set()

    verdict = None
    judge_error = None
    if judge:
        try:
            verdict = run_judge(case, answer,
                                agent=judge_agent_override or judge_agent)
        except (ModelAPIError, UnexpectedModelBehavior) as exc:
            judge_error = f"judge failed ({type(exc).__name__})"

    return {
        "id": case["id"],
        "retrieval_hit": retrieval_hit,
        "retrieval_rank": retrieval_rank,
        "det_pass": det_pass,
        "det_missing": sorted(f"{v:g} {u}" for v, u in det_missing),
        "judge_total": verdict.total if verdict else None,
        "judge": verdict.model_dump() if verdict else None,
        "grounded": answer.grounded,
        "answer": answer.answer,
        "attempts": attempts,
        "error": judge_error,
    }

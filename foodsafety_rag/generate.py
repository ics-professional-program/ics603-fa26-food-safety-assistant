"""Gemini generation (course module M4).

Two entry points:
- ask_model():        plain, ungrounded call (the 1.3 notebook + contrast script)
- answer_question():  grounded call over retrieved passages, structured output

The API key is read only from the environment. Tests always pass a stub client.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from pydantic import BaseModel

from foodsafety_rag.config import GEMINI_MODEL, get_settings
from foodsafety_rag.schemas import Answer, Citation, Passage

# Fail fast instead of hanging: the free-tier model can return 503 "high demand"
# for a while, and the SDK's default retry/backoff can stall an interactive
# request for minutes. Bound the per-request time and cap the retry attempts so a
# struggling call surfaces the friendly error in seconds, not minutes.
REQUEST_TIMEOUT_MS = 15_000   # per-request HTTP timeout (also caps a stalled stream)
REQUEST_RETRY_ATTEMPTS = 2    # total attempts incl. the original (one quick retry)


def _http_options() -> types.HttpOptions:
    return types.HttpOptions(
        timeout=REQUEST_TIMEOUT_MS,
        retry_options=types.HttpRetryOptions(
            attempts=REQUEST_RETRY_ATTEMPTS,
            initial_delay=0.5,
            max_delay=2.0,
        ),
    )


class GenerationError(RuntimeError):
    """Raised when the Gemini call fails; the API layer maps it to a friendly 502."""


class ModelAnswer(BaseModel):
    """The structured response schema Gemini fills in (native responseSchema)."""
    answer: str
    supported: bool
    citations: list[Citation]


GROUNDED_PROMPT = """\
You are a food-safety compliance assistant for restaurant staff.
Answer the question using ONLY the numbered passages below - do not use any
other knowledge. Quote temperatures, times, and concentrations exactly as
written. For each fact you state, add a citation with the passage's doc,
heading, and a short snippet. If the passages do not contain the answer, set
supported=false and say the trusted documents do not cover this question.

Passages:
{passages}

Question: {question}
"""


def get_client() -> genai.Client:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env, add your key "
            "(https://aistudio.google.com/apikey), and load it into the environment."
        )
    return genai.Client(api_key=settings.gemini_api_key, http_options=_http_options())


def ask_model(prompt: str, *, client: genai.Client | None = None) -> str:
    """One plain LLM call - no retrieval, no grounding. The 1.3 'first taste'."""
    client = client or get_client()
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    except Exception as exc:  # network, quota, auth - never leak a stack trace to users
        raise GenerationError(f"Gemini call failed ({type(exc).__name__})") from exc
    return response.text


def answer_question(question: str, passages: list[Passage],
                    *, client: genai.Client | None = None) -> Answer:
    """Grounded call: passages + question -> structured Answer."""
    client = client or get_client()
    passage_block = "\n\n".join(
        f"[{i + 1}] {p.doc} — {p.heading}\n{p.text}"
        for i, p in enumerate(passages)
    )
    prompt = GROUNDED_PROMPT.format(passages=passage_block, question=question)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ModelAnswer,
            },
        )
        parsed: ModelAnswer = response.parsed
    except Exception as exc:
        raise GenerationError(f"Gemini call failed ({type(exc).__name__})") from exc
    return Answer(
        question=question,
        answer=parsed.answer,
        grounded=parsed.supported,
        citations=parsed.citations,
        passages=passages,
    )


SOURCES_MARKER = "---SOURCES---"

STREAM_GROUNDED_PROMPT = """\
You are a food-safety compliance assistant for restaurant staff.
Answer the question using ONLY the numbered passages below - do not use any
other knowledge. Quote temperatures, times, and concentrations exactly as
written. If the passages do not contain the answer, say the trusted documents
do not cover this question.

First write the answer as plain prose. Then, on its own line, write exactly:
---SOURCES---
followed by a JSON array of the passages you used, each as an object with
"doc", "heading", and a short "snippet". Output nothing after the JSON array.

Passages:
{passages}

Question: {question}
"""


@dataclass
class StreamChunk:
    kind: str                       # "token" | "citations"
    text: str = ""
    citations: list[Citation] = field(default_factory=list)


def _parse_citations(tail: str) -> list[Citation]:
    """Parse the JSON array after the marker; return [] on any failure."""
    text = tail.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        raw = json.loads(text[start : end + 1])
        return [Citation(doc=c["doc"], heading=c["heading"], snippet=c["snippet"])
                for c in raw]
    except (ValueError, KeyError, TypeError):
        return []


def stream_grounded(question: str, passages: list[Passage],
                    *, client: genai.Client | None = None) -> Iterator[StreamChunk]:
    """Grounded streaming call: yields prose `token` chunks up to the sources
    marker, then one `citations` chunk parsed from the tail."""
    client = client or get_client()
    passage_block = "\n\n".join(
        f"[{i + 1}] {p.doc} — {p.heading}\n{p.text}" for i, p in enumerate(passages)
    )
    prompt = STREAM_GROUNDED_PROMPT.format(passages=passage_block, question=question)

    buffer = ""
    emitted = 0
    marker_at = -1
    holdback = len(SOURCES_MARKER) - 1
    try:
        for chunk in client.models.generate_content_stream(
            model=GEMINI_MODEL, contents=prompt
        ):
            piece = getattr(chunk, "text", None) or ""
            if not piece:
                continue
            buffer += piece
            if marker_at == -1:
                marker_at = buffer.find(SOURCES_MARKER)
            if marker_at == -1:
                # Emit all but the last `holdback` chars, in case a marker is forming.
                safe_end = max(emitted, len(buffer) - holdback)
                if safe_end > emitted:
                    yield StreamChunk(kind="token", text=buffer[emitted:safe_end])
                    emitted = safe_end
            elif emitted < marker_at:
                yield StreamChunk(kind="token", text=buffer[emitted:marker_at])
                emitted = marker_at
    except Exception as exc:  # network, quota, auth
        raise GenerationError(f"Gemini call failed ({type(exc).__name__})") from exc

    if marker_at == -1 and emitted < len(buffer):
        yield StreamChunk(kind="token", text=buffer[emitted:])       # no marker: flush prose
    citations = _parse_citations(buffer[marker_at + len(SOURCES_MARKER):]) if marker_at != -1 else []
    yield StreamChunk(kind="citations", citations=citations)

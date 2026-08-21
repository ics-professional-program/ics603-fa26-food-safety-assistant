"""LLM generation examples (course module M4).

Two entry points:
- ask_model():        plain, ungrounded call (the 1.3 notebook + contrast script)
- answer_question():  grounded Pydantic AI call with validated structured output

The endpoint, model, and key are read only from the environment, so the same
raw SDK examples run against the course-provided server, a local model server
(LM Studio, Ollama, vLLM), or api.openai.com. The structured path selects its
provider in ``foodsafety_rag.agent``.
"""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from openai import OpenAI
from pydantic_ai import ModelAPIError, UnexpectedModelBehavior

from foodsafety_rag.agent import food_safety_agent, is_local_url
from foodsafety_rag.config import get_settings
from foodsafety_rag.schemas import Answer, Citation, Passage, Usage

# Fail fast instead of hanging: a busy endpoint can stall an interactive request
# for minutes. Bound the per-request time and cap the retries so a struggling
# call surfaces the friendly error instead. The timeout is generous because a
# local model server generates far slower than a hosted one.
REQUEST_TIMEOUT_S = 90.0     # per-request HTTP timeout (also caps a stalled stream)
REQUEST_MAX_RETRIES = 1      # retries after the original attempt (one quick retry)

class GenerationError(RuntimeError):
    """Raised when the model call fails; the API layer maps it to a friendly 502."""


GROUNDED_PROMPT = """\
You are a food-safety compliance assistant for restaurant staff.
Answer the question using ONLY the numbered passages below - do not use any
other knowledge. Quote temperatures, times, and concentrations exactly as
written. Write plain prose with no Markdown formatting - no asterisks, no bold,
no bullet characters. For each fact you state, add a citation with the passage's
doc, heading, and a short snippet. If the passages do not contain the answer,
set supported=false and say the trusted documents do not cover this question.

Passages:
{passages}

Question: {question}
"""


def get_client() -> OpenAI:
    """Build the raw OpenAI-compatible client used by the early and stream demos."""

    settings = get_settings()
    if settings.llm_provider != "course":
        raise RuntimeError(
            "The raw SDK and streaming examples require LLM_PROVIDER=course. "
            "The structured /ask path supports all configured providers."
        )
    if not settings.llm_api_key:
        if not is_local_url(settings.llm_base_url):
            raise RuntimeError(
                f"LLM_API_KEY is not set, and {settings.llm_base_url} is not a "
                "local server. Add the course or endpoint key to .env."
            )
    return OpenAI(
        base_url=settings.llm_base_url,
        # A local server does not authenticate, but the client still wants a value.
        api_key=settings.llm_api_key or "local",
        timeout=REQUEST_TIMEOUT_S,
        max_retries=REQUEST_MAX_RETRIES,
    )


def ask_model(prompt: str, *, client: OpenAI | None = None) -> str:
    """One plain LLM call - no retrieval, no grounding. The 1.3 'first taste'."""
    client = client or get_client()
    try:
        response = client.chat.completions.create(
            model=get_settings().llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # network, quota, auth - never leak a stack trace to users
        raise GenerationError(f"LLM call failed ({type(exc).__name__})") from exc
    return response.choices[0].message.content or ""


def _usage(response, elapsed_s: float) -> Usage | None:
    """Read the token counts off a response. Not every OpenAI-compatible server
    reports them, so a missing usage block is normal rather than an error."""
    u = getattr(response, "usage", None)
    if u is None or u.prompt_tokens is None:
        return None
    return Usage(prompt_tokens=u.prompt_tokens,
                 completion_tokens=u.completion_tokens or 0,
                 latency_ms=int(elapsed_s * 1000))


def _passage_block(passages: list[Passage]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {p.doc} — {p.heading}\n{p.text}"
        for i, p in enumerate(passages)
    )


def answer_question(question: str, passages: list[Passage],
                    *, agent=food_safety_agent) -> Answer:
    """Grounded call: passages + question -> Pydantic-validated Answer."""

    prompt = GROUNDED_PROMPT.format(passages=_passage_block(passages), question=question)
    started = time.monotonic()
    try:
        result = agent.run_sync(prompt, deps=passages)
    except (ModelAPIError, UnexpectedModelBehavior) as exc:
        raise GenerationError(f"LLM call failed ({type(exc).__name__})") from exc

    run_usage = result.usage
    usage = None
    if run_usage.input_tokens or run_usage.output_tokens:
        usage = Usage(
            prompt_tokens=run_usage.input_tokens or 0,
            completion_tokens=run_usage.output_tokens or 0,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    parsed = result.output
    return Answer(
        question=question,
        answer=parsed.answer,
        grounded=parsed.supported,
        citations=parsed.citations,
        passages=passages,
        usage=usage,
    )


SOURCES_MARKER = "---SOURCES---"

STREAM_GROUNDED_PROMPT = """\
You are a food-safety compliance assistant for restaurant staff.
Answer the question using ONLY the numbered passages below - do not use any
other knowledge. Quote temperatures, times, and concentrations exactly as
written. If the passages do not contain the answer, say the trusted documents
do not cover this question.

First write the answer as plain prose, with no Markdown formatting - no
asterisks, no bold, no bullet characters. Then, on its own line, write
exactly:
---SOURCES---
followed by a JSON array of the passages you used, each as an object with
"doc", "heading", and a short "snippet". Output nothing after the JSON array.

Passages:
{passages}

Question: {question}
"""


@dataclass
class StreamChunk:
    kind: str                       # "token" | "citations" | "usage"
    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    usage: Usage | None = None


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


def _delta_text(chunk) -> str:
    """The text in one streamed chunk. Some servers send trailing chunks with no
    choices (usage-only), so this never assumes a delta is there."""
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    return getattr(choices[0].delta, "content", None) or ""


def stream_grounded(question: str, passages: list[Passage],
                    *, client: OpenAI | None = None) -> Iterator[StreamChunk]:
    """Grounded streaming call: yields prose `token` chunks up to the sources
    marker, then one `citations` chunk parsed from the tail."""
    client = client or get_client()
    prompt = STREAM_GROUNDED_PROMPT.format(
        passages=_passage_block(passages), question=question)

    buffer = ""
    emitted = 0
    marker_at = -1
    holdback = len(SOURCES_MARKER) - 1
    usage = None
    started = time.monotonic()
    try:
        for chunk in client.chat.completions.create(
            model=get_settings().llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            # Ask for the token counts on the final chunk; a streamed response
            # omits them otherwise.
            stream_options={"include_usage": True},
        ):
            usage = _usage(chunk, time.monotonic() - started) or usage
            piece = _delta_text(chunk)
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
        raise GenerationError(f"LLM call failed ({type(exc).__name__})") from exc

    if marker_at == -1 and emitted < len(buffer):
        yield StreamChunk(kind="token", text=buffer[emitted:])       # no marker: flush prose
    citations = _parse_citations(buffer[marker_at + len(SOURCES_MARKER):]) if marker_at != -1 else []
    yield StreamChunk(kind="citations", citations=citations)
    yield StreamChunk(kind="usage", usage=usage)

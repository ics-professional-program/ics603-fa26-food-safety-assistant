"""Event-generator form of the grounded pipeline (course module M4/M8).

Yields event dicts for each stage so the API can stream them as SSE. The
non-streaming pipeline.grounded_answer stays the simple path for POST /ask.
"""

import time
from collections.abc import Iterator

from foodsafety_rag import embed, store
from foodsafety_rag.config import SIMILARITY_THRESHOLD, TOP_K
from foodsafety_rag.generate import GenerationError, stream_grounded
from foodsafety_rag.pipeline import NOT_FOUND_ANSWER
from foodsafety_rag.schemas import Citation, Passage

_FRIENDLY_GEN_ERROR = "The language model call failed. Please try again in a moment."


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _chunk_text(text: str) -> list[str]:
    """Split text into word-ish pieces so the decline message 'streams' too."""
    return [w + " " for w in text.split(" ")]


def stream_events(question: str, *, conn, client=None) -> Iterator[dict]:
    store.log_query(conn, question)

    t = time.monotonic()
    vec = embed.embed_text(question)
    yield {"type": "stage", "step": "embed", "status": "done",
           "detail": f"{len(vec)}-dim", "elapsed_ms": _ms(t)}

    t = time.monotonic()
    rows = store.similarity_search(conn, vec, TOP_K)
    passages = [Passage(**r) for r in rows]
    yield {"type": "stage", "step": "retrieve", "status": "done",
           "candidates": [{"doc": p.doc, "heading": p.heading, "score": p.score}
                          for p in passages],
           "elapsed_ms": _ms(t)}

    top = passages[0].score if passages else 0.0
    grounded = bool(passages) and top >= SIMILARITY_THRESHOLD
    yield {"type": "stage", "step": "guard", "status": "done",
           "top_score": top, "threshold": SIMILARITY_THRESHOLD, "grounded": grounded}

    if not grounded:
        for piece in _chunk_text(NOT_FOUND_ANSWER):
            yield {"type": "token", "text": piece}
        yield {"type": "answer", "question": question, "answer": NOT_FOUND_ANSWER,
               "grounded": False, "citations": [],
               "passages": [p.model_dump() for p in passages]}
        return

    t = time.monotonic()
    yield {"type": "stage", "step": "generate", "status": "start", "detail": "asking Gemini"}
    prose: list[str] = []
    citations: list[Citation] = []
    try:
        for sc in stream_grounded(question, passages, client=client):
            if sc.kind == "token":
                prose.append(sc.text)
                yield {"type": "token", "text": sc.text}
            elif sc.kind == "citations":
                citations = sc.citations
    except GenerationError:
        yield {"type": "error", "detail": _FRIENDLY_GEN_ERROR}
        return

    if not citations:  # fall back to the retrieved passages as the evidence
        citations = [Citation(doc=p.doc, heading=p.heading, snippet=p.text[:160])
                     for p in passages]
    yield {"type": "stage", "step": "generate", "status": "done", "elapsed_ms": _ms(t)}
    yield {"type": "answer", "question": question, "answer": "".join(prose).strip(),
           "grounded": True,
           "citations": [c.model_dump() for c in citations],
           "passages": [p.model_dump() for p in passages]}

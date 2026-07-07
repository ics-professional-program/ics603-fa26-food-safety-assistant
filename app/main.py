"""FastAPI service (course module M4). Thin shell over foodsafety_rag.

Endpoints: POST /ask, GET /, GET /health.
REPLAY=1 serves captured fixture answers (class-demo fallback: no key, no DB).
"""

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from foodsafety_rag import pipeline, store, stream
from foodsafety_rag.config import SIMILARITY_THRESHOLD, get_settings
from foodsafety_rag.generate import GenerationError
from foodsafety_rag.schemas import Answer, AskRequest

STATIC_DIR = Path(__file__).parent / "static"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _sse(event: dict) -> str:
    data = {k: v for k, v in event.items() if k != 'type'}
    return f"event: {event['type']}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _replay_stream(question: str):
    fixture = app.state.fixtures.get(_normalize(question))
    if fixture is None:
        yield _sse({"type": "stage", "step": "guard", "status": "done",
                    "top_score": 0.0, "threshold": SIMILARITY_THRESHOLD,
                    "grounded": False})
        for piece in stream._chunk_text(pipeline.NOT_FOUND_ANSWER):
            yield _sse({"type": "token", "text": piece})
        yield _sse({"type": "answer", "question": question,
                    "answer": pipeline.NOT_FOUND_ANSWER, "grounded": False,
                    "citations": [], "passages": []})
        yield _sse({"type": "done"})
        return

    yield _sse({"type": "stage", "step": "embed", "status": "done",
                "detail": "384-dim (replayed)", "elapsed_ms": 0})
    yield _sse({"type": "stage", "step": "retrieve", "status": "done",
                "candidates": [{"doc": p.doc, "heading": p.heading, "score": p.score}
                               for p in fixture.passages], "elapsed_ms": 0})
    yield _sse({"type": "stage", "step": "guard", "status": "done",
                "top_score": fixture.passages[0].score if fixture.passages else 0.0,
                "threshold": SIMILARITY_THRESHOLD, "grounded": fixture.grounded})
    if fixture.grounded:
        yield _sse({"type": "stage", "step": "generate", "status": "start",
                    "detail": "replaying"})
    for piece in stream._chunk_text(fixture.answer):
        yield _sse({"type": "token", "text": piece})
    yield _sse({"type": "answer", **fixture.model_dump()})
    yield _sse({"type": "done"})


def _normalize(question: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", question.lower()).strip()


def load_fixtures(fixtures_dir: Path) -> dict[str, Answer]:
    fixtures: dict[str, Answer] = {}
    if fixtures_dir.is_dir():
        for path in sorted(fixtures_dir.glob("*.json")):
            answer = Answer.model_validate(
                json.loads(path.read_text(encoding="utf-8")))
            fixtures[_normalize(answer.question)] = answer
    return fixtures


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.replay:
        app.state.fixtures = load_fixtures(FIXTURES_DIR)
    elif not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "key, or set REPLAY=1 to serve captured answers offline."
        )
    yield


app = FastAPI(title="Food-Safety Compliance Assistant", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/ask", response_model=Answer)
def ask(request: AskRequest) -> Answer:
    settings = get_settings()

    if settings.replay:
        fixture = app.state.fixtures.get(_normalize(request.question))
        if fixture is not None:
            return fixture
        return Answer(question=request.question, answer=pipeline.NOT_FOUND_ANSWER,
                      grounded=False)

    try:
        with store.get_conn(settings.database_url) as conn:
            return pipeline.grounded_answer(request.question, conn=conn)
    except psycopg.OperationalError:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Is the pgvector container running? "
                   "Start it with: docker compose up -d db",
        )
    except GenerationError:
        raise HTTPException(
            status_code=502,
            detail="The language model call failed. Please try again in a moment.",
        )


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    settings = get_settings()

    if settings.replay:
        return StreamingResponse(_replay_stream(request.question),
                                 media_type="text/event-stream", headers=SSE_HEADERS)

    def live():
        try:
            with store.get_conn(settings.database_url) as conn:
                for event in stream.stream_events(request.question, conn=conn):
                    yield _sse(event)
        except psycopg.OperationalError:
            yield _sse({"type": "error",
                        "detail": "Database unavailable. Is the pgvector container "
                                  "running? Start it with: docker compose up -d db"})
        except Exception:
            yield _sse({"type": "error",
                        "detail": "The request failed. Please try again in a moment."})
        else:
            yield _sse({"type": "done"})
            return
        yield _sse({"type": "done"})

    return StreamingResponse(live(), media_type="text/event-stream", headers=SSE_HEADERS)

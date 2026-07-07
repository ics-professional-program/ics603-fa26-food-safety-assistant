import psycopg
import pytest
from fastapi.testclient import TestClient

from foodsafety_rag.generate import GenerationError
from foodsafety_rag.schemas import Answer


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")  # fake; nothing real is called
    monkeypatch.delenv("REPLAY", raising=False)
    from app import main
    with TestClient(main.app) as c:
        yield c, main


class FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_returns_answer_contract(client, monkeypatch):
    c, main = client
    answer = Answer(question="q?", answer="165°F.", grounded=True)
    monkeypatch.setattr(main.store, "get_conn", lambda url: FakeConn())
    monkeypatch.setattr(main.pipeline, "grounded_answer",
                        lambda q, *, conn, client=None: answer)
    r = c.post("/ask", json={"question": "q?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"question", "answer", "grounded", "citations", "passages"}
    assert body["grounded"] is True


def test_empty_question_is_422(client):
    c, _ = client
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.post("/ask", json={}).status_code == 422


def test_db_down_is_503_with_readable_message(client, monkeypatch):
    c, main = client

    def boom(url):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(main.store, "get_conn", boom)
    r = c.post("/ask", json={"question": "q?"})
    assert r.status_code == 503
    assert "docker compose" in r.json()["detail"]


def test_gemini_error_is_friendly_502(client, monkeypatch):
    c, main = client
    monkeypatch.setattr(main.store, "get_conn", lambda url: FakeConn())

    def boom(q, *, conn, client=None):
        raise GenerationError("Gemini call failed (QuotaError)")

    monkeypatch.setattr(main.pipeline, "grounded_answer", boom)
    r = c.post("/ask", json={"question": "q?"})
    assert r.status_code == 502
    assert "Traceback" not in r.text


def test_replay_mode_serves_fixture(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # replay needs no key
    monkeypatch.setenv("REPLAY", "1")
    fixture = Answer(question="What is the minimum internal temperature for poultry?",
                     answer="165°F (74°C) for 15 seconds.", grounded=True)
    (tmp_path / "poultry.json").write_text(fixture.model_dump_json(), encoding="utf-8")
    from app import main
    monkeypatch.setattr(main, "FIXTURES_DIR", tmp_path)
    with TestClient(main.app) as c:
        r = c.post("/ask", json={
            "question": "What is the minimum internal temperature for poultry?"})
        assert r.status_code == 200
        assert r.json()["answer"] == "165°F (74°C) for 15 seconds."
        # unknown question in replay mode -> graceful grounded=false, not an error
        r2 = c.post("/ask", json={"question": "Something never captured?"})
        assert r2.status_code == 200
        assert r2.json()["grounded"] is False


def test_startup_fails_without_key_outside_replay(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("REPLAY", raising=False)
    from app import main
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        with TestClient(main.app):
            pass


def test_index_serves_branded_page(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "Food-Safety Compliance Assistant" in r.text
    assert "#024731" in r.text  # UH Green - PMCS branding present


def _parse_sse(text):
    """Parse SSE text into a list of (event_type, json_data) tuples."""
    import json as _json
    events = []
    etype, data = None, None
    for line in text.splitlines():
        if line.startswith("event: "):
            etype = line[len("event: "):]
        elif line.startswith("data: "):
            data = _json.loads(line[len("data: "):])
        elif line == "":
            if etype is not None:
                events.append((etype, data))
            etype, data = None, None
    return events


def test_ask_stream_live_emits_stage_then_answer(client, monkeypatch):
    c, main = client
    monkeypatch.setattr(main.store, "get_conn", lambda url: FakeConn())

    def fake_events(question, *, conn, client=None):
        yield {"type": "stage", "step": "embed", "status": "done", "detail": "384-dim"}
        yield {"type": "token", "text": "165°F."}
        yield {"type": "answer", "question": question, "answer": "165°F.",
               "grounded": True, "citations": [], "passages": []}

    monkeypatch.setattr(main.stream, "stream_events", fake_events)
    r = c.post("/ask/stream", json={"question": "q?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    types = [t for t, _ in events]
    assert types[0] == "stage" and "answer" in types and types[-1] == "done"
    answer = next(d for t, d in events if t == "answer")
    assert set(answer) == {"question", "answer", "grounded", "citations", "passages"}


def test_ask_stream_empty_question_is_422(client):
    c, _ = client
    assert c.post("/ask/stream", json={"question": ""}).status_code == 422


def test_ask_stream_db_down_emits_error_frame(client, monkeypatch):
    c, main = client

    def boom(url):
        import psycopg
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(main.store, "get_conn", boom)
    r = c.post("/ask/stream", json={"question": "q?"})
    assert r.status_code == 200                      # stream already opened
    events = _parse_sse(r.text)
    assert any(t == "error" and "docker compose" in d["detail"] for t, d in events)
    assert events[-1][0] == "done"


def test_ask_stream_unexpected_error_emits_friendly_error(client, monkeypatch):
    c, main = client
    monkeypatch.setattr(main.store, "get_conn", lambda url: FakeConn())

    def boom(q, *, conn, client=None):
        raise ValueError("kaboom")
        yield  # make it a generator

    monkeypatch.setattr(main.stream, "stream_events", boom)
    r = c.post("/ask/stream", json={"question": "q?"})
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert any(t == "error" and "try again" in d["detail"].lower() for t, d in events)
    assert events[-1][0] == "done"
    assert "Traceback" not in r.text


def test_index_has_streaming_trace_ui(client):
    c, _ = client
    html = c.get("/").text
    assert "/ask/stream" in html          # UI targets the streaming endpoint
    assert 'id="trace"' in html           # trace panel container present
    assert "getReader" in html            # reads the stream incrementally


def test_ask_stream_replay_streams_fixture(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("REPLAY", "1")
    fixture = Answer(
        question="What is the minimum internal temperature for poultry?",
        answer="165°F (74°C) for 15 seconds.", grounded=True)
    (tmp_path / "poultry.json").write_text(fixture.model_dump_json(), encoding="utf-8")
    from app import main
    monkeypatch.setattr(main, "FIXTURES_DIR", tmp_path)
    with TestClient(main.app) as c:
        r = c.post("/ask/stream", json={
            "question": "What is the minimum internal temperature for poultry?"})
        assert r.status_code == 200
        events = _parse_sse(r.text)
        types = [t for t, _ in events]
        assert "stage" in types and "token" in types
        answer = next(d for t, d in events if t == "answer")
        assert answer["answer"] == "165°F (74°C) for 15 seconds."
        assert answer["grounded"] is True
        assert types[-1] == "done"
        # unknown question -> graceful declined answer, still valid stream
        r2 = c.post("/ask/stream", json={"question": "never captured?"})
        a2 = next(d for t, d in _parse_sse(r2.text) if t == "answer")
        assert a2["grounded"] is False

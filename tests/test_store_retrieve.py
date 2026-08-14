from foodsafety_rag import store
from foodsafety_rag.embed import embed_text

# Fixed 3-doc fixture corpus (deterministic retrieval target)
FIXTURE_DOCS = [
    ("fda", "Cooking Temperatures", "fda/cook.md", "Poultry",
     "Poultry\nRaw poultry must reach 165°F (74°C) for 15 seconds."),
    ("fda", "Holding", "fda/hold.md", "Cold holding",
     "Cold holding\nTCS foods held cold must stay at 41°F (5°C) or below."),
    ("sop", "Handwashing", "sop/wash.md", "How to wash",
     "How to wash\nWash hands with soap and warm water for at least 20 seconds."),
]


def _seed(conn):
    store.reset_index(conn)
    for source, title, path, heading, text in FIXTURE_DOCS:
        doc_id = store.upsert_document(conn, source, title, path)
        store.insert_chunk(conn, doc_id, heading, text, embed_text(text))


def test_schema_tables_exist(db_conn):
    rows = db_conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert {"documents", "chunks", "query_log"} <= names


def test_upsert_document_is_idempotent(db_conn):
    store.reset_index(db_conn)
    id1 = store.upsert_document(db_conn, "fda", "Cooking", "fda/cook.md")
    id2 = store.upsert_document(db_conn, "fda", "Cooking v2", "fda/cook.md")
    assert id1 == id2
    count = db_conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    assert count == 1


def test_similarity_search_returns_scored_rows(db_conn):
    _seed(db_conn)
    vec = embed_text("What temperature must chicken be cooked to?")
    rows = store.similarity_search(db_conn, vec, top_k=3)
    assert len(rows) == 3
    assert set(rows[0]) == {"doc", "heading", "text", "score"}
    assert rows[0]["heading"] == "Poultry"          # best match first
    assert rows[0]["score"] > rows[-1]["score"]     # descending
    assert 0.0 < rows[0]["score"] <= 1.0


def test_log_query_inserts_row(db_conn):
    before = db_conn.execute("SELECT count(*) FROM query_log").fetchone()[0]
    store.log_query(db_conn, "test question?")
    after = db_conn.execute("SELECT count(*) FROM query_log").fetchone()[0]
    assert after == before + 1


def test_log_outcome_records_what_the_answer_cost(db_conn):
    from foodsafety_rag.schemas import Usage

    query_id = store.log_query(db_conn, "what did this cost?")
    store.log_outcome(db_conn, query_id, grounded=True,
                      usage=Usage(prompt_tokens=812, completion_tokens=64,
                                  latency_ms=1250))
    row = db_conn.execute(
        "SELECT grounded, prompt_tokens, completion_tokens, latency_ms "
        "FROM query_log WHERE id = %s", (query_id,)).fetchone()
    assert row == (True, 812, 64, 1250)


def test_declined_question_logs_no_token_cost(db_conn):
    """The guard stops a declined question before any model call, so the row
    records the decline with no tokens spent."""
    query_id = store.log_query(db_conn, "can I bring my dog in?")
    store.log_outcome(db_conn, query_id, grounded=False)
    row = db_conn.execute(
        "SELECT grounded, prompt_tokens, latency_ms FROM query_log WHERE id = %s",
        (query_id,)).fetchone()
    assert row == (False, None, None)


from foodsafety_rag.config import SIMILARITY_THRESHOLD
from foodsafety_rag.retrieve import retrieve
from foodsafety_rag.schemas import Passage


def test_retrieve_returns_passages_best_first(db_conn):
    _seed(db_conn)
    passages = retrieve("What temperature must chicken be cooked to?",
                        conn=db_conn, top_k=2)
    assert len(passages) == 2
    assert all(isinstance(p, Passage) for p in passages)
    assert passages[0].heading == "Poultry"
    assert passages[0].score >= SIMILARITY_THRESHOLD  # on-corpus question clears guard


def test_off_corpus_question_falls_below_threshold(db_conn):
    _seed(db_conn)
    passages = retrieve("How do I renew my car registration in Hawaii?",
                        conn=db_conn, top_k=2)
    assert passages[0].score < SIMILARITY_THRESHOLD  # grounding guard will trip

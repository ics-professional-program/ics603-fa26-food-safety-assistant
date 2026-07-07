"""Postgres + pgvector storage (course modules M7 + M8).

One database holds relational rows AND vectors: documents, chunks
(embedding vector(384)), and query_log.
"""

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from foodsafety_rag.config import EMBED_DIM

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id     serial PRIMARY KEY,
    source text NOT NULL,
    title  text NOT NULL,
    path   text UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          serial PRIMARY KEY,
    document_id integer NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    heading     text NOT NULL,
    text        text NOT NULL,
    embedding   vector({EMBED_DIM}) NOT NULL
);

CREATE TABLE IF NOT EXISTS query_log (
    id         serial PRIMARY KEY,
    question   text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
"""


def get_conn(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url, autocommit=True)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        pass  # vector extension not installed yet; init_schema() installs it
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_SQL)
    register_vector(conn)


def reset_index(conn: psycopg.Connection) -> None:
    """Idempotent rebuild support: clear documents and chunks (keeps query_log)."""
    conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")


def upsert_document(conn: psycopg.Connection, source: str, title: str, path: str) -> int:
    row = conn.execute(
        """
        INSERT INTO documents (source, title, path) VALUES (%s, %s, %s)
        ON CONFLICT (path) DO UPDATE SET source = EXCLUDED.source
        RETURNING id
        """,
        (source, title, path),
    ).fetchone()
    return row[0]


def insert_chunk(conn: psycopg.Connection, document_id: int, heading: str,
                 text: str, embedding: list[float]) -> None:
    conn.execute(
        "INSERT INTO chunks (document_id, heading, text, embedding) "
        "VALUES (%s, %s, %s, %s)",
        (document_id, heading, text, Vector(embedding)),
    )


def similarity_search(conn: psycopg.Connection, embedding: list[float],
                      top_k: int) -> list[dict]:
    """Cosine similarity search. score = 1 - cosine distance (higher = closer)."""
    vec = Vector(embedding)
    rows = conn.execute(
        """
        SELECT d.title, c.heading, c.text, 1 - (c.embedding <=> %s) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> %s
        LIMIT %s
        """,
        (vec, vec, top_k),
    ).fetchall()
    return [
        {"doc": r[0], "heading": r[1], "text": r[2], "score": float(r[3])}
        for r in rows
    ]


def log_query(conn: psycopg.Connection, question: str) -> None:
    conn.execute("INSERT INTO query_log (question) VALUES (%s)", (question,))

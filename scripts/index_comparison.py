"""The 11.1 before/after index comparison, with recorded numbers.

Measures the same query against the same table twice: first with NO vector
index (pgvector falls back to an exact sequential scan), then after CREATE
INDEX ... USING hnsw. Reports median latency for both, the index build time,
and recall@4 - how many of the exact top-4 the approximate index returns.
Run against the FULL corpus (ingest without --skip-bulk), or there is
nothing to measure.

Usage:
    uv run python scripts/index_comparison.py
    uv run python scripts/index_comparison.py --write docs/retrieval-reference.md
"""

import argparse
import statistics
import time
from pathlib import Path

from foodsafety_rag import embed, store
from foodsafety_rag.config import TOP_K, get_settings

QUESTIONS = [
    "What is the minimum internal temperature for cooking poultry?",
    "How hot does the water at a handwashing sink need to be?",
    "What temperature must hot food be held at?",
    "How do I calibrate a probe thermometer?",
]
INDEX_NAME = "chunks_embedding_hnsw"
INDEX_SQL = (f"CREATE INDEX {INDEX_NAME} ON chunks "
             "USING hnsw (embedding vector_cosine_ops)")
REPS = 10
DEPTH = 10          # ids compared at depth 10; recall reported at TOP_K


def _search(conn, vec, limit):
    from pgvector import Vector
    v = Vector(vec)
    return conn.execute(
        """
        SELECT c.id, d.title, c.heading
        FROM chunks c JOIN documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> %s
        LIMIT %s
        """,
        (v, limit),
    ).fetchall()


def _measure(conn, vec):
    """Median latency over REPS runs (after 2 warmups) and the result rows."""
    for _ in range(2):
        _search(conn, vec, DEPTH)
    times = []
    rows = None
    for _ in range(REPS):
        start = time.perf_counter()
        rows = _search(conn, vec, DEPTH)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path,
                        help="also write a markdown report to this path")
    args = parser.parse_args()

    settings = get_settings()
    with store.get_conn(settings.database_url) as conn:
        n_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(f"{n_chunks} chunks in the table")
        conn.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
        conn.execute("ANALYZE chunks")

        vecs = {q: embed.embed_text(q) for q in QUESTIONS}

        exact = {}
        for q in QUESTIONS:
            ms, rows = _measure(conn, vecs[q])
            exact[q] = (ms, rows)

        t0 = time.perf_counter()
        conn.execute(INDEX_SQL)
        build_s = time.perf_counter() - t0
        print(f"index build: {build_s:.2f}s")

        hnsw = {}
        for q in QUESTIONS:
            ms, rows = _measure(conn, vecs[q])
            hnsw[q] = (ms, rows)

    lines = [
        "# Retrieval reference numbers (session 11.1)",
        "",
        f"Measured over the full corpus: **{n_chunks} chunks** "
        f"(curated + generated Food Code chapters). Exact scan = no vector "
        f"index; HNSW = `{INDEX_SQL}`. Median of {REPS} runs each; "
        f"index build took **{build_s:.2f}s**.",
        "",
        "| Question | exact (ms) | HNSW (ms) | speedup | recall@4 |",
        "|---|---|---|---|---|",
    ]
    print(f"\n{'question':58} {'exact':>9} {'hnsw':>9} {'recall@4':>9}")
    for q in QUESTIONS:
        e_ms, e_rows = exact[q]
        h_ms, h_rows = hnsw[q]
        e_top = {r[0] for r in e_rows[:TOP_K]}
        h_top = {r[0] for r in h_rows[:TOP_K]}
        recall = len(e_top & h_top) / TOP_K
        print(f"{q[:58]:58} {e_ms:>7.2f}ms {h_ms:>7.2f}ms {recall:>9.2f}")
        lines.append(f"| {q} | {e_ms:.2f} | {h_ms:.2f} "
                     f"| {e_ms / h_ms:.1f}x | {recall:.2f} |")

    lines += ["", "## Expected top-4 (the Predict step)", "",
              "Exact-scan results - what students should predict before "
              "running the query:", ""]
    for q in QUESTIONS:
        lines.append(f"**{q}**")
        lines.append("")
        for i, (_, title, heading) in enumerate(exact[q][1][:TOP_K], start=1):
            lines.append(f"{i}. {title} — {heading}")
        lines.append("")

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nwrote {args.write}")


if __name__ == "__main__":
    main()

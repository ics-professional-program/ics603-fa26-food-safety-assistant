"""Build the retrieval index: corpus/*.md -> chunks -> embeddings -> Postgres.

Idempotent: each run resets documents/chunks and rebuilds cleanly.
Run:  python scripts/ingest_corpus.py
"""

from pathlib import Path

from foodsafety_rag import embed, ingest, store
from foodsafety_rag.config import get_settings

CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def main() -> None:
    settings = get_settings()
    chunks = ingest.load_corpus(CORPUS_DIR)
    print(f"Chunked {len(chunks)} passages; embedding locally (no API key needed)...")
    vectors = embed.embed_texts([c.text for c in chunks])

    with store.get_conn(settings.database_url) as conn:
        store.init_schema(conn)
        store.reset_index(conn)
        doc_ids: dict[str, int] = {}
        for chunk, vector in zip(chunks, vectors):
            if chunk.path not in doc_ids:
                doc_ids[chunk.path] = store.upsert_document(
                    conn, chunk.source, chunk.title, chunk.path)
            store.insert_chunk(conn, doc_ids[chunk.path],
                               chunk.heading, chunk.text, vector)
    print(f"Indexed {len(chunks)} chunks from {len(doc_ids)} documents.")


if __name__ == "__main__":
    main()

"""Retrieval (course module M8): question -> top-k grounded passages."""

from foodsafety_rag import embed, store
from foodsafety_rag.config import TOP_K
from foodsafety_rag.schemas import Passage


def retrieve(question: str, *, conn, top_k: int = TOP_K) -> list[Passage]:
    vec = embed.embed_text(question)
    return [Passage(**row) for row in store.similarity_search(conn, vec, top_k)]

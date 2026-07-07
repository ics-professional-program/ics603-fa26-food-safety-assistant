"""Local sentence-transformers embeddings (course module M8). No API key needed."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from foodsafety_rag.config import EMBED_MODEL_NAME


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = _model().encode(texts, normalize_embeddings=True)
    return [[float(x) for x in v] for v in vectors]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]

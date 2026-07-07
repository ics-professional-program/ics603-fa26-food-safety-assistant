import math

from foodsafety_rag.config import EMBED_DIM
from foodsafety_rag.embed import embed_text, embed_texts


def test_embedding_shape():
    vecs = embed_texts(["poultry must reach 165F", "wash your hands"])
    assert len(vecs) == 2
    assert all(len(v) == EMBED_DIM for v in vecs)
    assert all(isinstance(x, float) for x in vecs[0])


def test_embedding_deterministic():
    a = embed_text("minimum internal temperature for poultry")
    b = embed_text("minimum internal temperature for poultry")
    assert a == b


def test_embedding_normalized():
    v = embed_text("cold holding temperature")
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-3  # normalized => cosine == dot product

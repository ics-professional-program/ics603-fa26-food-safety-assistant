FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /srv

# CPU-only torch keeps the image far smaller than the CUDA default
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY foodsafety_rag ./foodsafety_rag
RUN pip install .

# Bake the embedding model into the image so first request needs no download
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
COPY corpus ./corpus
COPY scripts ./scripts
COPY fixtures ./fixtures

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

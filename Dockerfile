FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# The uv binary comes from the image its authors publish, so the build does not
# install it separately. This is a complete version rather than a minor-version
# tag such as 0.12, so it keeps identifying the same image contents. Re-verify
# it before the term.
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

WORKDIR /srv

# Dependencies are installed before the application code is copied, so editing
# a source file does not invalidate this layer. --no-install-project installs
# the dependencies without the project itself, which is not present yet.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# Bake the embedding model into the image so the first request needs no
# download. The project is not installed at this point, so this calls the
# environment's interpreter directly rather than through `uv run`.
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY README.md ./
COPY foodsafety_rag ./foodsafety_rag
COPY app ./app
COPY corpus ./corpus
COPY scripts ./scripts
COPY fixtures ./fixtures
RUN uv sync --locked --no-dev

# The environment is already complete and correct at this point, so `uv run`
# should start the server rather than re-checking the lock file on every
# container start. Without this it recompiles bytecode at each boot.
ENV UV_NO_SYNC=1

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ICS 603 — Food-Safety Compliance Assistant

A Retrieval-Augmented Generation (RAG) grounded-Q&A demo app for ICS 603
(Fall 2026). Staff ask plain-language food-safety questions; the app answers
**only** from a trusted corpus (FDA Food Code-style excerpts + synthetic
Pacific Market Cafe SOPs) and shows the source passages it used. When the
corpus doesn't cover a question, it declines instead of guessing.

This is the course's reference architecture: the capstone project is your own
version of this same pattern. It is also the fallback application for sessions
that need a running app before your capstone is finished — 11.1 (add a vector
column to a real schema), 12.0 (deploy a container stack), and 13.0 (evaluate an
LLM app).

## The codebase IS the module map

| File | Course module | What it teaches |
|---|---|---|
| `foodsafety_rag/schemas.py` | M4 / M8 | Pydantic models — structure over guesswork |
| `foodsafety_rag/generate.py` | M4 | Calling an LLM API (Gemini) with structured output |
| `app/main.py` + `app/static/` | M4 | FastAPI service + minimal UI |
| `tests/` | M6 | Automated tests as the check on generated code |
| `foodsafety_rag/store.py` | M9 + M11 | SQL + vectors in ONE Postgres (pgvector) |
| `foodsafety_rag/embed.py` | M11 | Local sentence-transformer embeddings |
| `foodsafety_rag/ingest.py`, `retrieve.py` | M11 | Chunking and top-k retrieval |
| `foodsafety_rag/pipeline.py` | M11 | The grounding guard — decline, don't guess |
| `Dockerfile`, `docker-compose.yml` | M10 + M12 | Containers locally, then Jetstream |
| `notebooks/first_llm_call.ipynb` | 1.3 | Your first LLM call |

Module numbers follow the Fall 2026 teaching order: M4 LLM APIs, M6 software
engineering, M9 databases, M10 containers, M11 embeddings/RAG, M12 deployment.

## Quick start (Docker)

```bash
cp .env.example .env          # add your GEMINI_API_KEY (aistudio.google.com/apikey)
docker compose up -d --build
docker compose exec app python scripts/ingest_corpus.py
# open http://localhost:8000
```

If host port 8000 is already in use, set `APP_PORT`, e.g.
`APP_PORT=8020 docker compose up -d --build` and open http://localhost:8020.

## Local development (uv)

```bash
uv venv                                    # create .venv
uv pip install -e ".[dev]"                 # install package + dev tools
docker compose up -d db                    # Postgres + pgvector
set -a; . ./.env; set +a                   # load GEMINI_API_KEY into the shell
.venv/bin/python scripts/ingest_corpus.py  # build the index
.venv/bin/python -m uvicorn app.main:app --port 8000
```

Tests: `.venv/bin/python -m pytest` (Gemini is always mocked; store/retrieve
tests skip if the db container is down).

## Class-demo fallback (no key, no network)

Set `REPLAY=1` and start the app: `/ask` serves captured answers from
`fixtures/`. The notebook has an equivalent captured-response fallback cell.

## Live pipeline view

`POST /ask/stream` returns Server-Sent Events so the web UI can show the
pipeline running live — each stage (embed → retrieve → grounding guard →
generate) streams in as it happens, and the answer fills token-by-token.
`POST /ask` remains the simple one-shot JSON API (same `Answer` contract) for
scripts, tests, and the contrast demo. Replay mode (`REPLAY=1`) streams from
captured fixtures, so the live view still works with no key and no DB.

## Useful scripts

- `scripts/ingest_corpus.py` — (re)build the index. Idempotent.
- `scripts/contrast.py "your question"` — plain Gemini vs grounded pipeline.
- `scripts/capture_fixtures.py` — refresh the replay fixtures (set `ASK_URL`
  if the app is not on port 8000).

## Branding

The web UI is styled with the UH Mānoa PMCS design system, vendored as the
`skills/design-system` submodule
([`skill.design-system`](https://github.com/ics-professional-program/skill.design-system)).
Its brand tokens (colors, Helvetica Neue type, spacing, card/badge patterns)
are inlined into `app/static/index.html` — no build step, no external CDN.

## Notes

- Embeddings are local (`all-MiniLM-L6-v2`, 384-dim) — only generation needs
  the `GEMINI_API_KEY`. The key lives in `.env` (gitignored), never in code.
- Deployment target: Jetstream (Oct 27–29 course block). This repo is
  container-ready; no live deployment yet.

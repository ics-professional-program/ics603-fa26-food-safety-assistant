# ICS 603 — Food-Safety Compliance Assistant

A Retrieval-Augmented Generation (RAG) grounded-Q&A demo app for ICS 603
(Fall 2026). Staff ask plain-language food-safety questions; the app answers
**only** from a trusted corpus and shows the source passages it used. When
the corpus doesn't cover a question, it declines instead of guessing.

The corpus has two tiers (see [`corpus/README.md`](corpus/README.md) for
provenance): a curated core adapted from the real FDA Food Code (2022 and
2017 editions), Hawaii's HAR 11-50 food-safety rules, and fictional Pacific
Market Cafe SOPs — including three real, citable disagreements between
sources — plus `corpus/bulk/`, chapters 2-8 of both Food Code editions
converted mechanically (~1,400 passages) so the 11.1 index comparison has
enough rows to measure. `scripts/ingest_corpus.py --skip-bulk` ingests just
the curated core, which is the fast path for the small-corpus demos.

This is the course's reference architecture: the capstone project is your own
version of this same pattern. It is also the fallback application for sessions
that need a running app before your capstone is finished — 11.1 (add a vector
column to a real schema), 12.0 (deploy a container stack), and 13.0 (evaluate an
LLM app).

## The codebase IS the module map

| File | Course module | What it teaches |
|---|---|---|
| `foodsafety_rag/schemas.py` | M4 / M8 | Pydantic models — structure over guesswork |
| `foodsafety_rag/agent.py` | M4 / M6 | Typed Pydantic AI agent, provider selection, citation validation, and bounded retry |
| `foodsafety_rag/generate.py` | M4 | Typed generation plus intentionally low-level OpenAI-compatible raw and streaming calls |
| `app/main.py` + `app/static/` | M4 | FastAPI service + minimal UI |
| `tests/` | M6 | Automated tests as the check on generated code |
| `foodsafety_rag/store.py` | M9 + M11 | SQL + vectors in ONE Postgres (pgvector) |
| `query_log` table | M9 + M13 | What each question cost: grounded, tokens, latency |
| `foodsafety_rag/embed.py` | M11 | Local sentence-transformer embeddings |
| `foodsafety_rag/ingest.py`, `retrieve.py` | M11 | Chunking and top-k retrieval |
| `foodsafety_rag/pipeline.py` | M11 | The grounding guard — decline, don't guess |
| `Dockerfile`, `docker-compose.yml` | M10 + M12 | Containers locally, then Jetstream |
| `notebooks/first_llm_call.ipynb` | 1.3 | Your first LLM call |

Module numbers follow the Fall 2026 teaching order: M4 LLM APIs, M6 software
engineering, M9 databases, M10 containers, M11 embeddings/RAG, M12 deployment.

## Quick start (Docker)

```bash
cp .env.example .env          # select the provider, model, endpoint, and credential
docker compose up -d --build
docker compose exec app python scripts/ingest_corpus.py
# open http://localhost:8000
```

If host port 8000 is already in use, set `APP_PORT`, e.g.
`APP_PORT=8020 docker compose up -d --build` and open http://localhost:8020.

## The LLM provider

The structured `/ask` path uses Pydantic AI. Set its provider and model in `.env`:

| Variable | What it is |
|---|---|
| `LLM_PROVIDER` | `course`, `openai`, `anthropic`, or `google-cloud` |
| `LLM_BASE_URL` | The API base, e.g. `https://llm.jetstream-cloud.org/api` |
| `LLM_MODEL` | The model id as that server lists it, e.g. `gpt-oss-120b` |
| `LLM_API_KEY` | The course/OpenAI-compatible endpoint key (omit only for localhost) |

The course endpoint is Jetstream's Open WebUI, whose OpenAI-compatible API lives
under `/api` (so `/api/chat/completions`, `/api/models`). The same code runs
against a local model server (LM Studio, Ollama, vLLM) or api.openai.com by
using `LLM_PROVIDER=course` and changing the endpoint settings —
`GET {LLM_BASE_URL}/models` lists what a given server will accept as `LLM_MODEL`.
The optional OpenAI and Anthropic adapters use their normal `OPENAI_API_KEY` and
`ANTHROPIC_API_KEY`; Google Cloud uses its documented application-default or
service-account credentials.

The early `ask_model()` notebook and the true token-streaming implementation
intentionally retain the raw OpenAI SDK so students can compare the lower-level
protocol with the typed agent. With `LLM_PROVIDER=course`, `/ask/stream` uses that
raw streaming path. With an optional provider, it uses the typed Pydantic AI path,
validates the complete answer, and then sends word-sized UI chunks. `/ask` always
uses the typed provider-independent path.

Provider configuration is read when the application process imports the shared
agent. Restart Uvicorn or the container after changing `.env`; changing variables
inside an already-running process does not replace that agent's model.

Note for the Docker path: `localhost` inside the app container is the container
itself, not your laptop. A model server running on your machine is reachable as
`http://host.docker.internal:1234/v1`.

## Local development (uv)

```bash
uv sync                                    # create .venv from uv.lock, with dev tools
docker compose up -d db                    # Postgres + pgvector
set -a; . ./.env; set +a                   # load the LLM settings into the shell
uv run python scripts/ingest_corpus.py     # build the index
uv run uvicorn app.main:app --port 8000
```

Tests: `uv run pytest` (the LLM is always mocked; store/retrieve tests skip if
the db container is down).

The demo's `/ask` route is deliberately a regular `def`: FastAPI runs it in a
threadpool, where the synchronous retrieval pipeline and `agent.run_sync()` are
safe. The smaller 4.3 starter instead teaches an `async def` route that awaits
`agent.run()`. Do not call `run_sync()` from inside an async route.

`uv sync` installs the exact versions recorded in `uv.lock`, which is the same
thing the Dockerfile does with `uv sync --locked`. If you change a dependency in
`pyproject.toml`, run `uv lock` and commit the updated lock file, or the image
build will fail — that failure is deliberate, and it means the lock file needs
updating.

On Linux, `torch` is installed from PyTorch's CPU-only index rather than PyPI;
see the comment in `pyproject.toml`. The default Linux wheels carry CUDA and are
several gigabytes, and this app only runs the embedding model on CPU.

## Class-demo fallback (no key, no network)

Set `REPLAY=1` and start the app: `/ask` serves captured answers from
`fixtures/`. The notebook has an equivalent captured-response fallback cell.

## What a question costs

Every question is logged to `query_log` with whether it was grounded, the prompt
and completion token counts, and the generation latency. A declined question
records `grounded = false` with no tokens at all — the guard stops it before any
model call, so declining is free. The web UI shows the same token counts beside
the latency in the generate stage.

```sql
SELECT grounded, count(*), avg(latency_ms)::int AS ms,
       sum(prompt_tokens + completion_tokens) AS tokens
FROM query_log WHERE prompt_tokens IS NOT NULL GROUP BY grounded;
```

Over a term this accumulates into real measurements for 13.0 (evaluating LLM
applications) rather than a hypothetical cost model.

## Live pipeline view

`POST /ask/stream` returns Server-Sent Events so the web UI can show the
pipeline running live — each stage (embed → retrieve → grounding guard →
generate) streams in as it happens, and the answer fills token-by-token.
`POST /ask` remains the simple one-shot JSON API (same `Answer` contract) for
scripts, tests, and the contrast demo. Replay mode (`REPLAY=1`) streams from
captured fixtures, so the live view still works with no key and no DB.

## Useful scripts

- `scripts/ingest_corpus.py` — (re)build the index. Idempotent, and it also
  applies schema changes, so run it once after pulling a version that adds
  columns. `--skip-bulk` ingests the curated core only.
- `scripts/contrast.py "your question"` — plain model call vs grounded pipeline.
- `scripts/capture_fixtures.py` — refresh the replay fixtures (set `ASK_URL`
  if the app is not on port 8000).
- `scripts/convert_food_code.py` — regenerate `corpus/bulk/` from the
  official Food Code PDFs (the generated markdown is committed).
- `scripts/index_comparison.py` — the 11.1 before/after HNSW measurement;
  recorded numbers live in
  [`docs/retrieval-reference.md`](docs/retrieval-reference.md).
- `scripts/run_eval.py --label <name>` — run the labeled eval set
  ([`evals/`](evals/)); retrieval and generation scored separately.

## Branding

The web UI is styled with the UH Mānoa PMCS design system, vendored as the
`skills/design-system` submodule
([`skill.design-system`](https://github.com/ics-professional-program/skill.design-system)).
Its brand tokens (colors, Helvetica Neue type, spacing, card/badge patterns)
are inlined into `app/static/index.html` — no build step, no external CDN.

## Notes

- Embeddings are local (`all-MiniLM-L6-v2`, 384-dim) — only generation needs
  the endpoint. The key lives in `.env` (gitignored), never in code.
- Deployment target: Jetstream (Oct 27–29 course block). This repo is
  container-ready; no live deployment yet.

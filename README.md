# ICS 603 — Food-Safety Compliance Assistant

A Retrieval-Augmented Generation (RAG) grounded-Q&A demo app for ICS 603
(Fall 2026). Staff ask plain-language food-safety questions; the app answers
**only** from a trusted corpus and shows the source passages it used. When
the corpus doesn't cover a question, it declines instead of guessing.

The corpus has two tiers (see [`corpus/README.md`](corpus/README.md) for
provenance): a curated core adapted from the real FDA Food Code (2022 and
2017 editions), Hawaii's HAR 11-50 food-safety rules, and Pacific Market Cafe
SOPs written on the Institute of Child Nutrition's real HACCP-based SOP
templates — the cafe is fictional, its procedures are not invented — including
three real, citable disagreements between sources — plus `corpus/bulk/`, chapters 2-8 of both Food Code editions
converted mechanically (~1,400 passages).

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

## Local development (uv)

```bash
uv sync                                    # create .venv from uv.lock, with dev tools
docker compose up -d db                    # Postgres + pgvector
set -a; . ./.env; set +a                   # load the LLM settings into the shell
uv run python scripts/ingest_corpus.py     # build the index
uv run uvicorn app.main:app --port 8000
```

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

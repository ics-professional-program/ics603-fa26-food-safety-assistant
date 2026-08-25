# Evaluation set and harness

The labeled eval set for session 13.0: thirteen questions in
[`eval_set.json`](eval_set.json), each with a reference answer, the labeled
source chunk (or `null` for off-corpus questions), and whether the corpus can
answer it at all. The scoring code is
[`foodsafety_rag/evaluate.py`](../foodsafety_rag/evaluate.py); the runner is
[`scripts/run_eval.py`](../scripts/run_eval.py).

## What gets scored, and separately

- **Retrieval** — whether the labeled chunk appears in the top-k the app
  actually retrieves (hit@4, with its rank). Scored independently of the
  answer text.
- **Generation, deterministic** — every temperature/time value extracted from
  the reference answer must also appear in the answer, after unit
  normalization (F, C, sec, min, hr). Missing required values fail; extra
  values are reported but do not fail. For `answerable: false` cases the rule
  is inverted: the pipeline must **decline** — declining is the correct
  behavior, not a failure.
- **Generation, LLM-as-judge** — a fixed four-axis rubric (faithful to the
  passages, answers the question, cites sources, handles conflicts), each
  axis 0-2, run through the app's own provider selection.

The set is built so retrieval and generation can disagree, in both
directions. `roast-alternative` retrieves its passage cleanly but asks
generation to reproduce a table of six time/temperature pairs, which is
exactly where an LLM drops or garbles a row. `handwash-water-temp` is the
mirror: the labeled Hawaii chunk ranks ~8th behind four near-duplicate FDA
handwash chunks, so retrieval misses at k=4 while generation still writes a
plausible federal-only answer.

## Running it

```bash
docker compose up -d db
uv run python scripts/ingest_corpus.py          # full corpus, incl. bulk/
uv run python scripts/run_eval.py --label baseline
uv run python scripts/run_eval.py --label quick --skip-judge   # no LLM judge
```

Results land in `evals/results/<label>.json` (git-ignored). The table printed
at the end shows retrieval and generation as separate columns; the summary
line names any retrieval-good/generation-bad rows.

## Recorded runs

Run 2026-08-24 against the course endpoint (`gpt-oss-120b` behind
`llm.jetstream-cloud.org`), full corpus (1,462 chunks) ingested.

**Baseline** (`--label baseline`): retrieval 7/10 hits, deterministic 11/13
pass, judge mean 6.8/8. The instructive rows:

- `handwash-water-temp` — retrieval **MISS** (the labeled Hawaii chunk ranks
  8th behind four near-duplicate FDA chunks), deterministic **FAIL** (the
  answer gave only the 2022 edition's 85°F and omitted 2017's 100°F even
  though that chunk was retrieved), judge **3/8**. The conflict in the
  passages went unhandled.
- `dog-in-kitchen` and `handwash-duration` — retrieval MISS with a passing
  answer: generation worked from other relevant chunks. Retrieval and
  generation disagreeing in the other direction.
- `hot-holding-minimum` — ERROR: the endpoint failed the structured-output
  call twice. An eval also measures your infrastructure.

**The one-line change:** added to `GROUNDED_PROMPT` in
`foodsafety_rag/generate.py`:

> If the passages disagree with each other, present both values, name each
> source, and say which one governs.

**After** (`--label conflict-instruction`): `handwash-water-temp` went from
deterministic FAIL / judge 3 to **pass / judge 8** — the answer now presents
85°F and 100°F, names both editions, and says which governs. Judge mean rose
to 7.3/8. No case that completed in both runs regressed. (This run had four
ERROR rows — endpoint flakiness under sustained load, recorded as-is; the
per-case before/after on completed pairs is the signal.)

A separate finding from the first eval run: the citation validator's exact
string matching rejected legitimate citations whenever the model normalized
an em dash or completed a truncated title — 10 of 13 questions failed with
`Exceeded maximum output retries`. Fixed in `foodsafety_rag/agent.py` by
normalizing formatting variance before comparing (fabricated citations still
fail). The eval caught a real app bug before any student saw it.

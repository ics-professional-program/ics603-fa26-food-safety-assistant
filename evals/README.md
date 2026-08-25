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

<!-- Filled in by the before/after demonstration (Task 7): baseline scores,
the one-line prompt change, and the scores after it. -->

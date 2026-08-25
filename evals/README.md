# Evaluation set and harness

The labeled eval set for session 13.0: fourteen questions in
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

> The `baseline` and `conflict-instruction` numbers below predate the
> ICN-based rewrite of the `sop/` tier. They are kept because the pair is the
> point — the same cases before and after one prompt change — and that pair
> cannot be reproduced now without reverting the prompt. For current numbers
> see **icn-corpus** below.

**icn-corpus** (`--label icn-corpus`, run 2026-08-24 after the SOP rewrite,
1,478 chunks, 14 cases): retrieval 5/7 hits, deterministic 8/14 pass, judge
mean 6.9/8, with **5 rows ERROR** — `UnexpectedModelBehavior`, the endpoint
failing the structured-output call after its retries. Same flakiness the
`conflict-instruction` run hit; recorded as-is rather than re-rolled, because
an eval measures your infrastructure as well as your prompt.

The instructive row is the new one:

- `illness-return-to-work` — retrieval **hit @1**, deterministic **FAIL**: the
  answer gave the cafe's 48-hour house rule and omitted the Food Code's 24
  hours. It reads like a generation failure, and it is not. All four retrieved
  passages come from `sop/employee-illness.md`:

  ```
  0.754  SOP-03 - Symptoms requiring exclusion
  0.692  SOP-03 - Corrective action
  0.668  SOP-03 - Reportable diagnoses
  0.626  SOP-03 - Return to work
  ```

  `fda/employee-health-and-exclusions.md`, which carries the 24 hours, never
  reaches the context — one document consumed every slot at k=4, so the model
  could not have named both sides. The labeled chunk ranking first is exactly
  what hides this: hit@1 looks like healthy retrieval.

  This is the sharpest case in the set for 13.0's central distinction, and it
  is a real defect worth fixing rather than a case worth relabelling — the guard
  and the retriever are working as written, but four near-duplicate chunks from
  one document are not four sources. Per-document capping or a higher k would
  be the thing to measure next.

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

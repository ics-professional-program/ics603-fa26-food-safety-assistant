"""Run the labeled eval set against the running pipeline (session 13.0).

Retrieval and generation are always reported as separate columns, so a case
where the two disagree - right passage retrieved, poor answer written from
it - is visible by construction.

Requires: Postgres up (docker compose up -d db), the corpus ingested
(scripts/ingest_corpus.py), and an LLM configured in .env. Use --skip-judge
to run without the LLM-as-judge pass (the deterministic checks still run).

Usage:
    uv run python scripts/run_eval.py --label baseline
    uv run python scripts/run_eval.py --label after-prompt-fix --skip-judge
"""

import argparse
import json
from pathlib import Path

from foodsafety_rag import store
from foodsafety_rag.config import get_settings
from foodsafety_rag.evaluate import evaluate_question

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "evals" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True,
                        help="name for this run; results land in "
                             "evals/results/<label>.json")
    parser.add_argument("--skip-judge", action="store_true",
                        help="skip the LLM-as-judge pass")
    parser.add_argument("--eval-set", type=Path,
                        default=ROOT / "evals" / "eval_set.json")
    args = parser.parse_args()

    cases = json.loads(args.eval_set.read_text(encoding="utf-8"))
    settings = get_settings()
    results = []
    with store.get_conn(settings.database_url) as conn:
        for case in cases:
            result = evaluate_question(conn, case, judge=not args.skip_judge)
            results.append(result)
            print(f"  ran {result['id']}")

    header = (f"{'id':24} {'retrieval':>12} {'deterministic':>14} "
              f"{'judge':>6} {'grounded':>9}")
    print("\n" + header)
    print("-" * len(header))
    ret_hits = ret_total = det_passes = 0
    for r in results:
        if r["retrieval_hit"] is None:
            retrieval = "-"
        else:
            ret_total += 1
            ret_hits += r["retrieval_hit"]
            retrieval = (f"hit @{r['retrieval_rank']}" if r["retrieval_hit"]
                         else "MISS")
        det_passes += r["det_pass"]
        if r.get("error") and not r["answer"]:
            det = "ERROR"
        elif r["det_pass"]:
            det = "pass"
        else:
            det = "FAIL " + (",".join(r["det_missing"]) or "declined")
        judge = str(r["judge_total"]) if r["judge_total"] is not None else "-"
        print(f"{r['id']:24} {retrieval:>12} {det:>14} {judge:>6} "
              f"{str(r['grounded']):>9}")
    print("-" * len(header))
    judged = [r["judge_total"] for r in results if r["judge_total"] is not None]
    print(f"retrieval: {ret_hits}/{ret_total} hits    "
          f"generation (deterministic): {det_passes}/{len(results)} pass    "
          + (f"judge mean: {sum(judged)/len(judged):.1f}/8" if judged else ""))
    disagreeing = [r["id"] for r in results
                   if r["retrieval_hit"] and not r["det_pass"]]
    if disagreeing:
        print(f"retrieval-good / generation-bad cases: {', '.join(disagreeing)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "results": results},
                              indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Contrast demo: the SAME question through (a) a plain model call with no
context, and (b) the grounded RAG pipeline. Used for the course-launch
grounded-vs-ungrounded comparison.

Run:  python scripts/contrast.py "What is the minimum internal temperature for poultry?"
"""

import sys

from foodsafety_rag import store
from foodsafety_rag.config import get_settings
from foodsafety_rag.generate import ask_model
from foodsafety_rag.pipeline import grounded_answer

DEFAULT_QUESTION = "What is the minimum internal temperature for poultry?"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    bar = "=" * 72

    print(bar)
    print(f"QUESTION: {question}")
    print(bar)

    print("\n--- (a) Plain model call - no context, answers from model memory ---\n")
    print(ask_model(question))

    print("\n--- (b) Grounded pipeline - answers only from trusted documents ---\n")
    with store.get_conn(get_settings().database_url) as conn:
        answer = grounded_answer(question, conn=conn)
    print(f"grounded: {answer.grounded}")
    print(answer.answer)
    if answer.citations:
        print("\ncitations:")
        for c in answer.citations:
            print(f"  - {c.doc} — {c.heading}: {c.snippet}")
    print("\nretrieved passages (what the model was given):")
    for p in answer.passages:
        print(f"  [{p.score:.2f}] {p.doc} — {p.heading}")


if __name__ == "__main__":
    main()

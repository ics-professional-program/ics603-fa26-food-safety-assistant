"""The grounded pipeline: retrieve -> guard -> generate.

The grounding guard is the pedagogical heart of the app: when nothing
relevant is retrieved, we DECLINE (grounded=false) instead of letting the
model bluff. "Useful is not the same as correct."
"""

from foodsafety_rag import store
from foodsafety_rag.config import SIMILARITY_THRESHOLD
from foodsafety_rag.generate import answer_question
from foodsafety_rag.retrieve import retrieve
from foodsafety_rag.schemas import Answer

NOT_FOUND_ANSWER = (
    "I could not find an answer to this in the trusted food-safety documents "
    "(FDA Food Code excerpts and Pacific Market Cafe SOPs). Please check with "
    "your manager or the source documents directly."
)


def grounded_answer(question: str, *, conn, client=None) -> Answer:
    query_id = store.log_query(conn, question)
    passages = retrieve(question, conn=conn)
    if not passages or passages[0].score < SIMILARITY_THRESHOLD:
        store.log_outcome(conn, query_id, grounded=False)
        return Answer(
            question=question,
            answer=NOT_FOUND_ANSWER,
            grounded=False,
            citations=[],
            passages=passages,
        )
    answer = answer_question(question, passages, client=client)
    store.log_outcome(conn, query_id, grounded=answer.grounded, usage=answer.usage)
    return answer

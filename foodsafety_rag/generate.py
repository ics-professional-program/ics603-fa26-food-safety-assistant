"""Gemini generation (course module M4).

Two entry points:
- ask_model():        plain, ungrounded call (the 1.3 notebook + contrast script)
- answer_question():  grounded call over retrieved passages, structured output

The API key is read only from the environment. Tests always pass a stub client.
"""

from google import genai
from pydantic import BaseModel

from foodsafety_rag.config import GEMINI_MODEL, get_settings
from foodsafety_rag.schemas import Answer, Citation, Passage


class GenerationError(RuntimeError):
    """Raised when the Gemini call fails; the API layer maps it to a friendly 502."""


class ModelAnswer(BaseModel):
    """The structured response schema Gemini fills in (native responseSchema)."""
    answer: str
    supported: bool
    citations: list[Citation]


GROUNDED_PROMPT = """\
You are a food-safety compliance assistant for restaurant staff.
Answer the question using ONLY the numbered passages below - do not use any
other knowledge. Quote temperatures, times, and concentrations exactly as
written. For each fact you state, add a citation with the passage's doc,
heading, and a short snippet. If the passages do not contain the answer, set
supported=false and say the trusted documents do not cover this question.

Passages:
{passages}

Question: {question}
"""


def get_client() -> genai.Client:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env, add your key "
            "(https://aistudio.google.com/apikey), and load it into the environment."
        )
    return genai.Client(api_key=settings.gemini_api_key)


def ask_model(prompt: str, *, client: genai.Client | None = None) -> str:
    """One plain LLM call - no retrieval, no grounding. The 1.3 'first taste'."""
    client = client or get_client()
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    except Exception as exc:  # network, quota, auth - never leak a stack trace to users
        raise GenerationError(f"Gemini call failed ({type(exc).__name__})") from exc
    return response.text


def answer_question(question: str, passages: list[Passage],
                    *, client: genai.Client | None = None) -> Answer:
    """Grounded call: passages + question -> structured Answer."""
    client = client or get_client()
    passage_block = "\n\n".join(
        f"[{i + 1}] {p.doc} — {p.heading}\n{p.text}"
        for i, p in enumerate(passages)
    )
    prompt = GROUNDED_PROMPT.format(passages=passage_block, question=question)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ModelAnswer,
            },
        )
        parsed: ModelAnswer = response.parsed
    except Exception as exc:
        raise GenerationError(f"Gemini call failed ({type(exc).__name__})") from exc
    return Answer(
        question=question,
        answer=parsed.answer,
        grounded=parsed.supported,
        citations=parsed.citations,
        passages=passages,
    )

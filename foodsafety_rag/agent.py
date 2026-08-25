"""Typed Pydantic AI boundary for grounded answers.

Provider selection lives here so the retrieval pipeline and FastAPI routes keep
one stable interface. The raw ``ask_model`` and streaming examples remain in
``generate.py`` to show the lower-level OpenAI-compatible SDK path explicitly.
"""

import os
import re
from urllib.parse import urlparse

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, NativeOutput, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from foodsafety_rag.config import Settings, get_settings
from foodsafety_rag.schemas import Citation, Passage

LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")


class ModelAnswer(BaseModel):
    """The provider-independent output contract for a grounded answer."""

    answer: str
    supported: bool
    citations: list[Citation]


def is_local_url(base_url: str) -> bool:
    return (urlparse(base_url).hostname or "") in LOCAL_HOSTS


def build_model(settings: Settings | None = None):
    """Build the configured model without changing application-facing code."""

    settings = settings or get_settings()
    provider = settings.llm_provider

    if provider == "course":
        return OpenAIChatModel(
            settings.llm_model,
            provider=OpenAIProvider(
                base_url=settings.llm_base_url,
                # Local servers do not authenticate, but the SDK requires a value.
                api_key=settings.llm_api_key or "local",
            ),
        )
    if provider == "openai":
        # Explicitly select Chat Completions; bare ``openai:`` selects Responses.
        return f"openai-chat:{settings.llm_model}"
    if provider == "anthropic":
        return f"anthropic:{settings.llm_model}"
    if provider in {"google-cloud", "vertex"}:
        return f"google-cloud:{settings.llm_model}"

    raise RuntimeError(
        "LLM_PROVIDER must be one of: course, openai, anthropic, google-cloud"
    )


def check_credentials(settings: Settings | None = None) -> None:
    """Fail at startup when a selected hosted provider lacks credentials."""

    settings = settings or get_settings()
    provider = settings.llm_provider

    if provider == "course":
        if not settings.llm_api_key and not is_local_url(settings.llm_base_url):
            raise RuntimeError(
                f"LLM_API_KEY is not set, and {settings.llm_base_url} is not a "
                "local server. Add the course or endpoint key to .env."
            )
    elif provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
    elif provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic"
        )
    elif provider not in {"google-cloud", "vertex"}:
        # Also validates unknown values before the first request.
        build_model(settings)
    # google-cloud uses Application Default Credentials; the Google SDK reports
    # a detailed setup error if ADC is unavailable.


food_safety_agent = Agent(
    build_model(),
    output_type=NativeOutput(ModelAnswer),
    deps_type=list[Passage],
    instructions=(
        "Answer only from the supplied passages. Return a cautious structured "
        "answer and cite only passages that were actually supplied."
    ),
    retries={"output": 1},
    # Hosted provider strings can be selected before their SDK reads credentials.
    defer_model_check=True,
)


_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Fold the formatting variance models introduce when they repeat a title:
    em/en dashes become hyphens, the section sign is dropped, whitespace and
    case collapse. Semantic content is untouched, so a fabricated citation
    still cannot match."""
    text = (text.replace("—", "-").replace("–", "-")
            .replace("§", ""))
    return _WS_RE.sub(" ", text).strip().lower()


def citation_matches(citation: Citation, passages: list[Passage]) -> bool:
    """True when the citation names one of the supplied passages. The heading
    must match exactly (normalized); the doc title may extend a supplied title
    that was truncated at a line break, or be extended by it."""
    c_doc, c_heading = _normalize(citation.doc), _normalize(citation.heading)
    for passage in passages:
        if c_heading != _normalize(passage.heading):
            continue
        p_doc = _normalize(passage.doc)
        if c_doc == p_doc:
            return True
        shorter, longer = sorted((c_doc, p_doc), key=len)
        if len(shorter) >= 12 and longer.startswith(shorter):
            return True
    return False


@food_safety_agent.output_validator
def validate_citations(
    ctx: RunContext[list[Passage]], output: ModelAnswer
) -> ModelAnswer:
    """Reject structurally valid citations that do not name retrieved evidence."""

    invalid = [
        citation
        for citation in output.citations
        if not citation_matches(citation, ctx.deps)
    ]
    if invalid:
        raise ModelRetry("Every citation must name one of the supplied passages.")
    if not output.supported and output.citations:
        raise ModelRetry("An unsupported answer must not include citations.")
    return output

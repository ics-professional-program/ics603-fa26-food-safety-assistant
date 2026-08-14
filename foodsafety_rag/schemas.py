"""Pydantic models for the assistant (course modules M4/M6)."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc: str
    heading: str
    snippet: str


class Passage(BaseModel):
    doc: str
    heading: str
    text: str
    score: float


class Usage(BaseModel):
    """What one generation call cost, in the two units that meter it: tokens
    and time. Absent when no model call was made (a declined question) or when
    the endpoint reports no usage."""
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Answer(BaseModel):
    question: str
    answer: str
    grounded: bool
    citations: list[Citation] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)
    usage: Usage | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)

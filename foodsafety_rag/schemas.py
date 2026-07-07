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


class Answer(BaseModel):
    question: str
    answer: str
    grounded: bool
    citations: list[Citation] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)

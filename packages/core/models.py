from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class Document(BaseModel):
    doc_id: str
    title: str
    source_path: str
    doc_type: Literal["pdf", "html", "txt", "other"]
    metadata: dict = Field(default_factory=dict)


class ParentBlock(BaseModel):
    parent_id: str
    doc_id: str
    parent_type: Literal["page", "section", "table", "other"]
    title: str | None = None
    page_number: int | None = None
    section_path: str | None = None
    text: str


class ChildChunk(BaseModel):
    chunk_id: str
    doc_id: str
    parent_id: str
    text: str
    offset_start: int
    offset_end: int


class RetrievalChunkHit(BaseModel):
    chunk_id: str
    score: float
    source: Literal["vector", "keyword"]


class ParentCandidate(BaseModel):
    parent_id: str
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    chunk_hits: list[RetrievalChunkHit]


class RerankResult(BaseModel):
    parent_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    why_relevant: str
    support_type: Literal["definition", "number", "explanation", "risk", "event", "other"]
    combined_score: float | None = None


class Citation(BaseModel):
    parent_id: str
    quote: str
    page_number: int | None = None
    section_path: str | None = None


class AnswerSentence(BaseModel):
    text: str
    citations: list[Citation]


class AnswerPayload(BaseModel):
    answer: str
    sentences: list[AnswerSentence]
    citations: list[Citation]
    parents_used: list[str]
    confidence: str | None = None
    not_found_reason: str | None = None

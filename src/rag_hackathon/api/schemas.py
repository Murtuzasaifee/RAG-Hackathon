from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    doc_id: str | None = None


class IngestResponse(BaseModel):
    job_id: str
    doc_id: str
    version_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    doc_id: str
    version_id: str
    state: str
    stage: str
    progress: int = 0
    error: str | None = None


class QueryRequest(BaseModel):
    query: str
    doc_ids: list[str] | None = None
    version_ids: list[str] | None = None
    top_k: int = 20
    top_n: int = 5


class CitationResponse(BaseModel):
    doc_id: str
    version_id: str
    page: int
    section_path: list[str] = Field(default_factory=list)
    bbox: list[float] = Field(default_factory=list)
    chunk_text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    request_id: str
    timings_ms: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    message: str

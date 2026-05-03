from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NAMESPACE_RAG = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

ChunkType = Literal["text", "table", "image", "formula", "algorithm"]


class ParsedElement(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str  # document_title, paragraph_title, figure_title, text, table, image, formula, algorithm
    text: str
    bbox: list[float] = Field(default_factory=list)
    reading_order: int
    page: int


class ParsedDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    elements: list[ParsedElement] = Field(default_factory=list)


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    version_id: str
    chunk_index: int
    text: str
    page: int
    section_path: list[str] = Field(default_factory=list)
    bbox: list[float] = Field(default_factory=list)
    chunk_type: ChunkType = "text"


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    version_id: str
    page: int
    section_path: list[str] = Field(default_factory=list)
    bbox: list[float] = Field(default_factory=list)
    chunk_text: str
    score: float


class RetrievalHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    version_id: str
    chunk_index: int
    chunk_text: str
    page: int
    section_path: list[str] = Field(default_factory=list)
    bbox: list[float] = Field(default_factory=list)
    chunk_type: ChunkType = "text"
    score: float


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    citations: list[Citation] = Field(default_factory=list)
    model: str
    usage: dict = Field(default_factory=dict)


JobState = Literal["pending", "running", "done", "failed"]
JobStage = Literal[
    "queued",
    "parsing",
    "chunking",
    "embedding_dense",
    "embedding_sparse",
    "indexing",
    "done",
    "failed",
]


class JobStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    doc_id: str
    version_id: str
    state: JobState
    stage: JobStage
    progress: int = 0
    error: str | None = None
    created_at: datetime
    updated_at: datetime


def make_point_id(doc_id: str, version_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(NAMESPACE_RAG, f"{doc_id}:{version_id}:{chunk_index}"))

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class Citation(BaseModel):
    document: str
    page: int | None = None
    chunk_id: str
    source: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Citation]
    dense_count: int
    bm25_count: int
    fused_count: int
    reranked_count: int
    latency: dict[str, float]
    total_latency_ms: float
    request_id: str


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    size_bytes: int
    chunk_count: int
    created_at: str
    duplicate: bool = False


class DeleteResponse(BaseModel):
    deleted: bool
    document_id: str


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str


class UploadResponse(DocumentResponse):
    pass
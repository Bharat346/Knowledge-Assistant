from __future__ import annotations

from fastapi import APIRouter, File, Request, UploadFile

from app.schemas.api import DeleteResponse, DocumentResponse, HealthResponse, QueryRequest, QueryResponse
from app.services.rag_service import ServiceError

router = APIRouter()


def _service(request: Request):
    return request.app.state.rag_service


@router.get("/health", response_model=HealthResponse)
async def health() -> dict:
    return {"status": "ok"}


@router.post("/api/documents/upload", response_model=DocumentResponse)
async def upload_document(request: Request, file: UploadFile = File(...)) -> dict:
    maximum = request.app.state.settings.max_file_size_mb * 1024 * 1024
    content = await file.read(maximum + 1)
    if len(content) > maximum:
        raise ServiceError(413, "The uploaded file exceeds the configured size limit")
    return _service(request).upload_document(file.filename or "", content)


@router.get("/api/documents", response_model=list[DocumentResponse])
async def list_documents(request: Request) -> list[dict]:
    return _service(request).list_documents()


@router.delete("/api/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(request: Request, document_id: str) -> dict:
    _service(request).delete_document(document_id)
    return {"deleted": True, "document_id": document_id}


@router.post("/api/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> dict:
    request_id = request.state.request_id
    return _service(request).query(body.question, request_id)
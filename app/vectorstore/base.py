from __future__ import annotations

from typing import Protocol


class VectorStore(Protocol):
    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> None: ...

    def search(self, query_embedding: list[float], top_k: int) -> list[dict]: ...

    def delete_document(self, document_id: str) -> None: ...
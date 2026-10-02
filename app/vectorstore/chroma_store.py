from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from app.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_directory: Path, collection_name: str = "rag_chunks") -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        metadatas = []
        for chunk in chunks:
            metadata = {key: value for key, value in chunk["metadata"].items() if value is not None}
            metadatas.append(metadata)
        self.collection.upsert(
            ids=[chunk["chunk_id"] for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk["text"] for chunk in chunks],
            metadatas=metadatas,
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        if self.collection.count() == 0:
            return []
        response = self.collection.query(
            query_embeddings=[query_embedding], n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for chunk_id, text, metadata, distance in zip(
            response["ids"][0], response["documents"][0], response["metadatas"][0], response["distances"][0]
        ):
            results.append({
                "chunk_id": chunk_id, "text": text, "metadata": metadata,
                "score": 1.0 / (1.0 + max(float(distance), 0.0)),
            })
        return results

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})
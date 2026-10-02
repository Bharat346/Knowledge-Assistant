from __future__ import annotations

import hashlib
import math

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.services.rag_service import RAGService


class FakeEmbeddings:
    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * 64
        for token in text.lower().split():
            index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % len(vector)
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class MemoryVectorStore:
    def __init__(self):
        self.rows = {}

    def add(self, chunks, embeddings):
        for chunk, embedding in zip(chunks, embeddings):
            self.rows[chunk["chunk_id"]] = (chunk, embedding)

    def search(self, query_embedding, top_k):
        ranked = []
        for chunk, embedding in self.rows.values():
            score = sum(left * right for left, right in zip(query_embedding, embedding))
            ranked.append(({**chunk, "score": score}, score))
        return [row for row, score in sorted(ranked, key=lambda item: item[1], reverse=True)[:top_k] if score > 0]

    def delete_document(self, document_id):
        self.rows = {
            chunk_id: entry for chunk_id, entry in self.rows.items()
            if entry[0]["metadata"].get("document_id") != document_id
        }


class FakeLLM:
    def generate(self, query, context):
        return f"Grounded answer for: {query} [S1]"


class DeterministicReranker:
    def rerank(self, query, documents, top_k):
        terms = set(query.lower().split())
        ranked = sorted(
            documents,
            key=lambda row: len(terms & set(row["text"].lower().split())),
            reverse=True,
        )
        return [{**row, "rerank_score": float(index)} for index, row in enumerate(ranked[:top_k], start=1)]


@pytest.fixture
def providers():
    return FakeEmbeddings(), MemoryVectorStore(), FakeLLM(), DeterministicReranker()


@pytest.fixture
def client(tmp_path, providers):
    from app.main import create_app
    from app.retrieval.sparse import SparseRetriever

    embeddings, vector_store, llm, reranker = providers
    settings = Settings(
        gemini_api_key="test-key", data_dir=tmp_path, chunk_size=180, chunk_overlap=20,
        dense_top_k=5, bm25_top_k=5, final_top_k=3, max_context_tokens=800,
    )
    service = RAGService(
        settings, embedding_provider=embeddings, vector_store=vector_store, llm_provider=llm,
        sparse_retriever=SparseRetriever(tmp_path / "bm25_index.json"), reranker=reranker,
    )
    return TestClient(create_app(settings, service))
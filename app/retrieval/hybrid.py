from __future__ import annotations

from time import perf_counter

from app.embeddings.base import EmbeddingProvider
from app.retrieval.sparse import SparseRetriever, reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.vectorstore.base import VectorStore


class HybridRetriever:
    def __init__(
        self, embeddings: EmbeddingProvider, vector_store: VectorStore,
        sparse: SparseRetriever, reranker: Reranker,
        dense_top_k: int, bm25_top_k: int, rrf_k: int, final_top_k: int,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.sparse = sparse
        self.reranker = reranker
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k

    def retrieve(self, query: str) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        dense, sparse, fused, reranked, _ = self.retrieve_staged(query)
        return dense, sparse, fused, reranked

    def retrieve_staged(self, query: str) -> tuple[list[dict], list[dict], list[dict], list[dict], dict[str, float]]:
        timings: dict[str, float] = {}
        started = perf_counter()
        query_embedding = self.embeddings.embed_query(query)
        dense = self.vector_store.search(query_embedding, self.dense_top_k)
        timings["dense_retrieval"] = (perf_counter() - started) * 1000
        started = perf_counter()
        sparse = self.sparse.search(query, self.bm25_top_k)
        timings["bm25_retrieval"] = (perf_counter() - started) * 1000
        started = perf_counter()
        fused = reciprocal_rank_fusion(dense, sparse, self.rrf_k)
        timings["rrf_fusion"] = (perf_counter() - started) * 1000
        started = perf_counter()
        reranked = self.reranker.rerank(query, fused, self.final_top_k)
        timings["reranking"] = (perf_counter() - started) * 1000
        return dense, sparse, fused, reranked, timings
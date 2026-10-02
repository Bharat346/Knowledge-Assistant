from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from threading import RLock

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


class SparseRetriever:
    """BM25Okapi retrieval with a durable JSON corpus and in-memory scoring index."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._documents: dict[str, dict] = {}
        if self.index_path.exists():
            try:
                loaded = json.loads(self.index_path.read_text(encoding="utf-8"))
                self._documents = {row["chunk_id"]: row for row in loaded}
            except Exception:
                logger.exception("BM25 index could not be loaded; starting with an empty sparse index")
        self._rebuild()

    def _rebuild(self) -> None:
        self._rows = list(self._documents.values())
        self._tokens = [tokenize(row["text"]) for row in self._rows]
        self._index = BM25Okapi(self._tokens) if any(self._tokens) else None

    def _persist(self) -> None:
        temporary = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._rows, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.index_path)

    def add(self, chunks: list[dict]) -> None:
        with self._lock:
            for chunk in chunks:
                self._documents[chunk["chunk_id"]] = chunk
            self._rebuild()
            self._persist()

    def search(self, query: str, top_k: int) -> list[dict]:
        terms = tokenize(query)
        with self._lock:
            if not terms or self._index is None:
                return []
            scores = self._index.get_scores(terms)
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
            return [
                {**self._rows[index], "score": float(score)}
                for index, score in ranked
                if score != 0 or set(terms).intersection(self._tokens[index])
            ][:top_k]

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self._documents = {
                chunk_id: row for chunk_id, row in self._documents.items()
                if row["metadata"].get("document_id") != document_id
            }
            self._rebuild()
            self._persist()


def reciprocal_rank_fusion(
    dense_results: list[dict], sparse_results: list[dict], rrf_k: int = 60
) -> list[dict]:
    fused: dict[str, dict] = {}
    for results in (dense_results, sparse_results):
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = {**result, "score": 0.0}
            fused[chunk_id]["score"] += 1.0 / (rrf_k + rank)
    return sorted(fused.values(), key=lambda result: result["score"], reverse=True)
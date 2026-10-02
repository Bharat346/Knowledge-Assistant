from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._fallback_logged = False

    def rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        if not documents:
            return []
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            scores = self._model.predict([(query, document["text"]) for document in documents])
            ranked = [({**document, "rerank_score": float(score)}, float(score)) for document, score in zip(documents, scores)]
            return [item[0] for item in sorted(ranked, key=lambda item: item[1], reverse=True)[:top_k]]
        except Exception:
            if not self._fallback_logged:
                logger.exception("Local reranker unavailable; using lexical-overlap fallback")
                self._fallback_logged = True
            query_terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
            ranked = []
            for document in documents:
                document_terms = set(re.findall(r"[a-z0-9_]+", document["text"].lower()))
                score = len(query_terms & document_terms) / max(len(query_terms), 1)
                ranked.append(({**document, "rerank_score": score}, score))
            return [item[0] for item in sorted(ranked, key=lambda item: item[1], reverse=True)[:top_k]]
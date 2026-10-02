from __future__ import annotations

import logging
from threading import Lock

from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._lock = Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer(self.model_name)
                    except Exception as exc:
                        logger.exception("Embedding model initialization failed")
                        raise RuntimeError("Embedding model is unavailable") from exc
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._get_model().encode(
                texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.tolist()
        except Exception as exc:
            logger.exception("Document embedding failed")
            raise RuntimeError("Could not create document embeddings") from exc

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
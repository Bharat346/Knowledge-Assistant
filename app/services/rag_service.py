from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.config.settings import Settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.sentence_transformer import SentenceTransformerProvider
from app.generation.llm import GeminiProvider, LLMProvider, LLMProviderError
from app.ingestion.chunking import chunk_sections
from app.ingestion.parser import DocumentParseError, parse_document
from app.retrieval.context import select_context
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.sparse import SparseRetriever
from app.vectorstore.base import VectorStore
from app.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class RAGService:
    def __init__(
        self,
        settings: Settings,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        llm_provider: LLMProvider | None = None,
        vector_store: VectorStore | None = None,
        sparse_retriever: SparseRetriever | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings.validate_chunking()
        self.data_dir = settings.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = self.data_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.data_dir / "documents.json"
        self.registry: dict[str, dict] = {}
        if self.registry_path.exists():
            try:
                self.registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
            except Exception:
                logger.exception("Document registry could not be loaded")
        self.embeddings = embedding_provider or SentenceTransformerProvider(settings.embedding_model)
        self.llm = llm_provider or GeminiProvider(settings.gemini_api_key, settings.gemini_model)
        self.vector_store = vector_store or ChromaVectorStore(self.data_dir / "chroma")
        self.sparse = sparse_retriever or SparseRetriever(self.data_dir / "bm25_index.json")
        self.reranker = reranker or Reranker(settings.reranker_model)
        self.retriever = HybridRetriever(
            self.embeddings, self.vector_store, self.sparse, self.reranker,
            settings.dense_top_k, settings.bm25_top_k, settings.rrf_k, settings.final_top_k,
        )

    def _save_registry(self) -> None:
        temporary = self.registry_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.registry, indent=2), encoding="utf-8")
        temporary.replace(self.registry_path)

    def upload_document(self, filename: str, content: bytes) -> dict:
        maximum = self.settings.max_file_size_mb * 1024 * 1024
        if not filename or not filename.strip():
            raise ServiceError(400, "A filename is required")
        if not content:
            raise ServiceError(400, "The uploaded file is empty")
        if len(content) > maximum:
            raise ServiceError(413, "The uploaded file exceeds the configured size limit")
        safe_filename = re.sub(r"[\\/]+", "_", filename).strip(" .")
        digest = hashlib.sha256(content).hexdigest()
        existing = next((row for row in self.registry.values() if row["sha256"] == digest), None)
        if existing:
            return {key: value for key, value in existing.items() if key != "sha256"} | {"duplicate": True}
        document_id = str(uuid.uuid4())
        try:
            file_type, sections = parse_document(safe_filename, content)
        except DocumentParseError as exc:
            raise ServiceError(415 if "Unsupported" in str(exc) else 400, str(exc)) from exc
        relative_path = f"uploads/{document_id}_{safe_filename}"
        chunks = chunk_sections(
            sections, document_id=document_id, filename=safe_filename, file_type=file_type,
            source_path=relative_path, chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ServiceError(400, "No readable text was found in the uploaded file")
        records = [{"chunk_id": item.chunk_id, "text": item.text, "metadata": item.metadata} for item in chunks]
        try:
            embeddings = self.embeddings.embed_documents([record["text"] for record in records])
            if len(embeddings) != len(records):
                raise RuntimeError("Embedding provider returned an incomplete batch")
            self.vector_store.add(records, embeddings)
            self.sparse.add(records)
            (self.upload_dir / f"{document_id}_{safe_filename}").write_bytes(content)
        except Exception as exc:
            logger.exception("Indexing failed for upload %s", safe_filename)
            try:
                self.vector_store.delete_document(document_id)
                self.sparse.delete_document(document_id)
            except Exception:
                logger.exception("Failed to roll back partially indexed upload")
            raise ServiceError(503, "The document could not be indexed; check embedding and storage configuration") from exc
        row = {
            "document_id": document_id,
            "filename": safe_filename,
            "file_type": file_type,
            "size_bytes": len(content),
            "chunk_count": len(records),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sha256": digest,
        }
        self.registry[document_id] = row
        try:
            self._save_registry()
        except Exception as exc:
            logger.exception("Document registry persistence failed")
            self.vector_store.delete_document(document_id)
            self.sparse.delete_document(document_id)
            (self.upload_dir / f"{document_id}_{safe_filename}").unlink(missing_ok=True)
            raise ServiceError(503, "The document could not be saved") from exc
        return {key: value for key, value in row.items() if key != "sha256"} | {"duplicate": False}

    def list_documents(self) -> list[dict]:
        return [
            {key: value for key, value in row.items() if key != "sha256"}
            for row in sorted(self.registry.values(), key=lambda item: item["created_at"], reverse=True)
        ]

    def delete_document(self, document_id: str) -> None:
        row = self.registry.get(document_id)
        if row is None:
            raise ServiceError(404, "Document not found")
        try:
            self.vector_store.delete_document(document_id)
            self.sparse.delete_document(document_id)
        except Exception as exc:
            logger.exception("Document deletion failed for %s", document_id)
            raise ServiceError(503, "The document could not be removed from the indexes") from exc
        self.registry.pop(document_id)
        self._save_registry()
        (self.upload_dir / f"{document_id}_{row['filename']}").unlink(missing_ok=True)

    def query(self, question: str, request_id: str) -> dict:
        started = perf_counter()
        question = question.strip()
        if not question:
            raise ServiceError(400, "Question cannot be empty")
        try:
            dense, sparse, fused, reranked, latency = self.retriever.retrieve_staged(question)
        except Exception as exc:
            logger.exception("Retrieval failed for request %s", request_id)
            raise ServiceError(503, "Retrieval is temporarily unavailable") from exc
        context_started = perf_counter()
        context, selected = select_context(reranked, self.settings.max_context_tokens)
        latency["context_building"] = (perf_counter() - context_started) * 1000
        generation_started = perf_counter()
        if not context:
            answer = "I could not find relevant information in the indexed documents, so the context is insufficient to answer."
        else:
            try:
                answer = self.llm.generate(question, context)
            except Exception as exc:
                logger.exception("Generation failed for request %s", request_id)
                if isinstance(exc, LLMProviderError):
                    raise ServiceError(exc.status_code, str(exc)) from exc
                raise ServiceError(502, "Answer generation is temporarily unavailable") from exc
        latency["generation"] = (perf_counter() - generation_started) * 1000
        sources = []
        seen_sources = set()
        for result in selected:
            metadata = result.get("metadata", {})
            citation = {
                "document": metadata.get("filename", "unknown"),
                "page": metadata.get("page"),
                "chunk_id": result["chunk_id"],
                "source": metadata.get("source"),
            }
            key = (citation["document"], citation["page"], citation["chunk_id"])
            if key not in seen_sources:
                sources.append(citation)
                seen_sources.add(key)
        latency["total"] = (perf_counter() - started) * 1000
        return {
            "answer": answer,
            "sources": sources,
            "dense_count": len(dense),
            "bm25_count": len(sparse),
            "fused_count": len(fused),
            "reranked_count": len(reranked),
            "latency": latency,
            "total_latency_ms": latency["total"],
            "request_id": request_id,
        }
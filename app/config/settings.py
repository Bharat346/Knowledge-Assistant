from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = Field(default=700, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    dense_top_k: int = Field(default=10, gt=0)
    bm25_top_k: int = Field(default=10, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    final_top_k: int = Field(default=5, gt=0)
    max_file_size_mb: int = Field(default=30, gt=0)
    max_context_tokens: int = Field(default=6000, gt=0)
    data_dir: Path = Path("./data")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            dense_top_k=int(os.getenv("DENSE_TOP_K", "10")),
            bm25_top_k=int(os.getenv("BM25_TOP_K", "10")),
            rrf_k=int(os.getenv("RRF_K", "60")),
            final_top_k=int(os.getenv("FINAL_TOP_K", "5")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "30")),
            max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "6000")),
            data_dir=Path(os.getenv("DATA_DIR", "./data")),
        )

    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self
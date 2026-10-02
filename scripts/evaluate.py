"""Standalone BM25 evaluation: python scripts/evaluate.py."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.sparse import SparseRetriever


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k if k else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    return next((1.0 / rank for rank, item in enumerate(retrieved, 1) if item in relevant), 0.0)


def main() -> None:
    corpus = [
        ("retention", "Audit records must be retained for seven years."),
        ("security", "Security incidents must be reported to the response team."),
        ("access", "Access reviews happen quarterly for all production systems."),
    ]
    queries = [
        ("How long are audit records retained?", {"retention"}),
        ("Who receives security incident reports?", {"security"}),
        ("How often are production access reviews?", {"access"}),
    ]
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as directory:
        retriever = SparseRetriever(Path(directory) / "bm25.json")
        retriever.add([
            {"chunk_id": chunk_id, "text": text, "metadata": {"document_id": "eval"}}
            for chunk_id, text in corpus
        ])
        k = 2
        recalls, precisions, reciprocal_ranks = [], [], []
        for query, relevant in queries:
            found = [row["chunk_id"] for row in retriever.search(query, k)]
            recalls.append(recall_at_k(found, relevant, k))
            precisions.append(precision_at_k(found, relevant, k))
            reciprocal_ranks.append(reciprocal_rank(found, relevant))
    print(f"Recall@{k}: {sum(recalls) / len(recalls):.3f}")
    print(f"Precision@{k}: {sum(precisions) / len(precisions):.3f}")
    print(f"MRR: {sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}")


if __name__ == "__main__":
    main()
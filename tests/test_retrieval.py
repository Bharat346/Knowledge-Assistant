from app.retrieval.context import build_context, select_context
from app.retrieval.reranker import Reranker
from app.retrieval.sparse import SparseRetriever, reciprocal_rank_fusion


def make_result(chunk_id, text, document_id="doc"):
    return {
        "chunk_id": chunk_id, "text": text, "score": 1.0,
        "metadata": {"document_id": document_id, "filename": "guide.md", "page": 1, "source": "guide.md"},
    }


def test_bm25_is_ranked_and_persisted(tmp_path):
    index = SparseRetriever(tmp_path / "bm25.json")
    index.add([make_result("a", "kiwi mango orchard"), make_result("b", "kiwi kiwi kiwi harvest")])
    assert index.search("harvest", 2)[0]["chunk_id"] == "b"
    reloaded = SparseRetriever(tmp_path / "bm25.json")
    assert reloaded.search("orchard mango", 1)[0]["chunk_id"] == "a"
    reloaded.delete_document("doc")
    assert reloaded.search("kiwi", 3) == []


def test_rrf_combines_rankings_and_deduplicates():
    dense = [make_result("same", "dense"), make_result("dense-only", "dense only")]
    sparse = [make_result("same", "sparse"), make_result("sparse-only", "sparse only")]
    fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60)
    assert [item["chunk_id"] for item in fused] == ["same", "dense-only", "sparse-only"]
    assert fused[0]["score"] == 2 / 61


def test_context_removes_duplicates_and_obeys_budget():
    first = make_result("one", "A concise evidence statement.")
    context, selected = select_context([first, {**first}], max_tokens=80)
    assert context.count("A concise evidence statement") == 1
    assert len(selected) == 1
    assert "page=1" in build_context([first], 80)


def test_reranker_uses_model_scores():
    class Model:
        def predict(self, pairs):
            return [0.1, 0.9]

    reranker = Reranker("not-loaded")
    reranker._model = Model()
    results = reranker.rerank("question", [make_result("a", "first"), make_result("b", "second")], 1)
    assert results[0]["chunk_id"] == "b"
    assert results[0]["rerank_score"] == 0.9
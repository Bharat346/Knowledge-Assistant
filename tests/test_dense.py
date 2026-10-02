from app.vectorstore.chroma_store import ChromaVectorStore


def test_dense_vector_retrieval_returns_common_result_shape(tmp_path):
    store = ChromaVectorStore(tmp_path / "chroma")
    chunks = [
        {"chunk_id": "tea", "text": "green tea leaves", "metadata": {"filename": "tea.txt"}},
        {"chunk_id": "car", "text": "electric car battery", "metadata": {"filename": "car.txt"}},
    ]
    store.add(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    results = store.search([0.9, 0.1, 0.0], 2)
    assert results[0]["chunk_id"] == "tea"
    assert set(("chunk_id", "text", "metadata", "score")).issubset(results[0])
    assert ChromaVectorStore(tmp_path / "chroma").collection.count() == 2
    store.delete_document("missing")
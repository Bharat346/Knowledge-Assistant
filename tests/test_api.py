from app.generation.llm import LLMProviderError


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers.get("x-request-id")


def test_upload_duplicate_query_and_delete(client):
    content = b"The access policy keeps audit records for seven years."
    uploaded = client.post(
        "/api/documents/upload", files={"file": ("policy.txt", content, "text/plain")}
    )
    assert uploaded.status_code == 200
    document = uploaded.json()
    assert document["chunk_count"] == 1
    assert document["duplicate"] is False

    duplicate = client.post(
        "/api/documents/upload", files={"file": ("renamed.txt", content, "text/plain")}
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["document_id"] == document["document_id"]

    answer = client.post("/api/query", json={"question": "How long are audit records kept?"})
    assert answer.status_code == 200
    body = answer.json()
    assert body["answer"].startswith("Grounded answer")
    assert body["sources"][0]["document"] == "policy.txt"
    assert body["sources"][0]["chunk_id"]
    assert {"dense_retrieval", "bm25_retrieval", "rrf_fusion", "reranking", "generation", "total"}.issubset(body["latency"])

    deleted = client.delete(f"/api/documents/{document['document_id']}")
    assert deleted.status_code == 200
    assert client.get("/api/documents").json() == []
    empty_answer = client.post("/api/query", json={"question": "audit records"}).json()
    assert empty_answer["sources"] == []
    assert "insufficient" in empty_answer["answer"]


def test_invalid_upload_and_empty_query(client):
    unsupported = client.post("/api/documents/upload", files={"file": ("archive.exe", b"data")})
    assert unsupported.status_code == 415
    empty = client.post("/api/documents/upload", files={"file": ("empty.txt", b"")})
    assert empty.status_code == 400
    query = client.post("/api/query", json={"question": "   "})
    assert query.status_code == 400


def test_generation_provider_error_is_actionable(client):
    uploaded = client.post(
        "/api/documents/upload",
        files={"file": ("facts.txt", b"The archive is retained for five years.", "text/plain")},
    )
    assert uploaded.status_code == 200

    class RetiredModel:
        def generate(self, query, context):
            raise LLMProviderError(
                "Gemini model 'retired' is unavailable. Set GEMINI_MODEL to a supported model.", 503
            )

    client.app.state.rag_service.llm = RetiredModel()
    response = client.post("/api/query", json={"question": "How long is the archive retained?"})
    assert response.status_code == 503
    assert "Set GEMINI_MODEL" in response.json()["detail"]
    assert "Answer generation is temporarily unavailable" not in response.json()["detail"]


def test_generation_service_unavailable_suggests_retry(monkeypatch):
    from app.generation.llm import GeminiProvider, LLMProviderError
    import app.generation.llm as llm_module
    from google import genai

    class BusyService(Exception):
        code = 503

    calls = 0

    class Models:
        def generate_content(self, **kwargs):
            nonlocal calls
            calls += 1
            raise BusyService()

    class Client:
        def __init__(self, **kwargs):
            self.models = Models()

    monkeypatch.setattr(genai, "Client", Client)
    monkeypatch.setattr(llm_module, "sleep", lambda _: None)
    try:
        GeminiProvider("test-only", "gemini-3.8-flash").generate("question", "context")
    except LLMProviderError as exc:
        assert exc.status_code == 503
        assert "Retry your question" in str(exc)
    else:
        raise AssertionError("Expected a temporary Gemini service error")
    assert calls == 3


def test_generation_retries_temporary_failure(monkeypatch):
    import app.generation.llm as llm_module
    from app.generation.llm import GeminiProvider
    from google import genai

    calls = 0

    class BusyService(Exception):
        code = 503

    class Models:
        def generate_content(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise BusyService()
            return type("Response", (), {"text": "Recovered answer [S1]."})()

    class Client:
        def __init__(self, **kwargs):
            self.models = Models()

    monkeypatch.setattr(genai, "Client", Client)
    monkeypatch.setattr(llm_module, "sleep", lambda _: None)
    answer = GeminiProvider("test-only", "gemini-3.8-flash").generate("question", "context")
    assert answer == "Recovered answer [S1]."
    assert calls == 2
# RAG Knowledge Assistant

A local-first, single-process Retrieval-Augmented Generation application. It indexes supported files incrementally into persistent Chroma and BM25 indexes, fuses both result lists with Reciprocal Rank Fusion, reranks candidates locally, then asks Gemini to answer only from bounded retrieved context. Citation objects are assembled by the application from chunk metadata, never generated as trusted source data by the model.

## Architecture

```text
Upload -> validate -> parse -> clean/chunk -> batch local embeddings -> Chroma
                                                `-> persistent BM25 corpus

Question -> query embedding -> dense top-K ----\
             BM25 top-K ------------------------> RRF -> dedupe -> local reranker
                                                    -> token-bounded context
                                                    -> Gemini -> answer + metadata citations
```

`EmbeddingProvider`, `VectorStore`, and `LLMProvider` keep model and storage choices replaceable. BM25 is implemented with `rank-bm25`'s Okapi scorer; RRF is implemented explicitly in the retrieval package. The reranker attempts the configured local Sentence Transformers cross-encoder and logs a lexical-overlap fallback if the model cannot load.

## Directory Tree

```text
app/
  api/routes.py
  config/settings.py
  embeddings/{base.py,sentence_transformer.py}
  generation/llm.py
  ingestion/{chunking.py,parser.py}
  retrieval/{context.py,hybrid.py,reranker.py,sparse.py}
  schemas/api.py
  services/rag_service.py
  vectorstore/{base.py,chroma_store.py}
  main.py
frontend/{index.html,style.css,app.js}
scripts/evaluate.py
tests/{conftest.py,test_api.py,test_chunking.py,test_dense.py,test_retrieval.py}
data/
requirements.txt
.env.example
```

## Setup And Run

Requires Python 3.11 or later. From the repository directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp -n .env.example .env
```

Set `GEMINI_API_KEY` in `.env` to enable answer generation. The embedding model and reranker are downloaded by Sentence Transformers the first time the relevant code path is used; allow network access for that initial download, or point the model settings to local model directories. Importing the app and launching it against an empty database do not load either model. The Gemini key is only required when a query has retrieved context to send for generation.

```bash
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Data persists under `DATA_DIR` (`./data` by default). Keep `.env` private; it is ignored by git. The application never executes uploaded code.

## Configuration

All settings have defaults and are read from environment variables or `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | empty | Gemini API credential; never hardcoded |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini generation model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local dense embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder reranker |
| `CHUNK_SIZE` | `700` | Chunk character limit |
| `CHUNK_OVERLAP` | `100` | Chunk overlap in characters |
| `DENSE_TOP_K` | `10` | Dense candidates |
| `BM25_TOP_K` | `10` | Sparse candidates |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `FINAL_TOP_K` | `5` | Reranked context candidates |
| `MAX_FILE_SIZE_MB` | `30` | Maximum upload size |
| `MAX_CONTEXT_TOKENS` | `6000` | Conservative word/punctuation token budget |
| `DATA_DIR` | `./data` | Chroma, BM25, registry, and upload storage |

The recursive splitter uses paragraph/sentence/whitespace boundaries for prose and common declaration boundaries for Python/JavaScript/TypeScript, falling back to character splitting for oversized segments. Chunk size is measured in characters. PDF page numbers are one-based. Non-PDF documents have a `null` page citation.

## HTTP API

- `GET /health` returns `{"status":"ok"}`.
- `POST /api/documents/upload` accepts multipart form data with a `file` field. The response includes `document_id`, `filename`, `file_type`, `size_bytes`, `chunk_count`, `created_at`, and `duplicate`.
- `GET /api/documents` lists indexed documents.
- `DELETE /api/documents/{id}` removes chunks from both dense and sparse indexes and deletes the stored upload.
- `POST /api/query` accepts `{"question":"..."}`. The response includes `answer`, metadata-backed `sources`, dense/BM25/fused/reranked counts, per-stage milliseconds, total milliseconds, and a request ID. Requests also receive an `X-Request-ID` response header.

Supported extensions: PDF, Markdown, TXT, Python, JavaScript, TypeScript, JSON, and CSV. Invalid/empty files, unsupported extensions, oversized files, bad queries, unavailable indexing dependencies, missing Gemini credentials, and provider failures return safe JSON errors without a traceback.

## Quick API Examples

Upload a document and ask a question:

```bash
curl -F 'file=@./notes.pdf' http://127.0.0.1:8000/api/documents/upload

curl -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the key points?"}'
```

The query response contains the answer, metadata-backed citations (`document`, `page`, `chunk_id`, and source path), dense/BM25/fused/reranked counts, stage timings, total latency, and request ID.

## Troubleshooting

- **Gemini model unavailable:** set `GEMINI_MODEL` in `.env` to a currently available Gemini model. The default is `gemini-3.8-flash`.
- **Gemini temporarily busy:** the provider retries transient server errors briefly; retry the question if the service remains unavailable.
- **API key rejected or missing:** verify `GEMINI_API_KEY` in `.env`, then restart Uvicorn. Never commit `.env`.
- **Slow first query or upload:** the local embedding and reranker models are downloaded and loaded on first use; later calls reuse them in process.
- **Unexpected index state:** stop the app before changing `DATA_DIR`; the directory contains the Chroma database, BM25 corpus, document registry, and uploaded originals.

## Tests And Evaluation

Tests use deterministic in-memory providers for service/API checks; no Gemini key or model download is needed. The Chroma test uses a temporary persistent database.

```bash
python -m pytest -q
python scripts/evaluate.py
```

The standalone evaluator prints Recall@K, Precision@K, and MRR for a tiny deterministic BM25 corpus. It is deliberately not part of the production request path.
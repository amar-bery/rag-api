# RAG API

A minimal Retrieval-Augmented Generation (RAG) service that answers natural-language questions from a small corpus of FAQ markdown files and exposes the functionality over a local HTTP API.

---

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Set your OpenAI API key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 3. Start the server (ingestion runs automatically on startup)
npm start
```

The server ingests all `.md` files in `./faqs/` at startup, embeds every chunk via the OpenAI Embeddings API, and then begins accepting requests.

---

## API

### `GET /health`

Liveness check.

```bash
curl http://localhost:3000/health
```

```json
{ "status": "ok" }
```

---

### `POST /ask`

Ask a question. The server retrieves the most relevant chunks and generates a grounded answer.

**Request body:**

| Field      | Type    | Required | Default | Constraints          |
|------------|---------|----------|---------|----------------------|
| `question` | string  | yes      | —       | non-empty            |
| `top_k`    | integer | no       | 4       | 1–10 inclusive       |

**Example:**

```bash
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset my password and set up MFA?"}'
```

**Response `200`:**

```json
{
  "answer": "To reset your password, visit the login page and click \"Forgot Password.\" (source: authentication.md). To enable MFA, go to Account Settings → Security → Enable Two-Factor Authentication and scan the QR code with an authenticator app (source: authentication.md).",
  "sources": ["authentication.md"]
}
```

**Response `400` (bad input):**

```json
{ "error": "Invalid input: \"question\" must be a non-empty string." }
```

**Response `500` (OpenAI or internal error):**

```json
{ "error": "Internal server error. Please try again." }
```

---

## Environment Variables

| Variable           | Required | Default       | Description                        |
|--------------------|----------|---------------|------------------------------------|
| `OPENAI_API_KEY`   | yes      | —             | OpenAI secret key                  |
| `PORT`             | no       | `3000`        | HTTP server port                   |
| `FAQ_DIR`          | no       | `./faqs`      | Path to FAQ markdown directory     |
| `COMPLETION_MODEL` | no       | `gpt-4o-mini` | OpenAI chat completion model       |

The server **fails fast** at startup if `OPENAI_API_KEY` is not set.

---

## Project Structure

```
rag-api/
├── faqs/                    # FAQ markdown source documents
│   ├── authentication.md
│   ├── sso.md
│   ├── employee-policy.md
│   └── api-integrations.md
├── rag.js                   # RAG core: chunking, embedding, retrieval, generation
├── server.js                # Express HTTP API wrapper
├── ingest.js                # Standalone ingestion smoke-test script
├── package.json
├── .env.example
└── README.md
```

---

## Design Decisions & Trade-offs

### Chunking strategy
Chunks are ~200 characters with a 40-character overlap. The overlap prevents meaningful sentences from being split cleanly across chunk boundaries — a common failure mode where the answer straddles two adjacent chunks and neither scores highly enough on its own. Splitting is attempted on sentence boundaries (`. `) within the final 60 characters of each window to avoid mid-sentence cuts.

**Trade-off:** Larger chunks preserve more context per retrieval hit but reduce precision (more noise per chunk). Smaller chunks increase precision but can miss context. 200 chars is on the smaller end; 400–600 chars works better for prose-heavy documents.

### Vector store
All embeddings are stored in a plain JavaScript array in memory. This is appropriate for a corpus of a few hundred chunks.

**Trade-off:** No persistence — the store is rebuilt from scratch on every startup (one batched OpenAI Embeddings API call). For larger corpora or faster cold starts, persist embeddings to a JSON file, SQLite, or a purpose-built store like Chroma or pgvector.

### Embedding model
`text-embedding-3-small` — OpenAI's smallest and cheapest embedding model. Dimensionality: 1536. Accuracy is sufficient for keyword-heavy FAQ retrieval.

**Trade-off:** `text-embedding-3-large` is ~6x more expensive and meaningfully more accurate for semantic queries (paraphrases, ambiguous phrasing). Not necessary here.

### Cosine similarity
Implemented in plain JS — no numpy, no external library. At this scale (< 1000 chunks, 1536-dim vectors) the inner loop completes in < 1ms. For thousands of documents, switch to a pre-built ANN index (FAISS, hnswlib) or a vector database.

### Batched embeddings at ingest
All chunks across all files are embedded in a **single API call** during ingestion. This is cheaper (fewer HTTP round-trips, OpenAI charges per token not per request) and faster than embedding each chunk individually.

### Completion model
`gpt-4o-mini` by default. It's fast, inexpensive, and follows instructions reliably. The system prompt instructs the model to cite sources inline using a `(source: filename.md)` convention and to answer strictly from the provided context.

**Temperature = 0** for deterministic answers — important for a Q&A system where factual consistency matters.

### Source citation
Sources are derived from the top-k retrieved chunks, not extracted from the LLM's output text. This is more reliable — LLMs sometimes hallucinate or omit citations. The `sources` array in the response reflects which files actually contributed context, regardless of what the model wrote.

### Input validation
- `question` must be a non-empty string. An empty string would result in a degenerate embedding and a meaningless (but expensive) API call.
- `top_k` is bounded to [1, 10]. Allowing arbitrarily large values would send enormous context windows to the completion model, inflating cost and latency.

### Error handling
- **Fail-fast** on missing API key at startup — better to crash immediately than to serve 500s on every request.
- **503** if a request arrives before ingestion completes (possible under slow network conditions).
- OpenAI API errors bubble up as **500** with a generic user-facing message; the real error is logged server-side.

### What's intentionally omitted (and why)
- **Caching:** Query-level embedding caching would speed up repeated questions, but adds statefulness that's overkill here.
- **Re-ranking:** A cross-encoder re-ranker (e.g. `ms-marco-MiniLM`) would improve precision for ambiguous queries, but requires a separate model inference step.
- **Streaming:** The completion response is collected in full before responding. Streaming would improve perceived latency but complicates the response schema.
- **Auth:** Out of scope for a local prototype.

---

## Testing

```bash
# Smoke test ingestion and retrieval
node ingest.js

# Health check
curl http://localhost:3000/health

# Password reset question (should cite authentication.md)
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the password requirements?"}'

# Cross-document question (should cite multiple files)
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I authenticate API calls and log in with SSO?", "top_k": 4}'

# Validation: empty question → 400
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": ""}'

# Validation: top_k out of range → 400
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "top_k": 99}'
```

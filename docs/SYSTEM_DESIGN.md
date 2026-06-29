# Enterprise Knowledge Assistant - System Design Document

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER LAYER                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Web UI (HTML/CSS/JS)                                    │   │
│  │  - Chat interface with drag-drop upload                  │   │
│  │  - Search mode selector (Semantic/Keyword/Hybrid)        │   │
│  │  - Toggles: Query Rewriting, Re-ranking                  │   │
│  │  - 5-star feedback per answer                            │   │
│  └──────────────────────┬───────────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────────┘
                          │ HTTP/REST
┌─────────────────────────┼───────────────────────────────────────┐
│                     API LAYER                                    │
│  ┌──────────────────────▼───────────────────────────────────┐   │
│  │  Express Server                                          │   │
│  │  - Rate limiting (100 req / 15min)                       │   │
│  │  - Input validation (length, format, XSS)                │   │
│  │  - Error handling middleware (consistent JSON errors)     │   │
│  │  Endpoints: /upload, /ask, /documents, /feedback          │   │
│  └──────┬───────────────┬─────────────────┬──────────────────┘   │
└─────────┼───────────────┼─────────────────┼──────────────────────┘
          │               │                 │
┌─────────▼──────┐ ┌──────▼──────┐ ┌───────▼──────────────────────┐
│  Document      │ │  Query      │ │  Groq Client                 │
│  Processor     │ │  Engine     │ │  (OpenAI-compatible)         │
│                │ │             │ │                              │
│  - PDF extract │ │  - Validate │ │  Embeddings: nomic-embed-text│
│  - TXT/MD read │ │  - Rewrite  │ │  Chat: llama-3.3-70b-versatile│
│  - Chunk       │ │  - Embed    │ │  - Retry w/ backoff          │
│  - Metadata    │ │  - Search   │ │  - 60s timeout               │
│                │ │  - Re-rank  │ │  - Batch embedding (100/batch)│
└───────┬────────┘ │  - Generate │ └──────────────────────────────┘
        │          │  - Cite     │                ▲
        │          └──────┬──────┘                │
        │                 │                       │
┌───────▼─────────────────▼───────────────────────┘
│                  Vector Store                         │
│  ┌──────────────────────────────────────────────┐   │
│  │  In-Memory Vector + Inverted Index            │   │
│  │                                               │   │
│  │  Semantic:  Cosine similarity (dense vectors)  │   │
│  │  Keyword:   BM25 (TF-IDF + length norm)       │   │
│  │  Hybrid:    Reciprocal Rank Fusion (RRF)      │   │
│  │                                               │   │
│  │  Metadata: {filename, page, chunkIndex, ...}   │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## 2. Data Flow

### Upload Flow
```
File (PDF/TXT/MD)
  → Multer (disk storage, 50MB limit, extension filter)
  → DocumentProcessor.extractText (pdf-parse for PDF, fs for TXT/MD)
  → Chunking (500 chars, 100 overlap, sentence-boundary aware)
  → GroqClient.embedBatch (batch of 100, nomic-embed-text, 768-dim)
  → VectorStore.addDocument (embedding + metadata + inverted index)
  → Delete temp file
  → Return: {stats: {pages, chunks, avgChunkSize, ...}}
```

### Query Flow (RAG Pipeline)
```
User Question
  → Validate (length 3-2000, sanitize XSS)
  → [Optional] Query Rewriting (LLM generates 2-3 rephrasings)
  → Embed query (nomic-embed-text)
  → Search (semantic / keyword / hybrid, retrieve top 7)
  → [Optional] Re-rank (BM25 overlap scoring, combine with semantic)
  → Select top 5
  → Confidence check (threshold: 0.3)
  → Build context block with source labels
  → Generate answer (llama-3.3-70b-versatile, temperature 0.1, system prompt with strict rules)
  → Deduplicate sources (by filename+page)
  → Compute confidence (rank-decay weighted average)
  → Update conversation memory (last 10 exchanges)
  → Return: {answer, sources, confidence, retrievalStats}
```

## 3. Component Details

### 3.1 Groq Client
- **Endpoint**: `https://api.groq.com/openai/v1` (OpenAI-compatible)
- **Authentication**: Bearer token in Authorization header
- **Retry**: 3 attempts with exponential backoff + jitter
- **Timeout**: 60 seconds per request
- **Models**: `llama-3.3-70b-versatile` (chat), `nomic-embed-text` (embeddings)

### 3.2 Vector Store
- **Dense index**: Array of `{id, text, embedding, metadata, terms, normalizedText}`
- **Sparse index**: Inverted index `term → [{docId, tf}]` for BM25
- **Cosine similarity**: Standard dot-product / (norm_a * norm_b)
- **BM25**: k1=1.5, b=0.75, IDF formula, normalized to 0-1
- **Hybrid**: RRF with k=60, configurable semantic weight (default 0.7)

### 3.3 Document Processor
- **PDF**: `pdf-parse` library, splits by form feed (`\f`) for page boundaries
- **TXT/MD**: Direct `fs.readFileSync`
- **Chunking**: Fixed-size with sentence-boundary awareness
  - Looks back 100 chars for `. `, `! `, `? `, `\n`
  - Only uses boundary if chunk remains > 50 chars
  - Overlap ensures no boundary information loss
  - Infinite loop protection via `nextStart <= start` check

### 3.4 Query Engine
- Orchestrates: validate → rewrite → embed → search → rerank → generate → cite
- **Query Rewriting**: LLM generates 2 alternative phrasings, deduplicated
- **Re-ranking**: Combines semantic score (60%) + term overlap (40%)
- **Conversation Memory**: Per-session, last 10 Q&A pairs, passed to LLM
- **Confidence**: Weighted average with 1/(1 + rank*0.5) decay

### 3.5 Evaluator
- **Faithfulness**: LLM-judge scoring (0-1), checks if answer is grounded in context
- **Relevance**: LLM-judge scoring (0-1), checks if answer addresses the question
- **Overall**: 40% faithfulness + 40% relevance + 20% retrieval score
- **Batch mode**: Runs N test cases, computes aggregate statistics

## 4. Scalability Considerations

| Component | Current | Production Path |
|-----------|---------|-----------------|
| Vector Store | In-memory array | FAISS (local) or Pinecone/Weaviate (managed) |
| File Storage | Local disk → delete after processing | S3/GCS with event-triggered processing |
| Conversation Memory | In-memory Map | Redis with TTL |
| Feedback | In-memory array | PostgreSQL / MongoDB |
| Rate Limiting | In-memory (per-process) | Redis-backed (multi-instance) |
| Server | Single Express process | Cluster mode + nginx / Kubernetes |
| Embeddings | Batch 100, sequential | Async queue (Bull/BullMQ) + parallel embedding |
| PDF Processing | Synchronous, single file | Worker queue for concurrent processing |

### Scaling Steps
1. **Short term**: Add Redis for shared state, PM2 for cluster mode
2. **Medium term**: Replace in-memory vector store with FAISS or ChromaDB
3. **Long term**: Containerize (Docker), deploy on Kubernetes, add monitoring (Prometheus + Grafana)

## 5. Security Considerations
- API key validation on startup (fails fast)
- Input sanitization (XSS stripping)
- File extension whitelist (PDF, TXT, MD only)
- File size limit (50MB)
- Question length limits (3-2000 chars)
- Rate limiting (100 requests / 15 minutes)
- No API keys or secrets in source code
- Error messages sanitized in production mode
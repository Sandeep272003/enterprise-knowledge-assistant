# Enterprise Knowledge Assistant v3.0

> Production-grade RAG system with advanced operations. Powered by **Groq** (`llama-3.3-70b-versatile`) only — embeddings computed locally with zero API calls.

## Quick Start

```bash
cd enterprise-knowledge-assistant
npm install

# Set your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY from https://console.groq.com/keys

node server.js
# Open http://localhost:3000
```

## Architecture

```
Browser UI --> Express --> QueryEngine --> GroqClient --> Groq API
              |          |               |-- VectorStore   |-- Chat: llama-3.3-70b-versatile
              |          |               |-- LocalEmbedder |-- Embeddings: local (384-dim)
              |          |               |-- DocProcessor
              |          |               |-- Evaluator
              |          |-- Multer (upload)
              |          |-- Rate Limiter
              |          |-- Error Handler
```

## RAG Pipeline (10 stages)

```
1. Validate     --> Length, format, XSS checks
2. Classify    --> factual|procedural|analytical|comparative
3. Decompose   --> Split multi-part questions into sub-queries
4. Rewrite     --> LLM generates 2-3 alternative phrasings
5. Embed       --> Local hashing-trick (384-dim, zero API)
6. Search      --> Semantic / Keyword / Hybrid (BM25+RRF)
7. Re-rank     --> 60% semantic + 40% term overlap
8. Summarize   --> Condense long context (optional)
9. Generate    --> llama-3.3-70b with strict anti-hallucination prompt
10. Verify     --> Self-check: is answer grounded in context?
11. Related    --> Generate 3 follow-up questions
```

All 11 stages are toggleable from the UI sidebar.

After answer: generate 3 related follow-up questions.

### UI Operations
| Feature | How to Use |
|---------|------------|
| Search Mode | Click Semantic/Keyword/Hybrid buttons in sidebar |
| Query Rewriting | Toggle in sidebar |
| Re-ranking | Toggle in sidebar |
| Verification | Toggle in sidebar |
| Classification | Toggle in sidebar |
| Decomposition | Toggle in sidebar |
| Related Qs | Toggle in sidebar (default: on) |
| Streaming | Toggle in sidebar (default: on) |
| Auto-Summarize | Toggle in sidebar |
| Source Filter | Dropdown in sidebar to search within one doc |
| New Session | Header button or Ctrl+Shift+N |
| Export | Header button -> Markdown/JSON/Text |
| Analytics | Header button -> rich stats modal |
| Copy Answer | Copy button under each answer |
| Regenerate | Regenerate button under each answer |
| Delete Source | X button on each source in sidebar |
| Feedback | 5-star rating under each answer |

## All Features

### Core (Required)
| Feature | Implementation |
|---------|---------------|
| Document Ingestion | PDF/TXT/MD with per-page extraction and metadata |
| Smart Chunking | 500 chars, 100 overlap, sentence-boundary aware |
| Embedding | Local hashing-trick (384-dim) | Zero API calls, zero latency |
| Semantic Search | Cosine similarity on dense vectors |
| Keyword Search | BM25 with TF-IDF and length normalization |
| Answer Generation | llama-3.3-70b-versatile, temperature 0.1 |
| Source Citation | Document + page + score + snippet per source |
| Confidence Scoring | Rank-decay weighted average |
| User Interface | Dark chat UI with drag-drop upload |
| REST API | Full JSON with consistent error format |
| Input Validation | Length, format, XSS sanitization, rate limiting |

### Advanced Operations (Bonus)
| Feature | Description | Toggle |
|---------|-------------|--------|
| **Hybrid Search** | BM25 + semantic via Reciprocal Rank Fusion | Always on |
| **Query Rewriting** | LLM generates 2-3 alternative phrasings | Sidebar toggle |
| **Re-ranking** | 60% semantic + 40% keyword overlap post-retrieval | Sidebar toggle |
| **Answer Verification** | LLM self-check: is answer grounded in sources? | Sidebar toggle |
| **Query Classification** | Classifies: factual/procedural/analytical/comparative | Sidebar toggle |
| **Query Decomposition** | Splits multi-part questions into sub-queries | Sidebar toggle |
| **Related Questions** | Suggests 3 follow-up questions after each answer | Sidebar toggle |
| **SSE Streaming** | Real-time token-by-token answer delivery | Sidebar toggle + API |
| **Context Summarization** | Condenses long context for token efficiency | Per-request |
| **Source Filtering** | Search within a specific document only | Per-request sourceFilter |
| **Conversation Memory** | Last 10 Q&A pairs for contextual follow-ups | Automatic |
| **User Feedback** | 5-star rating per answer with analytics | UI + API |
| **Batch Upload** | Upload up to 10 documents at once | POST /api/upload/batch |
| **Conversation Export** | Download as Markdown / JSON / Plain Text | UI modal + API |
| **Analytics Dashboard** | Rich modal with feedback distribution charts | UI modal + API |
| **Evaluation Metrics** | LLM-as-judge faithfulness + relevance scoring | Evaluator module |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/upload | Upload single document |
| POST | /api/upload/batch | Upload up to 10 documents |
| POST | /api/ask | Full RAG with all advanced features |
| POST | /api/ask/stream | RAG with SSE streaming |
| GET | /api/documents | List indexed documents |
| DELETE | /api/documents | Clear all data |
| DELETE | /api/documents/:filename | Delete chunks from one source |
| GET | /api/search-modes | Available search modes |
| POST | /api/feedback | Submit answer rating (1-5) |
| GET | /api/feedback | Get all feedback + average |
| GET | /api/history | Conversation history |
| GET | /api/export/:sessionId | Export as markdown |
| GET | /api/analytics | Usage analytics |
| GET | /health | System health + feature flags |

## POST /api/ask — Full Request

```json
{
  "question": "What is the leave policy?",
  "searchMode": "hybrid",
  "rewriteQuery": true,
  "rerank": true,
  "enableVerification": true,
  "enableClassification": true,
  "enableDecomposition": true,
  "enableRelated": true,
  "enableSummarization": true,
  "sourceFilter": "HR_Policy.pdf"
}
```

## POST /api/ask — Full Response

```json
{
  "answer": "Employees are eligible for 24 paid leaves annually [Source 1].",
  "sources": [{"document":"HR.pdf","page":12,"score":0.92}],
  "confidence": 0.89,
  "relatedQuestions": ["How many sick days are provided?", ...],
  "queryClassification": {"type":"factual","reasoning":"...","strategy":"..."},
  "verification": {"grounded":true,"score":0.95,"issues":[]},
  "retrievalStats": {
    "chunksSearched": 42,
    "chunksRetrieved": 5,
    "topScore": 0.92,
    "searchMode": "hybrid",
    "queryRewritten": true,
    "reranked": true,
    "queriesUsed": 3,
    "latencyMs": 2400,
    "pipelineSteps": ["classify","rewrite","rerank","verify","related(3)"],
    "sourceCoverage": 2
  }
}
```

## Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| GROQ_API_KEY | (required) | From https://console.groq.com/keys |
| PORT | 3000 | Server port |
| CHAT_MODEL | llama-3.3-70b-versatile | LLM model |
| EMBEDDING_DIMENSION | 384 | Local embedding vector dimension |
| CHUNK_SIZE | 500 | Characters per chunk |
| CHUNK_OVERLAP | 100 | Overlap between chunks |
| ENABLE_QUERY_CLASSIFICATION | true | Query type classification |
| ENABLE_ANSWER_VERIFICATION | true | Self-verify answers |
| ENABLE_RELATED_QUESTIONS | true | Suggest follow-ups |
| ENABLE_STREAMING | true | SSE streaming endpoint |
| ENABLE_SOURCE_FILTER | true | Search within specific docs |
| ENABLE_QUERY_DECOMPOSITION | true | Split multi-part questions |
| ENABLE_CONTEXT_SUMMARIZATION | true | Auto-summarize long context |
| RATE_LIMIT_MAX | 100 | Requests per 15 min |

## File Structure

```
enterprise-knowledge-assistant/
|-- .env.example              # Copy to .env and fill in
|-- .gitignore                # Git ignore rules
|-- server.js                 # Express entry point
|-- package.json
|-- config/index.js           # All configuration
|-- src/
|   |-- core/
|   |   |-- localEmbedder.js  # Hashing-trick embedder (384-dim, zero API)
|   |   |-- groqClient.js      # 10 LLM operations (chat, stream, classify, verify, ...)
|   |   |-- vectorStore.js     # Hybrid store (semantic + BM25 + RRF)
|   |   |-- documentProcessor.js # PDF/TXT extraction + smart chunking
|   |   |-- queryEngine.js     # 10-stage RAG orchestrator
|   |   |-- evaluator.js       # LLM-based RAG evaluation
|   |-- api/routes.js          # All REST endpoints (12 endpoints)
|   |-- middleware/errorHandler.js
|   |-- utils/validators.js   # Input validation
|-- public/index.html         # Advanced chat UI
|-- tests/
|   |-- run_tests.js           # 116 automated tests
|   |-- run_evaluation.js      # RAG evaluation runner
|   |-- test_cases.txt         # 30+ test cases with expected I/O
|-- docs/SYSTEM_DESIGN.md     # Architecture document
|-- evaluation/results/
|-- uploads/
|-- sample-data/
```

## Testing

```bash
# Unit tests (no API key needed)
node tests/run_tests.js

# RAG evaluation (needs API key + documents)
GROQ_API_KEY=gsk_xxx node tests/run_evaluation.js
```

## Technology Choices

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | llama-3.3-70b-versatile via Groq | 70B, strong reasoning, LPU-fast |
| Embeddings | Local hashing-trick (384-dim) | Zero API calls, zero latency |
| Search | Hybrid BM25+Semantic+RRF | Best of both worlds |
| Verification | LLM self-check | Catches hallucinations prompt can't |
| Classification | LLM-based | Optimizes retrieval per query type |

## Known Limitations

1. In-memory state (lost on restart) - production: use FAISS/ChromaDB + Redis
2. No authentication - production: add JWT
3. Scanned PDFs not supported - production: add OCR

## Future Improvements

- [ ] Persistent vector store (ChromaDB)
- [ ] Redis for sessions + rate limiting
- [ ] JWT authentication
- [ ] Docker containerization
- [x] SSE streaming in UI (EventSource)
- [x] Per-source deletion
- [x] Multi-format export (MD/JSON/TXT)
- [x] Analytics modal with feedback distribution
- [x] Session management (new session button)
- [x] Copy answer to clipboard
- [x] Regenerate last answer
- [ ] Cross-encoder re-ranking
- [ ] WebSocket real-time updates
- [ ] Multi-modal (images, tables)
"# enterprise-knowledge-assistant" 

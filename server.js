/**
 * Enterprise Knowledge Assistant v5.0 — Main Server
 *
 * Advanced RAG with:
 *   - 20+ file format support (PDF, DOCX, XLSX, PPTX, CSV, JSON, HTML, XML, YAML, etc.)
 *   - Three processing modes: Simple, Hard, Deep
 *   - SSE streaming, query classification, answer verification
 *   - Related questions, query decomposition, context summarization
 *   - Deep insights extraction (entities, relationships, metrics, patterns, gaps)
 *   - @ file reference system, batch upload, source filtering
 *   - Conversation export, analytics, 5-star feedback
 *   - Answer inspector panel, format badges, mode-specific prompts
 *
 * Only LLM model: llama-3.3-70b-versatile (embeddings are local, zero API calls)
 */

require('dotenv').config();

var express = require('express');
var rateLimit = require('express-rate-limit');
var path = require('path');
var config = require('./config');
var GroqClient = require('./src/core/groqClient');
var VectorStore = require('./src/core/vectorStore');
var QueryEngine = require('./src/core/queryEngine');
var RAGEvaluator = require('./src/core/evaluator');
var createRouter = require('./src/api/routes');
var errorHandler = require('./src/middleware/errorHandler');

if (!config.GROQ_API_KEY) {
  console.error('');
  console.error('  ERROR: GROQ_API_KEY not set.');
  console.error('  1. Copy .env.example to .env: cp .env.example .env');
  console.error('  2. Add your key: https://console.groq.com/keys');
  console.error('  3. Run: node server.js');
  console.error('');
  process.exit(1);
}

var groqClient = new GroqClient();
var vectorStore = new VectorStore();
var queryEngine = new QueryEngine(groqClient, vectorStore);
var evaluator = new RAGEvaluator(groqClient);
var conversationStore = new Map();
var feedbackStore = [];
var analyticsStore = { totalQueries: 0, successfulQueries: 0 };

var app = express();
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

var limiter = rateLimit({
  windowMs: config.RATE_LIMIT_WINDOW_MS,
  max: config.RATE_LIMIT_MAX_REQUESTS,
  message: { error: 'Too many requests. Please try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
});
app.use('/api/', limiter);

var apiRouter = createRouter(queryEngine, vectorStore, groqClient, conversationStore, feedbackStore, analyticsStore);
app.use('/api', apiRouter);

app.get('/health', function(req, res) {
  res.json({
    status: 'healthy',
    version: '5.0.0',
    model: config.CHAT_MODEL,
    embeddingModel: 'local-hashing-trick-384d',
    chunksIndexed: vectorStore.size,
    sources: vectorStore.getSources(),
    sourceStats: vectorStore.getSourceStats(),
    uptime: Math.round(process.uptime()),
    searchModes: config.SEARCH_MODES,
    processingModes: config.PROCESSING_MODES,
    supportedFormats: config.ALLOWED_EXTENSIONS,
    formatCount: config.ALLOWED_EXTENSIONS.length,
    features: {
      queryRewriting: true, reRanking: true, hybridSearch: true,
      conversationMemory: true, userFeedback: true, evaluation: true,
      streaming: config.ENABLE_STREAMING,
      queryClassification: config.ENABLE_QUERY_CLASSIFICATION,
      answerVerification: config.ENABLE_ANSWER_VERIFICATION,
      relatedQuestions: config.ENABLE_RELATED_QUESTIONS,
      queryDecomposition: config.ENABLE_QUERY_DECOMPOSITION,
      contextSummarization: true, sourceFiltering: config.ENABLE_SOURCE_FILTER,
      batchUpload: true, conversationExport: true, analytics: true,
      deepInsights: true, processingModes: true, multiFormat: true,
      fileReferenceSystem: true,
    },
  });
});

app.use(errorHandler);

app.listen(config.PORT, function() {
  console.log('');
  console.log('  ╔════════════════════════════════════════════════════════════════════╗');
  console.log('  ║   Enterprise Knowledge Assistant v4.0 — Multi-Format RAG              ║');
  console.log('  ║   Model: ' + config.CHAT_MODEL + ' (local embed)'.slice(0, 49) + '      ║');
  console.log('  ╠════════════════════════════════════════════════════════════════════╣');
  console.log('  ║  URL:       http://localhost:' + config.PORT + '                                     ║');
  console.log('  ║  Upload:    POST /api/upload  |  POST /api/upload/batch             ║');
  console.log('  ║  Ask:       POST /api/ask     |  POST /api/ask/stream (SSE)        ║');
  console.log('  ║  Export:    GET  /api/export/:sessionId (markdown)                  ║');
  console.log('  ║  Analytics: GET  /api/analytics                                     ║');
  console.log('  ║  Formats:   GET  /api/supported-formats (' + config.ALLOWED_EXTENSIONS.length + ' types)          ║');
  console.log('  ║  Modes:     GET  /api/processing-modes (simple/hard/deep)          ║');
  console.log('  ║  Health:    GET  /health                                              ║');
  console.log('  ╠════════════════════════════════════════════════════════════════════╣');
  console.log('  ║  Formats: PDF DOCX XLSX PPTX CSV TSV JSON HTML XML YAML TOML     ║');
  console.log('  ║           INI CFG CONF TXT MD RTF LOG (+20 total)                   ║');
  console.log('  ╠════════════════════════════════════════════════════════════════════╣');
  console.log('  ║  Modes:   Simple (fast) | Hard (thorough) | Deep (exhaustive)       ║');
  console.log('  ║  Features: @ file ref, classification, verification, decomposition,  ║');
  console.log('  ║           related Qs, insights extraction, context summarization     ║');
  console.log('  ╚════════════════════════════════════════════════════════════════════╝');
  console.log('');
});

module.exports = { app: app, queryEngine: queryEngine, vectorStore: vectorStore, groqClient: groqClient, evaluator: evaluator, conversationStore: conversationStore, feedbackStore: feedbackStore, analyticsStore: analyticsStore };
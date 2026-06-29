/**
 * Application Configuration
 * Loads from .env file first, then environment variables, then defaults.
 */

require('dotenv').config();

module.exports = {
  // ── Server ───────────────────────────────────────────────────
  PORT: parseInt(process.env.PORT, 10) || 3000,
  NODE_ENV: process.env.NODE_ENV || 'development',

  // ── Groq API ─────────────────────────────────────────────────
  GROQ_API_KEY: process.env.GROQ_API_KEY || '',
  GROQ_API_BASE: 'https://api.groq.com/openai/v1',

  // Model (only one LLM model used — llama-3.3-70b-versatile for all chat/LLM ops)
  CHAT_MODEL: process.env.CHAT_MODEL || 'llama-3.3-70b-versatile',

  // Embeddings: Local hashing-trick embedder (zero API calls, no external model needed)
  EMBEDDING_DIMENSION: parseInt(process.env.EMBEDDING_DIMENSION, 10) || 384,

  // ── Processing Modes ─────────────────────────────────────────
  // simple  — ~500 words, concise detailed answers
  // hard    — ~1000 words, clear & in-depth analysis
  // deep    — ~2000 words, exhaustive deep-dive with every detail
  PROCESSING_MODES: ['simple', 'hard', 'deep'],

  // Mode-specific LLM settings with word count targets
  MODE_SETTINGS: {
    simple: { temperature: 0.1, maxTokens: 800,  topK: 3,  targetWords: 500,  wordRange: [400, 600],  systemPrefix: 'Provide a detailed yet concise answer in approximately 500 words.' },
    hard:   { temperature: 0.15, maxTokens: 1600, topK: 7,  targetWords: 1000, wordRange: [800, 1200], systemPrefix: 'Provide a clear, in-depth, and well-structured analysis in approximately 1000 words.' },
    deep:   { temperature: 0.2,  maxTokens: 4096, topK: 12, targetWords: 2000, wordRange: [1800, 2500], systemPrefix: 'Provide an exhaustive, multi-perspective deep analysis with every detail from the sources in approximately 2000 words.' },
  },

  // ── Chunking ─────────────────────────────────────────────────
  CHUNK_SIZE: parseInt(process.env.CHUNK_SIZE, 10) || 500,
  CHUNK_OVERLAP: parseInt(process.env.CHUNK_OVERLAP, 10) || 100,
  MIN_CHUNK_LENGTH: 30,

  // ── Retrieval ────────────────────────────────────────────────
  TOP_K_DEFAULT: 5,
  TOP_K_MAX: 20,
  CONFIDENCE_THRESHOLD: 0.3,
  RERANK_TOP_N: 7,

  // ── Search Modes ─────────────────────────────────────────────
  SEARCH_MODES: ['semantic', 'hybrid', 'keyword'],

  // ── Upload ───────────────────────────────────────────────────
  MAX_FILE_SIZE: 50 * 1024 * 1024,
  KEEP_UPLOADED_FILES: true, // Store files in uploads dir for reliability

  // All supported file types grouped by category
  ALLOWED_EXTENSIONS: [
    // Documents
    '.pdf', '.txt', '.md',
    // Office
    '.docx', '.xlsx', '.pptx',
    // Data
    '.csv', '.tsv', '.json',
    // Web
    '.html', '.htm',
    // Code / Config
    '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    // Logs / Plain
    '.log', '.rtf',
  ],

  // Human-readable categories for UI display
  FILE_CATEGORIES: {
    'Documents':  ['.pdf', '.txt', '.md', '.rtf', '.docx'],
    'Spreadsheets': ['.xlsx', '.csv', '.tsv'],
    'Presentations': ['.pptx'],
    'Data / Config': ['.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'],
    'Web': ['.html', '.htm'],
    'Logs': ['.log'],
  },

  UPLOAD_DIR: 'uploads',

  // ── Rate Limiting ────────────────────────────────────────────
  RATE_LIMIT_WINDOW_MS: 15 * 60 * 1000,
  RATE_LIMIT_MAX_REQUESTS: parseInt(process.env.RATE_LIMIT_MAX, 10) || 100,

  // ── Conversation Memory ──────────────────────────────────────
  MAX_CONVERSATION_HISTORY: 10,

  // ── Validation ───────────────────────────────────────────────
  MAX_QUESTION_LENGTH: 2000,
  MIN_QUESTION_LENGTH: 3,

  // ── Advanced Feature Flags ────────────────────────────────────
  ENABLE_QUERY_CLASSIFICATION: process.env.ENABLE_QUERY_CLASSIFICATION !== 'false',
  ENABLE_ANSWER_VERIFICATION: process.env.ENABLE_ANSWER_VERIFICATION !== 'false',
  ENABLE_RELATED_QUESTIONS: process.env.ENABLE_RELATED_QUESTIONS !== 'false',
  ENABLE_STREAMING: process.env.ENABLE_STREAMING !== 'false',
  ENABLE_SOURCE_FILTER: process.env.ENABLE_SOURCE_FILTER !== 'false',
  ENABLE_QUERY_DECOMPOSITION: process.env.ENABLE_QUERY_DECOMPOSITION !== 'false',
};

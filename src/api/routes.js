/**
 * API Routes — Advanced v4 (Multi-Format, Mode-Aware)
 *
 * Endpoints:
 *   POST   /api/upload            Upload + index single document (20+ formats)
 *   POST   /api/upload/batch      Upload multiple documents at once
 *   POST   /api/ask               Full RAG (standard JSON)
 *   POST   /api/ask/stream        RAG with SSE streaming
 *   GET    /api/documents         List indexed documents
 *   DELETE /api/documents/:filename Delete specific source
 *   DELETE /api/documents         Clear all
 *   GET    /api/search-modes      Available search modes
 *   GET    /api/processing-modes  Available processing modes
 *   POST   /api/feedback          Submit rating
 *   GET    /api/feedback          Get all feedback
 *   GET    /api/history           Conversation history
 *   GET    /api/export/:sessionId Export conversation as markdown
 *   GET    /api/analytics         Usage analytics
 *   GET    /health                System health
 */

var express = require('express');
var multer = require('multer');
var path = require('path');
var fs = require('fs');
var config = require('../../config');
var processDocument = require('../core/documentProcessor').processDocument;
var getFormatInfo = require('../core/documentProcessor').getFormatInfo;
var v = require('../utils/validators');

function createRouter(queryEngine, vectorStore, groqClient, conversationStore, feedbackStore, analyticsStore) {
  var router = express.Router();

  // ── Upload config ────────────────────────────────────────
  var UPLOAD_DIR = path.join(process.cwd(), config.UPLOAD_DIR);
  if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

  var storage = multer.diskStorage({
    destination: UPLOAD_DIR,
    filename: function(req, file, cb) { cb(null, Date.now() + '-' + file.originalname); },
  });

  var upload = multer({
    storage: storage,
    limits: { fileSize: config.MAX_FILE_SIZE },
    fileFilter: function(req, file, cb) {
      try { v.validateFileExtension(file.originalname); cb(null, true); }
      catch (err) { cb(err); }
    },
  });

  // ═════════════════════════════════════════════════════════
  //  POST /api/upload
  // ═════════════════════════════════════════════════════════
  router.post('/upload', upload.single('file'), async function(req, res, next) {
    try {
      if (!req.file) return res.status(400).json({ error: 'No file uploaded. Use field name "file".' });

      var filePath = req.file.path;
      var originalName = req.file.originalname;
      var ext = path.extname(originalName).toLowerCase();
      console.log('\n[UPLOAD] ' + originalName + ' (' + (req.file.size / 1024).toFixed(1) + 'KB) [' + (getFormatInfo(ext) || ext) + ']');

      var result = await processDocument(filePath, originalName);
      console.log('[UPLOAD] ' + result.stats.totalPages + ' section(s) -> ' + result.stats.totalChunks + ' chunks');

      if (result.chunks.length === 0) {
        if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
        return res.status(400).json({ error: 'No text content extracted from this file.' });
      }

      console.log('[UPLOAD] Embedding ' + result.chunks.length + ' chunks...');
      var texts = result.chunks.map(function(c) { return c.text; });
      var embeddings = groqClient.embedBatch(texts);

      for (var i = 0; i < result.chunks.length; i++) {
        vectorStore.addDocument(result.chunks[i].text, embeddings[i], result.chunks[i].metadata);
      }

      // Keep uploaded files in uploads dir for reliability (don't delete)
      console.log('[UPLOAD] Done. File stored: ' + filePath + ' | Total: ' + vectorStore.size + ' chunks');

      res.json({
        success: true,
        stats: {
          filename: originalName,
          format: ext,
          formatLabel: result.stats.formatLabel,
          totalPages: result.stats.totalPages,
          chunksCreated: result.stats.totalChunks,
          totalCharacters: result.stats.totalCharacters,
          avgChunkSize: result.stats.avgChunkSize,
          totalChunksInStore: vectorStore.size,
          uniqueSources: vectorStore.getSources().length,
        },
      });
    } catch (err) { next(err); }
  });

  // ═════════════════════════════════════════════════════════
  //  POST /api/upload/batch
  // ═════════════════════════════════════════════════════════
  router.post('/upload/batch', upload.array('files', 10), async function(req, res, next) {
    try {
      if (!req.files || req.files.length === 0) {
        return res.status(400).json({ error: 'No files uploaded. Use field name "files".' });
      }

      var results = [];
      for (var fi = 0; fi < req.files.length; fi++) {
        var file = req.files[fi];
        try {
          var result = await processDocument(file.path, file.originalname);
          if (result.chunks.length === 0) {
            if (fs.existsSync(file.path)) fs.unlinkSync(file.path);
            results.push({ filename: file.originalname, success: false, error: 'No text extracted' });
            continue;
          }
          var texts = result.chunks.map(function(c) { return c.text; });
          var embeddings = groqClient.embedBatch(texts);
          for (var i = 0; i < result.chunks.length; i++) {
            vectorStore.addDocument(result.chunks[i].text, embeddings[i], result.chunks[i].metadata);
          }
          // Keep uploaded files in uploads dir for reliability
          console.log('[BATCH] ' + file.originalname + ' stored. Chunks: ' + result.stats.totalChunks);
          results.push({
            filename: file.originalname, success: true,
            format: result.stats.format, formatLabel: result.stats.formatLabel,
            chunks: result.stats.totalChunks, pages: result.stats.totalPages,
          });
        } catch (err) {
          if (fs.existsSync(file.path)) fs.unlinkSync(file.path);
          results.push({ filename: file.originalname, success: false, error: err.message });
        }
      }

      console.log('[BATCH UPLOAD] ' + results.length + ' files processed. Total: ' + vectorStore.size + ' chunks');
      res.json({ success: true, results: results, totalChunksInStore: vectorStore.size });
    } catch (err) { next(err); }
  });

  // ═════════════════════════════════════════════════════════
  //  POST /api/ask
  // ═════════════════════════════════════════════════════════
  router.post('/ask', async function(req, res, next) {
    try {
      var question = v.sanitizeInput(v.validateQuestion(req.body.question));
      var searchMode = v.validateSearchMode(req.body.searchMode);
      var processingMode = v.validateProcessingMode(req.body.processingMode);
      var rewriteQuery = v.parseBoolean(req.body.rewriteQuery, false);
      var doRerank = v.parseBoolean(req.body.rerank, false);
      var enableVerification = v.parseBoolean(req.body.enableVerification, false);
      var enableClassification = v.parseBoolean(req.body.enableClassification, false);
      var enableDecomposition = v.parseBoolean(req.body.enableDecomposition, false);
      var enableRelated = v.parseBoolean(req.body.enableRelated, false);
      var enableSummarization = v.parseBoolean(req.body.enableSummarization, false);
      var sourceFilter = req.body.sourceFilter || null;
      var sessionId = String(req.body.sessionId || 'default').slice(0, 100);

      if (vectorStore.size === 0) {
        return res.status(400).json({ error: 'No documents indexed yet. Upload documents first.' });
      }

      if (analyticsStore) analyticsStore.totalQueries++;

      console.log('\n[ASK] "' + question.substring(0, 80) + '" [' + searchMode + '/' + processingMode + ']');

      var result = await queryEngine.ask({
        question: question, searchMode: searchMode, processingMode: processingMode,
        rewriteQuery: rewriteQuery, rerank: doRerank,
        sessionId: sessionId, conversationStore: conversationStore,
        sourceFilter: sourceFilter,
        enableVerification: enableVerification,
        enableClassification: enableClassification,
        enableDecomposition: enableDecomposition,
        enableRelated: enableRelated,
        enableSummarization: enableSummarization,
      });

      if (analyticsStore) analyticsStore.successfulQueries++;
      console.log('[ASK] ' + result.retrievalStats.latencyMs + 'ms conf=' + result.confidence +
        ' mode=' + processingMode + ' pipeline=[' + result.retrievalStats.pipelineSteps.join(',') + ']');

      res.json(result);
    } catch (err) { next(err); }
  });

  // ═════════════════════════════════════════════════════════
  //  POST /api/ask/stream (SSE, MODE-AWARE)
  // ═════════════════════════════════════════════════════════
  router.post('/ask/stream', async function(req, res, next) {
    try {
      var question = v.sanitizeInput(v.validateQuestion(req.body.question));
      var searchMode = v.validateSearchMode(req.body.searchMode);
      var processingMode = v.validateProcessingMode(req.body.processingMode);
      var doRerank = v.parseBoolean(req.body.rerank, false);
      var sessionId = String(req.body.sessionId || 'default').slice(0, 100);
      var sourceFilter = req.body.sourceFilter || null;
      var enableVerification = v.parseBoolean(req.body.enableVerification, false);
      var enableClassification = v.parseBoolean(req.body.enableClassification, false);
      var enableDecomposition = v.parseBoolean(req.body.enableDecomposition, false);
      var enableRelated = v.parseBoolean(req.body.enableRelated, false);
      var enableSummarization = v.parseBoolean(req.body.enableSummarization, false);
      var rewriteQuery = v.parseBoolean(req.body.rewriteQuery, false);

      if (vectorStore.size === 0) {
        return res.status(400).json({ error: 'No documents indexed yet.' });
      }

      if (analyticsStore) analyticsStore.totalQueries++;

      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });

      var streamResult = await queryEngine.askStream({
        question: question, searchMode: searchMode, processingMode: processingMode,
        rerank: doRerank, sessionId: sessionId,
        conversationStore: conversationStore,
        sourceFilter: sourceFilter,
        enableVerification: enableVerification,
        enableClassification: enableClassification,
        enableDecomposition: enableDecomposition,
        enableRelated: enableRelated,
        enableSummarization: enableSummarization,
        rewriteQuery: rewriteQuery,
      });

      // Send metadata first (includes classification, insights, pipeline steps)
      res.write('event: metadata\n');
      res.write('data: ' + JSON.stringify(streamResult.metadata) + '\n\n');

      // Stream tokens (async generator — MUST await .next())
      var fullAnswer = '';
      var _iterator = streamResult.stream;
      var _result = await _iterator.next();

      while (!_result.done) {
        var chunk = _result.value;
        if (chunk && chunk.type === 'token') {
          fullAnswer += chunk.data;
          res.write('event: token\n');
          res.write('data: ' + JSON.stringify({ text: chunk.data }) + '\n\n');
        } else if (chunk && chunk.type === 'done') {
          // Save conversation memory
          if (conversationStore) {
            if (!conversationStore.has(sessionId)) conversationStore.set(sessionId, []);
            conversationStore.get(sessionId).push({ role: 'user', content: question });
            conversationStore.get(sessionId).push({ role: 'assistant', content: fullAnswer });
          }
          var doneData = chunk.data || {};

          // Post-stream: verification and related questions
          var postVerification = null;
          var postRelated = [];
          var isInsufficient = fullAnswer.indexOf("don't have sufficient information") !== -1;

          // Verification (all modes)
          if (streamResult._enableVerification && !isInsufficient) {
            try {
              postVerification = await groqClient.verifyAnswer(question, fullAnswer, streamResult._topResults);
              streamResult.metadata.verification = postVerification;
              streamResult.metadata.retrievalStats.pipelineSteps.push('verify');
              res.write('event: verification\n');
              res.write('data: ' + JSON.stringify(postVerification) + '\n\n');
            } catch (e) { /* non-critical */ }
          }

          // Related questions (all modes)
          if (streamResult._enableRelated && !isInsufficient && streamResult._sources) {
            try {
              postRelated = await groqClient.generateRelatedQuestions(question, fullAnswer, streamResult._sources);
              if (postRelated.length > 0) {
                streamResult.metadata.relatedQuestions = postRelated;
                streamResult.metadata.retrievalStats.pipelineSteps.push('related(' + postRelated.length + ')');
                res.write('event: related\n');
                res.write('data: ' + JSON.stringify({ questions: postRelated }) + '\n\n');
              }
            } catch (e) { /* non-critical */ }
          }

          // Send updated metadata with verification & related
          res.write('event: metadata_update\n');
          res.write('data: ' + JSON.stringify(streamResult.metadata) + '\n\n');

          if (analyticsStore) analyticsStore.successfulQueries++;

          res.write('event: done\n');
          res.write('data: ' + JSON.stringify({ answer: fullAnswer, usage: doneData.usage, mode: doneData.mode }) + '\n\n');
        } else if (chunk && chunk.type === 'error') {
          res.write('event: error\n');
          res.write('data: ' + JSON.stringify({ error: chunk.data }) + '\n\n');
        }
        _result = await _iterator.next();
      }

      res.end();
    } catch (err) {
      console.error('[STREAM ERROR] ' + err.message);
      try {
        res.write('event: error\n');
        res.write('data: ' + JSON.stringify({ error: err.message }) + '\n\n');
        res.end();
      } catch (e) { /* Response already ended */ }
    }
  });

  // ═════════════════════════════════════════════════════════
  //  GET /api/documents
  // ═════════════════════════════════════════════════════════
  router.get('/documents', function(req, res) {
    res.json({
      totalChunks: vectorStore.size,
      uniqueSources: vectorStore.getSources().length,
      sources: vectorStore.getSourceStats(),
      chunks: vectorStore.getAllDocuments(),
    });
  });

  // ═════════════════════════════════════════════════════════
  //  DELETE /api/documents/:filename
  // ═════════════════════════════════════════════════════════
  router.delete('/documents/:filename', function(req, res) {
    var filename = decodeURIComponent(req.params.filename);
    var before = vectorStore.size;
    vectorStore.removeBySource(filename);
    var removed = before - vectorStore.size;

    // Also remove the actual file from uploads dir if it exists
    var uploadDir = path.join(process.cwd(), config.UPLOAD_DIR);
    if (fs.existsSync(uploadDir)) {
      var files = fs.readdirSync(uploadDir);
      for (var i = 0; i < files.length; i++) {
        if (files[i].endsWith('-' + filename)) {
          try { fs.unlinkSync(path.join(uploadDir, files[i])); console.log('[DELETE] Removed file: ' + files[i]); } catch (e) {}
        }
      }
    }

    res.json({ success: true, removedChunks: removed, remainingChunks: vectorStore.size, filename: filename });
  });

  // ═════════════════════════════════════════════════════════
  //  DELETE /api/documents
  // ═════════════════════════════════════════════════════════
  router.delete('/documents', function(req, res) {
    var count = vectorStore.size;
    vectorStore.clear();
    conversationStore.clear();

    // Clear all uploaded files
    var uploadDir = path.join(process.cwd(), config.UPLOAD_DIR);
    if (fs.existsSync(uploadDir)) {
      var files = fs.readdirSync(uploadDir);
      for (var i = 0; i < files.length; i++) {
        try { fs.unlinkSync(path.join(uploadDir, files[i])); } catch (e) {}
      }
      console.log('[CLEAR] Removed ' + files.length + ' files from uploads dir');
    }

    res.json({ success: true, clearedChunks: count });
  });

  // ═════════════════════════════════════════════════════════
  //  GET /api/search-modes
  // ═════════════════════════════════════════════════════════
  router.get('/search-modes', function(req, res) {
    res.json({
      modes: config.SEARCH_MODES.map(function(m) {
        return {
          id: m,
          description: {
            semantic: 'Dense vector similarity — best for intent and context',
            keyword: 'BM25 term matching — best for exact keywords',
            hybrid: 'Semantic + keyword via RRF — best overall accuracy',
          }[m],
        };
      }),
      default: 'semantic',
    });
  });

  // ═════════════════════════════════════════════════════════
  //  GET /api/processing-modes (NEW)
  // ═════════════════════════════════════════════════════════
  router.get('/processing-modes', function(req, res) {
    res.json({
      modes: config.PROCESSING_MODES.map(function(m) {
        var s = config.MODE_SETTINGS[m];
        return {
          id: m,
          label: m.charAt(0).toUpperCase() + m.slice(1),
          description: {
            simple: 'Detailed concise answers (~500 words). Fast processing, low token usage.',
            hard: 'Clear, in-depth analysis (~1000 words). Best balance of depth and speed.',
            deep: 'Exhaustive multi-perspective deep-dive (~2000 words). Maximum detail with insights extraction.',
          }[m],
          maxTokens: s.maxTokens,
          topK: s.topK,
          temperature: s.temperature,
          targetWords: s.targetWords,
          wordRange: s.wordRange,
        };
      }),
      default: 'hard',
    });
  });

  // ═════════════════════════════════════════════════════════
  //  GET /api/supported-formats (NEW)
  // ═════════════════════════════════════════════════════════
  router.get('/supported-formats', function(req, res) {
    res.json({
      categories: config.FILE_CATEGORIES,
      allExtensions: config.ALLOWED_EXTENSIONS,
      totalCount: config.ALLOWED_EXTENSIONS.length,
    });
  });

  // ═════════════════════════════════════════════════════════
  //  POST /api/feedback
  // ═════════════════════════════════════════════════════════
  router.post('/feedback', function(req, res) {
    var body = req.body;
    if (!body.question || !body.answer || body.rating === undefined) {
      return res.status(400).json({ error: 'question, answer, and rating are required' });
    }
    if ([1, 2, 3, 4, 5].indexOf(Number(body.rating)) === -1) {
      return res.status(400).json({ error: 'rating must be 1-5' });
    }
    var entry = {
      id: feedbackStore.length + 1,
      question: body.question,
      answer: body.answer,
      rating: Number(body.rating),
      mode: body.mode || 'hard',
      comment: body.comment || '',
      timestamp: new Date().toISOString(),
    };
    feedbackStore.push(entry);
    res.json({ success: true, feedbackId: entry.id });
  });

  router.get('/feedback', function(req, res) {
    var avg = feedbackStore.length > 0
      ? (feedbackStore.reduce(function(s, f) { return s + f.rating; }, 0) / feedbackStore.length).toFixed(2)
      : 'N/A';
    res.json({ totalFeedback: feedbackStore.length, averageRating: avg, feedback: feedbackStore });
  });

  // ═════════════════════════════════════════════════════════
  //  GET /api/history
  // ═════════════════════════════════════════════════════════
  router.get('/history', function(req, res) {
    var sessionId = String(req.query.sessionId || 'default').slice(0, 100);
    res.json({ sessionId: sessionId, history: conversationStore.get(sessionId) || [] });
  });

  // ═════════════════════════════════════════════════════════
  //  GET /api/export/:sessionId
  // ═════════════════════════════════════════════════════════
  router.get('/export/:sessionId', function(req, res) {
    var sessionId = String(req.params.sessionId || 'default').slice(0, 100);
    var history = conversationStore.get(sessionId) || [];
    if (history.length === 0) {
      return res.status(404).json({ error: 'No conversation history for this session' });
    }

    var md = '# Conversation Export\n\n';
    md += 'Exported: ' + new Date().toISOString() + '\n';
    md += 'Session: ' + sessionId + '\n';
    md += 'Messages: ' + history.length + '\n\n---\n\n';

    for (var i = 0; i < history.length; i++) {
      var msg = history[i];
      if (msg.role === 'user') {
        md += '## User\n\n' + msg.content + '\n\n';
      } else {
        md += '## Assistant\n\n' + msg.content + '\n\n';
      }
    }

    res.setHeader('Content-Type', 'text/markdown');
    res.setHeader('Content-Disposition', 'attachment; filename="conversation-' + sessionId + '.md"');
    res.send(md);
  });

  // ═════════════════════════════════════════════════════════
  //  GET /api/analytics
  // ═════════════════════════════════════════════════════════
  router.get('/analytics', function(req, res) {
    var totalSessions = conversationStore.size;
    var totalMessages = 0;
    conversationStore.forEach(function(h) { totalMessages += h.length; });

    var avgFeedback = 'N/A';
    if (feedbackStore.length > 0) {
      avgFeedback = (feedbackStore.reduce(function(s, f) { return s + f.rating; }, 0) / feedbackStore.length).toFixed(2);
    }

    var dist = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
    feedbackStore.forEach(function(f) { dist[f.rating] = (dist[f.rating] || 0) + 1; });

    res.json({
      totalQueries: analyticsStore.totalQueries,
      successfulQueries: analyticsStore.successfulQueries,
      totalSessions: totalSessions,
      totalMessages: totalMessages,
      totalChunksIndexed: vectorStore.size,
      uniqueSources: vectorStore.getSources().length,
      sourceStats: vectorStore.getSourceStats(),
      averageFeedback: avgFeedback,
      feedbackDistribution: dist,
      totalFeedback: feedbackStore.length,
      uptime: Math.round(process.uptime()),
    });
  });

  return router;
}

module.exports = createRouter;
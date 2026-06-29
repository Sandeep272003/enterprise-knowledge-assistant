/**
 * Query Engine — Advanced RAG Orchestrator v4 (Mode-Aware)
 *
 * Pipeline stages:
 *   1. Validate question
 *   2. Determine processing mode settings (simple/hard/deep)
 *   3. [Optional] Classify query type (factual/procedural/analytical/comparative)
 *   4. [Optional] Decompose multi-part questions into sub-queries
 *   5. [Optional] Rewrite queries for better retrieval
 *   6. Embed query (local hashing-trick)
 *   7. Search (semantic / keyword / hybrid) with optional source filtering
 *   8. [Optional] Re-rank results
 *   9. [Optional] Context summarization for long retrieved content
 *   10. [Deep Mode] Extract key insights, entities, relationships
 *   11. Generate answer with context + conversation memory (mode-specific prompts)
 *   12. [Optional] Answer self-verification (grounded check)
 *   13. [Optional] Generate related follow-up questions
 *   14. Build response with full quality metrics
 */

var config = require('../../config');

class QueryEngine {
  constructor(groqClient, vectorStore) {
    this.groq = groqClient;
    this.store = vectorStore;
  }

  /**
   * Full RAG pipeline with all advanced features.
   */
  async ask(params) {
    var question = params.question;
    var searchMode = params.searchMode || 'semantic';
    var processingMode = params.processingMode || 'hard';
    var modeSettings = config.MODE_SETTINGS[processingMode] || config.MODE_SETTINGS['hard'];
    var topK = modeSettings.topK || config.TOP_K_DEFAULT;
    var rewriteQuery = params.rewriteQuery || false;
    var rerank = params.rerank || false;
    var sessionId = params.sessionId || 'default';
    var conversationStore = params.conversationStore;
    var sourceFilter = params.sourceFilter || null;
    var enableVerification = params.enableVerification !== false ? config.ENABLE_ANSWER_VERIFICATION : false;
    var enableRelated = params.enableRelated !== false ? config.ENABLE_RELATED_QUESTIONS : false;
    var enableClassification = params.enableClassification !== false ? config.ENABLE_QUERY_CLASSIFICATION : false;
    var enableDecomposition = params.enableDecomposition !== false ? config.ENABLE_QUERY_DECOMPOSITION : false;
    var enableSummarization = params.enableSummarization || false;

    var startTime = Date.now();
    var pipelineSteps = [processingMode + '-mode'];

    // ── Step 1: Validate ─────────────────────────────────
    if (question.trim().length < config.MIN_QUESTION_LENGTH) {
      throw new Error('Question too short (minimum ' + config.MIN_QUESTION_LENGTH + ' characters)');
    }
    if (question.length > config.MAX_QUESTION_LENGTH) {
      throw new Error('Question too long (maximum ' + config.MAX_QUESTION_LENGTH + ' characters)');
    }

    // ── Step 2: Classify Query (ALL modes) ───────────────
    var queryClassification = null;
    if (enableClassification) {
      try {
        queryClassification = await this.groq.classifyQuery(question);
        pipelineSteps.push('classify');
      } catch (e) { /* non-critical */ }
    }

    // ── Step 3: Query Decomposition (ALL modes) ──────────
    var subQueries = [question];
    if (enableDecomposition) {
      try {
        var decomposed = await this.groq.decomposeQuery(question);
        if (decomposed.length > 1) {
          subQueries = decomposed;
          pipelineSteps.push('decompose(' + decomposed.length + ')');
        }
      } catch (e) { /* non-critical */ }
    }

    // ── Step 4: Query Rewriting ───────────────────────────
    var allQueries = [];
    if (rewriteQuery) {
      try {
        var rewritten = await this.groq.rewriteQuery(question);
        allQueries = allQueries.concat(rewritten);
        pipelineSteps.push('rewrite');
      } catch (e) { allQueries.push(question); }
    }
    allQueries = allQueries.concat(subQueries);
    // Deduplicate
    var seenQ = {};
    var uniqueQueries = [];
    for (var qi = 0; qi < allQueries.length; qi++) {
      if (!seenQ[allQueries[qi]]) { seenQ[allQueries[qi]] = true; uniqueQueries.push(allQueries[qi]); }
    }

    // ── Step 5: Embed query ───────────────────────────────
    var queryEmbedding = this.groq.embed(question);

    // ── Step 6: Multi-query retrieval ─────────────────────
    var allResults = [];
    var seenIds = {};
    var searchTopK = Math.max(topK, config.RERANK_TOP_N);

    for (var qi2 = 0; qi2 < uniqueQueries.length; qi2++) {
      var q = uniqueQueries[qi2];
      var results = this.store.search({
        queryEmbedding: queryEmbedding,
        queryText: q,
        mode: searchMode,
        topK: searchTopK,
        semanticWeight: 0.7,
      });

      if (sourceFilter) {
        results = results.filter(function(r) { return r.metadata.filename === sourceFilter; });
      }

      for (var ri = 0; ri < results.length; ri++) {
        if (!seenIds[results[ri].id]) {
          seenIds[results[ri].id] = true;
          allResults.push(results[ri]);
        }
      }
    }

    // ── Step 7: Re-ranking ────────────────────────────────
    if (rerank && allResults.length > 1) {
      allResults = this._rerankWithLLM(question, allResults);
      pipelineSteps.push('rerank');
    }

    var topResults = allResults.slice(0, topK);

    // ── Step 8: Confidence check ─────────────────────────
    if (topResults.length === 0 || topResults[0].score < config.CONFIDENCE_THRESHOLD) {
      return this._buildResponse({
        answer: "I don't have sufficient information in the knowledge base to answer this question accurately. Please try uploading relevant documents or rephrasing your question.",
        sources: [], confidence: 0, topResults: [],
        searchMode: searchMode, processingMode: processingMode,
        rewriteQuery: rewriteQuery, rerank: rerank,
        queriesUsed: uniqueQueries.length, startTime: startTime, pipelineSteps: pipelineSteps,
        queryClassification: queryClassification, sourceFilter: sourceFilter,
      });
    }

    // ── Step 9: Context optimization ──────────────────────
    var contextChunks = topResults.map(function(r) {
      return { text: r.text, metadata: r.metadata, score: r.score };
    });

    if (enableSummarization) {
      try {
        var summary = await this.groq.summarizeContext(topResults, 4000);
        contextChunks = [{ text: summary, metadata: { filename: 'summarized', page: 0 }, score: 1.0 }];
        pipelineSteps.push('summarize');
      } catch (e) { /* fallback to full context */ }
    }

    // ── Step 10: Deep Mode — Insight Extraction ──────────
    var insights = null;
    if (processingMode === 'deep') {
      try {
        insights = await this.groq.extractInsights(question, topResults);
        pipelineSteps.push('insights');
      } catch (e) { /* non-critical */ }
    }

    // ── Step 11: Generate answer ─────────────────────────
    var history = conversationStore ? (conversationStore.get(sessionId) || []) : [];
    var genResult = await this.groq.generateAnswer(question, contextChunks, history, { mode: processingMode });
    var answer = genResult.answer;
    var model = genResult.model;
    var usage = genResult.usage;

    // ── Step 12: Answer Verification (ALL modes) ──────────
    var verification = null;
    if (enableVerification && answer.indexOf("don't have sufficient information") === -1) {
      try {
        verification = await this.groq.verifyAnswer(question, answer, topResults);
        pipelineSteps.push('verify');
      } catch (e) { /* non-critical */ }
    }

    // ── Step 13: Related Questions ───────────────────────
    var relatedQuestions = [];
    var sources = this._deduplicateSources(topResults);
    if (enableRelated && answer.indexOf("don't have sufficient information") === -1) {
      try {
        relatedQuestions = await this.groq.generateRelatedQuestions(question, answer, sources);
        if (relatedQuestions.length > 0) pipelineSteps.push('related(' + relatedQuestions.length + ')');
      } catch (e) { /* non-critical */ }
    }

    // ── Update conversation memory ──────────────────────
    if (conversationStore) {
      if (!conversationStore.has(sessionId)) conversationStore.set(sessionId, []);
      var sessionHistory = conversationStore.get(sessionId);
      sessionHistory.push({ role: 'user', content: question });
      sessionHistory.push({ role: 'assistant', content: answer });
      if (sessionHistory.length > config.MAX_CONVERSATION_HISTORY * 2) {
        conversationStore.set(sessionId, sessionHistory.slice(-config.MAX_CONVERSATION_HISTORY * 2));
      }
    }

    var confidence = this._computeConfidence(topResults);

    return this._buildResponse({
      answer: answer, sources: sources, confidence: confidence, topResults: topResults,
      searchMode: searchMode, processingMode: processingMode,
      rewriteQuery: rewriteQuery, rerank: rerank,
      queriesUsed: uniqueQueries.length, startTime: startTime, model: model, usage: usage,
      pipelineSteps: pipelineSteps, queryClassification: queryClassification,
      verification: verification, relatedQuestions: relatedQuestions,
      sourceFilter: sourceFilter, contextChunks: contextChunks, insights: insights,
    });
  }

  // ═══════════════════════════════════════════════════════════
  //  STREAMING VARIANT (FULL PIPELINE — ALL MODES)
  // ═══════════════════════════════════════════════════════════
  //  All advanced features (classification, decomposition, rewriting,
  //  verification, related Qs, summarization, insights) now work for
  //  ALL processing modes (simple/hard/deep) in streaming.

  async askStream(params) {
    var question = params.question;
    var searchMode = params.searchMode || 'semantic';
    var processingMode = params.processingMode || 'hard';
    var modeSettings = config.MODE_SETTINGS[processingMode] || config.MODE_SETTINGS['hard'];
    var topK = modeSettings.topK || config.TOP_K_DEFAULT;
    var rewriteQuery = params.rewriteQuery || false;
    var rerank = params.rerank || false;
    var sessionId = params.sessionId || 'default';
    var conversationStore = params.conversationStore;
    var sourceFilter = params.sourceFilter || null;
    var enableVerification = params.enableVerification !== false ? config.ENABLE_ANSWER_VERIFICATION : false;
    var enableRelated = params.enableRelated !== false ? config.ENABLE_RELATED_QUESTIONS : false;
    var enableClassification = params.enableClassification !== false ? config.ENABLE_QUERY_CLASSIFICATION : false;
    var enableDecomposition = params.enableDecomposition !== false ? config.ENABLE_QUERY_DECOMPOSITION : false;
    var enableSummarization = params.enableSummarization || false;

    var startTime = Date.now();
    var pipelineSteps = [processingMode + '-mode', 'stream'];

    // ── Step 1: Classify Query (ALL modes) ───────────────
    var queryClassification = null;
    if (enableClassification) {
      try {
        queryClassification = await this.groq.classifyQuery(question);
        pipelineSteps.push('classify');
      } catch (e) { /* non-critical */ }
    }

    // ── Step 2: Query Decomposition (ALL modes) ──────────
    var subQueries = [question];
    if (enableDecomposition) {
      try {
        var decomposed = await this.groq.decomposeQuery(question);
        if (decomposed.length > 1) {
          subQueries = decomposed;
          pipelineSteps.push('decompose(' + decomposed.length + ')');
        }
      } catch (e) { /* non-critical */ }
    }

    // ── Step 3: Query Rewriting ──────────────────────────
    var allQueries = [];
    if (rewriteQuery) {
      try {
        var rewritten = await this.groq.rewriteQuery(question);
        allQueries = allQueries.concat(rewritten);
        pipelineSteps.push('rewrite');
      } catch (e) { allQueries.push(question); }
    }
    allQueries = allQueries.concat(subQueries);
    // Deduplicate
    var seenQ = {};
    var uniqueQueries = [];
    for (var qi = 0; qi < allQueries.length; qi++) {
      if (!seenQ[allQueries[qi]]) { seenQ[allQueries[qi]] = true; uniqueQueries.push(allQueries[qi]); }
    }

    // ── Step 4: Embed query ──────────────────────────────
    var queryEmbedding = this.groq.embed(question);

    // ── Step 5: Multi-query retrieval ────────────────────
    var allResults = [];
    var seenIds = {};
    var searchTopK = Math.max(topK, config.RERANK_TOP_N);

    for (var qi2 = 0; qi2 < uniqueQueries.length; qi2++) {
      var q = uniqueQueries[qi2];
      var results = this.store.search({
        queryEmbedding: queryEmbedding, queryText: q,
        mode: searchMode, topK: searchTopK, semanticWeight: 0.7,
      });
      if (sourceFilter) {
        results = results.filter(function(r) { return r.metadata.filename === sourceFilter; });
      }
      for (var ri = 0; ri < results.length; ri++) {
        if (!seenIds[results[ri].id]) {
          seenIds[results[ri].id] = true;
          allResults.push(results[ri]);
        }
      }
    }

    // ── Step 6: Re-ranking ───────────────────────────────
    if (rerank && allResults.length > 1) {
      allResults = this._rerankWithLLM(question, allResults);
      pipelineSteps.push('rerank');
    }

    var topResults = allResults.slice(0, topK);

    // ── Step 7: Confidence check ─────────────────────────
    if (topResults.length === 0 || topResults[0].score < config.CONFIDENCE_THRESHOLD) {
      return {
        metadata: this._buildResponse({
          answer: "I don't have sufficient information in the knowledge base to answer this question accurately. Please try uploading relevant documents or rephrasing your question.",
          sources: [], confidence: 0, topResults: [], searchMode: searchMode,
          processingMode: processingMode, rewriteQuery: rewriteQuery, rerank: rerank,
          queriesUsed: uniqueQueries.length, startTime: startTime, pipelineSteps: pipelineSteps,
          queryClassification: queryClassification, sourceFilter: sourceFilter,
        }),
        stream: (function() { return (function* () { yield { type: 'done', data: {} }; })(); })(),
      };
    }

    // ── Step 8: Context optimization ─────────────────────
    var contextChunks = topResults.map(function(r) {
      return { text: r.text, metadata: r.metadata, score: r.score };
    });

    if (enableSummarization) {
      try {
        var summary = await this.groq.summarizeContext(topResults, 4000);
        contextChunks = [{ text: summary, metadata: { filename: 'summarized', page: 0 }, score: 1.0 }];
        pipelineSteps.push('summarize');
      } catch (e) { /* fallback to full context */ }
    }

    var history = conversationStore ? (conversationStore.get(sessionId) || []) : [];

    // ── Step 9: Deep Mode — Insight Extraction ───────────
    var insights = null;
    if (processingMode === 'deep') {
      try { insights = await this.groq.extractInsights(question, topResults); pipelineSteps.push('insights'); } catch (e) {}
    }

    // ── Step 10: Stream answer ───────────────────────────
    var stream = this.groq.streamAnswer(question, contextChunks, history, { mode: processingMode });
    var sources = this._deduplicateSources(topResults);
    var confidence = this._computeConfidence(topResults);

    return {
      metadata: this._buildResponse({
        answer: '', sources: sources, confidence: confidence,
        topResults: topResults, searchMode: searchMode,
        processingMode: processingMode, rewriteQuery: rewriteQuery, rerank: rerank,
        queriesUsed: uniqueQueries.length, startTime: startTime, pipelineSteps: pipelineSteps,
        insights: insights, queryClassification: queryClassification, sourceFilter: sourceFilter,
      }),
      stream: stream,
      _topResults: topResults,
      _question: question,
      _history: history,
      _conversationStore: conversationStore,
      _sessionId: sessionId,
      _enableVerification: enableVerification,
      _enableRelated: enableRelated,
      _processingMode: processingMode,
      _sources: sources,
    };
  }

  // ═══════════════════════════════════════════════════════════
  //  INTERNAL HELPERS
  // ═══════════════════════════════════════════════════════════

  _buildResponse(opts) {
    return {
      answer: opts.answer,
      sources: opts.sources || [],
      confidence: opts.confidence || 0,
      relatedQuestions: opts.relatedQuestions || [],
      queryClassification: opts.queryClassification || null,
      verification: opts.verification || null,
      insights: opts.insights || null,
      processingMode: opts.processingMode || 'hard',
      retrievalStats: {
        chunksSearched: this.store.size,
        chunksRetrieved: (opts.topResults || []).length,
        topScore: (opts.topResults || [])[0] ? (opts.topResults[0].score || 0) : 0,
        sourceCoverage: this._sourceCoverage(opts.sources || []),
        searchMode: opts.searchMode,
        processingMode: opts.processingMode || 'hard',
        queryRewritten: !!opts.rewriteQuery,
        reranked: !!opts.rerank,
        queriesUsed: opts.queriesUsed || 1,
        model: opts.model || null,
        latencyMs: Date.now() - opts.startTime,
        pipelineSteps: opts.pipelineSteps || [],
        sourceFilter: opts.sourceFilter || null,
        usage: opts.usage || null,
      },
    };
  }

  _sourceCoverage(sources) { return sources.length; }

  _rerankWithLLM(query, results) {
    var queryTerms = {};
    query.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).forEach(function(t) {
      if (t.length > 2) queryTerms[t] = true;
    });
    var termCount = Object.keys(queryTerms).length;
    return results.map(function(r) {
      var overlapCount = 0;
      var docTerms = r.terms || [];
      for (var i = 0; i < docTerms.length; i++) { if (queryTerms[docTerms[i]]) overlapCount++; }
      var termOverlap = termCount > 0 ? overlapCount / termCount : 0;
      var rerankScore = r.score * 0.6 + termOverlap * 0.4;
      var out = {}; for (var k in r) out[k] = r[k]; out.score = rerankScore; out.rerankScore = rerankScore;
      return out;
    }).sort(function(a, b) { return b.score - a.score; });
  }

  _deduplicateSources(results) {
    var map = {};
    for (var i = 0; i < results.length; i++) {
      var r = results[i];
      var key = r.metadata.filename + '#page' + r.metadata.page;
      if (!map[key] || r.score > map[key].score) {
        map[key] = {
          document: r.metadata.filename, page: r.metadata.page,
          score: parseFloat(r.score.toFixed(4)),
          snippet: r.text.substring(0, 200).replace(/\n/g, ' ') + '...',
          chunkIndex: r.metadata.chunkIndex,
          format: r.metadata.format || '',
        };
      }
    }
    var arr = []; for (var k2 in map) arr.push(map[k2]);
    return arr;
  }

  _computeConfidence(results) {
    if (!results.length) return 0;
    var weightedSum = 0, weightTotal = 0;
    for (var i = 0; i < results.length; i++) {
      var w = 1 / (1 + i * 0.5);
      weightedSum += results[i].score * w;
      weightTotal += w;
    }
    return parseFloat((weightedSum / weightTotal).toFixed(4));
  }
}

module.exports = QueryEngine;
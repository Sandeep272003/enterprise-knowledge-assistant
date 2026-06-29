/**
 * Advanced Vector Store
 * 
 * Supports three search modes:
 * 1. Semantic Search: Cosine similarity on dense embeddings
 * 2. Keyword Search (BM25-like): TF-IDF scoring with term frequency
 * 3. Hybrid Search: Weighted combination of semantic + keyword scores
 * 
 * Design Decisions:
 * - In-memory for prototype simplicity (swap to FAISS/ChromaDB for production)
 * - Inverted index for O(1) term lookup in keyword mode
 * - Normalized scores for cross-mode comparison
 * - Re-ranking support via external scorer function
 * 
 * Why Hybrid Search?
 *   Semantic search excels at understanding intent but misses exact keyword matches.
 *   BM25 excels at exact term matching but misses paraphrased queries.
 *   Hybrid combines both: RRF (Reciprocal Rank Fusion) for robust score merging.
 */

class VectorStore {
  constructor() {
    this.documents = [];
    this.idCounter = 0;
    this.invertedIndex = {};  // term -> [{docId, tf}]
    this.docTermCounts = {};  // docId -> total terms
    this.docCount = 0;
    this.avgDocLength = 0;
  }

  // ── Document Management ─────────────────────────────────────────────

  addDocument(text, embedding, metadata = {}) {
    const doc = {
      id: ++this.idCounter,
      text,
      embedding,
      metadata,
      normalizedText: text.toLowerCase(),
      terms: this._tokenize(text),
    };
    this.documents.push(doc);
    this._addToInvertedIndex(doc);
    this._updateStats();
    return doc;
  }

  // ── Search Methods ──────────────────────────────────────────────────

  /**
   * Unified search interface
   * @param {object} options
   * @param {number[]} options.queryEmbedding - For semantic/hybrid
   * @param {string} options.queryText - For keyword/hybrid
   * @param {string} options.mode - 'semantic' | 'keyword' | 'hybrid'
   * @param {number} options.topK - Number of results
   * @param {number} options.semanticWeight - Weight for semantic in hybrid (0-1)
   * @returns {Array} scored results
   */
  search(options) {
    const {
      queryEmbedding,
      queryText,
      mode = 'semantic',
      topK = 5,
      semanticWeight = 0.7,
    } = options;

    let results;

    switch (mode) {
      case 'keyword':
        results = this._keywordSearch(queryText, topK * 2);
        break;
      case 'hybrid':
        results = this._hybridSearch(queryEmbedding, queryText, topK * 2, semanticWeight);
        break;
      case 'semantic':
      default:
        results = this._semanticSearch(queryEmbedding, topK * 2);
        break;
    }

    return results.slice(0, topK);
  }

  /**
   * Re-rank results using an external scoring function
   * @param {Array} results - From search()
   * @param {string} query - Original question
   * @param {Function} scorer - (query, docText) => score
   * @returns {Array} Re-ordered results
   */
  rerank(results, query, scorer) {
    return results
      .map((r) => ({ ...r, rerankScore: scorer(query, r.text) }))
      .sort((a, b) => (b.rerankScore || 0) - (a.rerankScore || 0));
  }

  // ── Semantic Search (Cosine Similarity) ─────────────────────────────

  _semanticSearch(queryEmbedding, topK) {
    const scored = this.documents.map((doc) => ({
      ...doc,
      score: this._cosineSimilarity(queryEmbedding, doc.embedding),
      searchMode: 'semantic',
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  // ── Keyword Search (BM25-style) ─────────────────────────────────────

  _keywordSearch(queryText, topK) {
    const queryTerms = this._tokenize(queryText);
    const k1 = 1.5; // BM25 term saturation
    const b = 0.75; // BM25 length normalization

    const scored = this.documents.map((doc) => {
      let bm25Score = 0;
      for (const term of queryTerms) {
        const postings = this.invertedIndex[term] || [];
        const docPosting = postings.find((p) => p.docId === doc.id);
        if (!docPosting) continue;

        const tf = docPosting.tf;
        const df = postings.length;
        const idf = Math.log((this.docCount - df + 0.5) / (df + 0.5) + 1);
        const docLen = doc.terms.length || 1;
        const tfNorm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLen / this.avgDocLength)));
        bm25Score += idf * tfNorm;
      }

      return { ...doc, score: bm25Score, searchMode: 'keyword' };
    });

    scored.sort((a, b) => b.score - a.score);
    // Normalize to 0-1
    const maxScore = scored[0]?.score || 1;
    scored.forEach((s) => { s.score = maxScore > 0 ? s.score / maxScore : 0; });
    return scored.slice(0, topK);
  }

  // ── Hybrid Search (RRF - Reciprocal Rank Fusion) ────────────────────

  _hybridSearch(queryEmbedding, queryText, topK, semanticWeight) {
    const semanticResults = this._semanticSearch(queryEmbedding, topK * 2);
    const keywordResults = this._keywordSearch(queryText, topK * 2);

    // Reciprocal Rank Fusion
    const k = 60; // RRF constant (standard value)
    const rrfScores = new Map();

    const addRRF = (results, weight) => {
      results.forEach((r, rank) => {
        const rrfScore = weight / (k + rank + 1);
        const existing = rrfScores.get(r.id) || { doc: r, score: 0 };
        rrfScores.set(r.id, { doc: r, score: existing.score + rrfScore });
      });
    };

    addRRF(semanticResults, semanticWeight);
    addRRF(keywordResults, 1 - semanticWeight);

    const merged = Array.from(rrfScores.values());
    merged.sort((a, b) => b.score - a.score);

    // Normalize scores
    const maxScore = merged[0]?.score || 1;
    return merged.slice(0, topK).map((m) => ({
      ...m.doc,
      score: maxScore > 0 ? m.score / maxScore : 0,
      searchMode: 'hybrid',
    }));
  }

  // ── Helpers ─────────────────────────────────────────────────────────

  _cosineSimilarity(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
  }

  _tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter((t) => t.length > 1);
  }

  _addToInvertedIndex(doc) {
    const termFreq = {};
    for (const term of doc.terms) {
      termFreq[term] = (termFreq[term] || 0) + 1;
    }
    this.docTermCounts[doc.id] = doc.terms.length;

    for (const [term, tf] of Object.entries(termFreq)) {
      if (!this.invertedIndex[term]) this.invertedIndex[term] = [];
      this.invertedIndex[term].push({ docId: doc.id, tf });
    }
  }

  _updateStats() {
    this.docCount = this.documents.length;
    const totalTerms = Object.values(this.docTermCounts).reduce((a, b) => a + b, 0);
    this.avgDocLength = this.docCount > 0 ? totalTerms / this.docCount : 0;
  }

  // ── Public Accessors ────────────────────────────────────────────────

  getAllDocuments() {
    return this.documents.map((d) => ({
      id: d.id,
      filename: d.metadata.filename || 'unknown',
      chunkIndex: d.metadata.chunkIndex,
      page: d.metadata.page,
      textPreview: d.text.substring(0, 150) + '...',
      termCount: d.terms.length,
    }));
  }

  get size() {
    return this.documents.length;
  }

  clear() {
    this.documents = [];
    this.idCounter = 0;
    this.invertedIndex = {};
    this.docTermCounts = {};
    this.docCount = 0;
    this.avgDocLength = 0;
  }

  /**
   * Remove all documents from a specific source file.
   * Rebuilds inverted index after removal.
   */
  removeBySource(filename) {
    const keep = [];
    for (const doc of this.documents) {
      if (doc.metadata.filename !== filename) keep.push(doc);
    }
    this.documents = keep;
    // Rebuild inverted index
    this.invertedIndex = {};
    this.docTermCounts = {};
    for (const doc of this.documents) {
      this._addToInvertedIndex(doc);
    }
    this._updateStats();
  }

  getSources() {
    return [...new Set(this.documents.map((d) => d.metadata.filename).filter(Boolean))];
  }

  getSourceStats() {
    const stats = {};
    for (const doc of this.documents) {
      const name = doc.metadata.filename || 'unknown';
      if (!stats[name]) stats[name] = { chunks: 0, pages: new Set() };
      stats[name].chunks++;
      if (doc.metadata.page) stats[name].pages.add(doc.metadata.page);
    }
    return Object.entries(stats).map(([name, s]) => ({
      filename: name,
      chunks: s.chunks,
      pages: s.pages.size,
    }));
  }
}

module.exports = VectorStore;
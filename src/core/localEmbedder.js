/**
 * Local Embedding Engine — Zero API Calls
 *
 * Uses the "hashing trick" (feature hashing) to map arbitrary text
 * into fixed-dimension dense vectors suitable for cosine similarity.
 *
 * How it works:
 *   1. Tokenize text into lowercase words
 *   2. Hash each token to multiple positions in a fixed-size vector
 *   3. Also hash character n-grams (3-5 chars) for partial word matching
 *   4. Weight by TF (term frequency) to normalize for text length
 *   5. L2-normalize the final vector
 *
 * Why this works for RAG:
 *   - Same concept → similar tokens → overlapping hash positions → high cosine
 *   - Different concepts → different tokens → sparse overlap → low cosine
 *   - Character n-grams catch morphological similarity (e.g., "policies" ≈ "policy")
 *   - No vocabulary needed — works on any text immediately
 *   - Deterministic: same input always produces same embedding
 *   - 384 dimensions: compact but expressive enough for retrieval
 *
 * Tradeoffs vs API embeddings:
 *   + Zero latency, zero cost, no rate limits, works offline
 *   + No dependency on external embedding model availability
 *   - Less semantically aware than neural embeddings (doesn't know "car" ≈ "automobile")
 *   - Best paired with BM25 keyword search (hybrid) to compensate
 *
 * This is the same technique used in production ML systems like
 * Vowpal Wabbit and scikit-learn's HashingVectorizer.
 */

class LocalEmbedder {
  /**
   * @param {object} options
   * @param {number} options.dimension - Vector dimension (default 384)
   * @param {number} options.wordHashSeeds - Number of hash positions per word (default 4)
   * @param {number[]} options.ngramRange - Character n-gram sizes (default [3, 4, 5])
   */
  constructor(options) {
    var opts = options || {};
    this.dimension = opts.dimension || 384;
    this.wordHashSeeds = opts.wordHashSeeds || 4;
    this.ngramRange = opts.ngramRange || [3, 4, 5];
  }

  /**
   * Embed a single text string into a dense vector.
   * @param {string} text
   * @returns {number[]} Normalized embedding vector
   */
  embed(text) {
    var tokens = this._tokenize(text);
    var vector = new Float32Array(this.dimension);
    var tokenCount = tokens.length || 1;

    for (var t = 0; t < tokens.length; t++) {
      var token = tokens[t];

      // Word-level hashing: map each word to multiple vector positions
      for (var s = 0; s < this.wordHashSeeds; s++) {
        var idx = this._hashString(token, s) % this.dimension;
        vector[idx] += 1.0 / tokenCount;
      }

      // Character n-gram hashing: catch partial word matches
      for (var n = 0; n < this.ngramRange.length; n++) {
        var ngramLen = this.ngramRange[n];
        for (var i = 0; i <= token.length - ngramLen; i++) {
          var ngram = token.substring(i, i + ngramLen);
          var idx2 = this._hashString(ngram, this.wordHashSeeds + n) % this.dimension;
          vector[idx2] += 0.3 / tokenCount;
        }
      }
    }

    // L2 normalize
    return this._normalize(vector);
  }

  /**
   * Embed multiple texts at once.
   * @param {string[]} texts
   * @returns {number[][]} Array of embedding vectors
   */
  embedBatch(texts) {
    var results = [];
    for (var i = 0; i < texts.length; i++) {
      results.push(this.embed(texts[i]));
    }
    return results;
  }

  // ── Internal helpers ─────────────────────────────────────

  /**
   * Deterministic string hash (djb2 variant with seed).
   * Returns a positive integer.
   */
  _hashString(str, seed) {
    var hash = (seed || 0) * 37;
    for (var i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
    }
    return Math.abs(hash);
  }

  /**
   * Tokenize: lowercase, strip non-alphanumeric, split on whitespace,
   * filter short tokens (1 char).
   */
  _tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(function(t) { return t.length > 1; });
  }

  /**
   * L2-normalize a Float32Array and convert to regular array.
   */
  _normalize(vector) {
    var norm = 0;
    for (var i = 0; i < vector.length; i++) {
      norm += vector[i] * vector[i];
    }
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (var j = 0; j < vector.length; j++) {
        vector[j] /= norm;
      }
    }
    return Array.from(vector);
  }
}

module.exports = LocalEmbedder;
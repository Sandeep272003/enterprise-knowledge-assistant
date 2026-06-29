/**
 * RAG Evaluator
 * 
 * Provides automated evaluation metrics for the RAG system:
 * - Retrieval Precision: What % of retrieved chunks are relevant?
 * - Retrieval Recall: What % of all relevant chunks were retrieved?
 * - Answer Relevance: Does the answer address the question?
 * - Faithfulness: Is the answer grounded in the retrieved context?
 * 
 * Uses LLM-as-judge for faithfulness and relevance scoring
 * (industry standard when gold labels are unavailable).
 */

const config = require('../../config');

class RAGEvaluator {
  /**
   * @param {object} groqClient - GroqClient instance
   */
  constructor(groqClient) {
    this.groq = groqClient;
  }

  /**
   * Evaluate a single RAG interaction
   * @param {object} params
   * @param {string} params.question
   * @param {string} params.answer
   * @param {Array} params.retrievedChunks - [{text, metadata, score}]
   * @param {Array} params.relevantDocIds - [optional] Ground truth doc IDs for precision/recall
   * @returns {Promise<object>} Evaluation metrics
   */
  async evaluate({ question, answer, retrievedChunks, relevantDocIds = [] }) {
    const metrics = {};

    // ── Retrieval Metrics ──────────────────────────────────────
    metrics.retrievalCount = retrievedChunks.length;
    metrics.topScore = retrievedChunks[0]?.score || 0;
    metrics.avgScore =
      retrievedChunks.length > 0
        ? retrievedChunks.reduce((s, c) => s + c.score, 0) / retrievedChunks.length
        : 0;
    metrics.scoreVariance =
      retrievedChunks.length > 1
        ? this._variance(retrievedChunks.map((c) => c.score))
        : 0;

    // If ground truth available, compute precision/recall
    if (relevantDocIds.length > 0) {
      const retrievedIds = new Set(retrievedChunks.map((c) => c.metadata.filename));
      const relevantIds = new Set(relevantDocIds);
      const truePositives = [...retrievedIds].filter((id) => relevantIds.has(id)).length;

      metrics.precision = retrievedIds.size > 0 ? truePositives / retrievedIds.size : 0;
      metrics.recall = relevantIds.size > 0 ? truePositives / relevantIds.size : 0;
      metrics.f1 =
        metrics.precision + metrics.recall > 0
          ? (2 * metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
          : 0;
    }

    // ── LLM-Based Metrics ─────────────────────────────────────
    const faithfulness = await this._judgeFaithfulness(answer, retrievedChunks);
    const relevance = await this._judgeRelevance(question, answer);

    metrics.faithfulness = faithfulness.score;
    metrics.faithfulnessExplanation = faithfulness.explanation;
    metrics.answerRelevance = relevance.score;
    metrics.relevanceExplanation = relevance.explanation;

    // ── Aggregate ──────────────────────────────────────────────
    metrics.overallScore = parseFloat(
      ((metrics.faithfulness * 0.4 + metrics.answerRelevance * 0.4 + (metrics.topScore || 0) * 0.2) * 100).toFixed(1)
    );

    return metrics;
  }

  /**
   * Run a batch of test cases and return aggregate metrics
   * @param {Array<{question, expectedAnswer, relevantDocs}>} testCases
   * @param {Function} askFn - async (question) => full RAG response
   * @returns {Promise<{aggregate: object, results: Array}>}
   */
  async runEvaluation(testCases, askFn) {
    const results = [];
    const startTime = Date.now();

    for (const testCase of testCases) {
      try {
        const response = await askFn(testCase.question);
        const evaluation = await this.evaluate({
          question: testCase.question,
          answer: response.answer,
          retrievedChunks: response._retrievedChunks || [],
          relevantDocIds: testCase.relevantDocs || [],
        });

        results.push({
          testCase: {
            question: testCase.question,
            expectedAnswer: testCase.expectedAnswer || 'N/A',
          },
          actualAnswer: response.answer,
          confidence: response.confidence,
          evaluation,
        });
      } catch (error) {
        results.push({
          testCase: { question: testCase.question },
          error: error.message,
          evaluation: { overallScore: 0 },
        });
      }
    }

    // Aggregate
    const valid = results.filter((r) => !r.error);
    const aggregate = {
      totalTests: results.length,
      passed: valid.filter((r) => r.evaluation.overallScore >= 60).length,
      failed: results.length - valid.filter((r) => r.evaluation.overallScore >= 60).length,
      avgOverallScore: valid.length > 0
        ? (valid.reduce((s, r) => s + r.evaluation.overallScore, 0) / valid.length).toFixed(1)
        : 0,
      avgFaithfulness: valid.length > 0
        ? (valid.reduce((s, r) => s + r.evaluation.faithfulness, 0) / valid.length).toFixed(2)
        : 0,
      avgRelevance: valid.length > 0
        ? (valid.reduce((s, r) => s + r.evaluation.answerRelevance, 0) / valid.length).toFixed(2)
        : 0,
      avgConfidence: valid.length > 0
        ? (valid.reduce((s, r) => s + r.confidence, 0) / valid.length).toFixed(2)
        : 0,
      totalLatencyMs: Date.now() - startTime,
    };

    return { aggregate, results };
  }

  // ── Private LLM Judges ───────────────────────────────────────────

  async _judgeFaithfulness(answer, contextChunks) {
    const contextText = contextChunks.map((c) => c.text).join('\n\n');

    const prompt = `You are evaluating whether an AI answer is faithful to the provided context.

CONTEXT:
${contextText}

ANSWER:
${answer}

Rate faithfulness on a scale of 0 to 1:
- 1.0 = Answer is fully grounded in the context, no external information
- 0.5 = Answer mixes context with some external information
- 0.0 = Answer is completely ungrounded or hallucinated

Respond in EXACTLY this JSON format (no other text):
{"score": 0.0-1.0, "explanation": "brief explanation"}`;

    try {
      const response = await this.groq._request(`${config.GROQ_API_BASE}/chat/completions`, {
        model: config.CHAT_MODEL,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.0,
        max_tokens: 200,
      });

      const content = response.choices[0].message.content.trim();
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return {
          score: Math.max(0, Math.min(1, parseFloat(parsed.score) || 0)),
          explanation: parsed.explanation || '',
        };
      }
    } catch (e) {
      // Fallback
    }
    return { score: 0.5, explanation: 'Evaluation failed' };
  }

  async _judgeRelevance(question, answer) {
    const prompt = `You are evaluating whether an AI answer is relevant to a user question.

QUESTION:
${question}

ANSWER:
${answer}

Rate relevance on a scale of 0 to 1:
- 1.0 = Answer directly and comprehensively addresses the question
- 0.5 = Answer partially addresses the question
- 0.0 = Answer is irrelevant or does not address the question

Respond in EXACTLY this JSON format (no other text):
{"score": 0.0-1.0, "explanation": "brief explanation"}`;

    try {
      const response = await this.groq._request(`${config.GROQ_API_BASE}/chat/completions`, {
        model: config.CHAT_MODEL,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.0,
        max_tokens: 200,
      });

      const content = response.choices[0].message.content.trim();
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return {
          score: Math.max(0, Math.min(1, parseFloat(parsed.score) || 0)),
          explanation: parsed.explanation || '',
        };
      }
    } catch (e) {
      // Fallback
    }
    return { score: 0.5, explanation: 'Evaluation failed' };
  }

  _variance(numbers) {
    if (numbers.length === 0) return 0;
    const mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
    return numbers.reduce((s, n) => s + Math.pow(n - mean, 2), 0) / numbers.length;
  }
}

module.exports = RAGEvaluator;
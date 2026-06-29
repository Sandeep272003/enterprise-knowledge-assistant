/**
 * RAG Evaluation Runner
 * 
 * Runs automated evaluation on the RAG system using LLM-as-judge.
 * Requires: GROQ_API_KEY and uploaded documents.
 * 
 * Usage: GROQ_API_KEY=xxx node tests/run_evaluation.js
 */

const config = require('../config');
const GroqClient = require('../src/core/groqClient');
const VectorStore = require('../src/core/vectorStore');
const QueryEngine = require('../src/core/queryEngine');
const RAGEvaluator = require('../src/core/evaluator');
const fs = require('fs');
const path = require('path');

async function main() {
  if (!config.GROQ_API_KEY) {
    console.error('ERROR: GROQ_API_KEY required. Run: GROQ_API_KEY=xxx node tests/run_evaluation.js');
    process.exit(1);
  }

  console.log('═══ RAG Evaluation Runner ═══\n');
  console.log(`Model: ${config.CHAT_MODEL}`);
  console.log(`Embedding: ${config.EMBEDDING_MODEL}\n`);

  const groq = new GroqClient();
  const store = new VectorStore();
  const engine = new QueryEngine(groq, store);
  const evaluator = new RAGEvaluator(groq);
  const conversationStore = new Map();

  // Upload test documents if available
  const sampleDir = path.join(__dirname, '..', 'sample-data');
  if (fs.existsSync(sampleDir)) {
    const files = fs.readdirSync(sampleDir).filter(f => f.endsWith('.pdf') || f.endsWith('.txt') || f.endsWith('.md'));
    for (const file of files) {
      console.log(`Loading sample: ${file}`);
      const { processDocument } = require('../src/core/documentProcessor');
      const { chunks, stats } = await processDocument(path.join(sampleDir, file), file);
      const texts = chunks.map(c => c.text);
      const embeddings = await groq.embedBatch(texts);
      for (let i = 0; i < chunks.length; i++) {
        store.addDocument(chunks[i].text, embeddings[i], chunks[i].metadata);
      }
      console.log(`  → ${stats.totalChunks} chunks indexed`);
    }
  }

  if (store.size === 0) {
    console.log('WARNING: No sample documents found. Upload documents via the API first, or add files to sample-data/');
    console.log('Creating a minimal test document for evaluation...\n');

    // Create minimal test data
    const { processDocument } = require('../src/core/documentProcessor');
    const testContent = `Employee Leave Policy
The company provides 24 days of paid annual leave to all full-time employees. Part-time employees receive leave proportional to their working hours. Leave must be requested at least 2 weeks in advance for absences longer than 3 days. Unused leave may be carried over up to 5 days into the next year.

Customer Refund Policy
Customers are eligible for a full refund within 30 days of purchase. Refunds are processed within 5-7 business days. Items must be in original packaging. Digital products are non-refundable after 14 days. Shipping costs are non-refundable.

Technical Setup Guide
The API v2 requires an API key from the developer portal. Base URL is https://api.company.com/v2. Authentication uses Bearer token. Rate limits: 100 requests/minute for standard tier, 1000 for premium. Webhook callbacks must respond within 5 seconds.`;

    const testFile = '/tmp/eka_test_docs.txt';
    fs.writeFileSync(testFile, testContent);
    const { chunks, stats } = await processDocument(testFile, 'test_docs.txt');
    const texts = chunks.map(c => c.text);
    const embeddings = await groq.embedBatch(texts);
    for (let i = 0; i < chunks.length; i++) {
      store.addDocument(chunks[i].text, embeddings[i], chunks[i].metadata);
    }
    fs.unlinkSync(testFile);
    console.log(`Test document: ${stats.totalChunks} chunks indexed\n`);
  }

  // Define test cases
  const testCases = [
    {
      question: 'What is the employee leave policy?',
      expectedAnswer: '24 days of paid annual leave',
      relevantDocs: ['test_docs.txt'],
    },
    {
      question: 'How many days of leave do full-time employees get?',
      expectedAnswer: '24 days',
      relevantDocs: ['test_docs.txt'],
    },
    {
      question: 'What is the refund policy timeframe?',
      expectedAnswer: '30 days',
      relevantDocs: ['test_docs.txt'],
    },
    {
      question: 'What is the API rate limit for standard tier?',
      expectedAnswer: '100 requests per minute',
      relevantDocs: ['test_docs.txt'],
    },
    {
      question: 'Can unused leave be carried over?',
      expectedAnswer: 'Yes, up to 5 days',
      relevantDocs: ['test_docs.txt'],
    },
  ];

  // Run evaluation
  const askFn = async (question) => {
    const result = await engine.ask({
      question,
      searchMode: 'hybrid',
      rerank: true,
      sessionId: 'eval',
      conversationStore,
    });
    return result;
  };

  console.log(`Running ${testCases.length} test cases...\n`);
  const { aggregate, results } = await evaluator.runEvaluation(testCases, askFn);

  // Print results
  for (const r of results) {
    if (r.error) {
      console.log(`❌ ${r.testCase.question.substring(0, 60)}... → ERROR: ${r.error}`);
    } else {
      const score = r.evaluation.overallScore;
      const icon = score >= 70 ? '✅' : score >= 50 ? '⚠️' : '❌';
      console.log(`${icon} [${score}%] ${r.testCase.question.substring(0, 60)}...`);
      console.log(`   Expected: ${r.testCase.expectedAnswer}`);
      console.log(`   Faithfulness: ${r.evaluation.faithfulness} | Relevance: ${r.evaluation.answerRelevance} | Confidence: ${r.confidence}`);
      console.log(`   Answer: ${r.actualAnswer.substring(0, 100)}...`);
      console.log('');
    }
  }

  console.log('═══ Aggregate Results ═══');
  console.log(`Tests: ${aggregate.totalTests} total, ${aggregate.passed} passed, ${aggregate.failed} failed`);
  console.log(`Avg Overall Score: ${aggregate.avgOverallScore}`);
  console.log(`Avg Faithfulness: ${aggregate.avgFaithfulness}`);
  console.log(`Avg Relevance: ${aggregate.avgRelevance}`);
  console.log(`Avg Confidence: ${aggregate.avgConfidence}`);
  console.log(`Total Latency: ${aggregate.totalLatencyMs}ms\n`);

  // Save results
  const resultsDir = path.join(__dirname, '..', 'evaluation', 'results');
  if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });

  const report = {
    timestamp: new Date().toISOString(),
    model: config.CHAT_MODEL,
    embeddingModel: config.EMBEDDING_MODEL,
    aggregate,
    results: results.map(r => ({
      question: r.testCase?.question,
      expected: r.testCase?.expectedAnswer,
      actual: r.actualAnswer?.substring(0, 300),
      confidence: r.confidence,
      error: r.error,
      evaluation: r.evaluation,
    })),
  };

  const reportPath = path.join(resultsDir, `eval_${Date.now()}.json`);
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`Results saved to: ${reportPath}`);
}

main().catch(e => { console.error('Evaluation failed:', e); process.exit(1); });
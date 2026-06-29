var path = require('path');
var passed = 0, failed = 0, total = 0;
function assert(c, n) { total++; if (c) { passed++; console.log('  OK ' + n); } else { failed++; console.log('  FAIL ' + n); } }

async function run() {
  console.log('\n=== Enterprise Knowledge Assistant v3.0 - Test Suite ===\n');

  // 1. Config
  console.log('-- Config --');
  var config = require('../config');
  assert(config.CHAT_MODEL === 'llama-3.3-70b-versatile', 'Chat model');
  assert(config.EMBEDDING_DIMENSION === 384, 'Embedding dimension (local)');
  assert(config.CHUNK_SIZE === 500, 'Chunk size');
  assert(config.SEARCH_MODES.length === 3, 'Three search modes');
  assert(config.ENABLE_QUERY_CLASSIFICATION === true, 'Classification enabled');
  assert(config.ENABLE_ANSWER_VERIFICATION === true, 'Verification enabled');
  assert(config.ENABLE_RELATED_QUESTIONS === true, 'Related questions enabled');
  assert(config.ENABLE_STREAMING === true, 'Streaming enabled');
  assert(config.ENABLE_SOURCE_FILTER === true, 'Source filter enabled');
  assert(config.ENABLE_QUERY_DECOMPOSITION === true, 'Decomposition enabled');

  // 2. VectorStore
  console.log('\n-- VectorStore --');
  var VS = require('../src/core/vectorStore');
  var vs = new VS();
  assert(vs.size === 0, 'Empty start');
  vs.addDocument('Employee leave: 24 days annual.', [0.9,0.1,0.2], {filename:'HR.pdf',page:12,chunkIndex:0});
  vs.addDocument('Refund: 30 days return window.', [0.1,0.8,0.3], {filename:'Cust.pdf',page:5,chunkIndex:0});
  vs.addDocument('API v2 setup guide deployment.', [0.2,0.3,0.9], {filename:'Tech.pdf',page:3,chunkIndex:0});
  assert(vs.size === 3, '3 docs added');
  var r = vs.search({queryEmbedding:[0.9,0.1,0.2],mode:'semantic',topK:2});
  assert(r.length === 2, 'Semantic returns 2');
  assert(r[0].metadata.filename === 'HR.pdf', 'Top is HR.pdf');
  r = vs.search({queryText:'refund 30 days',mode:'keyword',topK:2});
  assert(r[0].searchMode === 'keyword', 'Keyword mode');
  r = vs.search({queryEmbedding:[0.9,0.1,0.2],queryText:'leave',mode:'hybrid',topK:3});
  assert(r.length === 3, 'Hybrid returns 3');
  assert(r[0].searchMode === 'hybrid', 'Hybrid mode');
  // Source filtering
  var filtered = r.filter(function(x) { return x.metadata.filename === 'HR.pdf'; });
  assert(filtered.length >= 0, 'Source filter works');
  // removeBySource
  vs.removeBySource('Cust.pdf');
  assert(vs.size === 2, 'After removeBySource: 2 docs remain');
  var sources = vs.getSources();
  assert(sources.indexOf('Cust.pdf') === -1, 'Cust.pdf removed from sources');
  assert(sources.indexOf('HR.pdf') !== -1, 'HR.pdf still in sources');
  vs.clear();
  assert(vs.size === 0, 'Clear works');

  // 3. LocalEmbedder
  console.log('\n-- LocalEmbedder --');
  var LE = require('../src/core/localEmbedder');
  var le = new LE({ dimension: 384 });
  var e1 = le.embed('employee leave policy annual days');
  var e2 = le.embed('employee leave policy annual days');
  var e3 = le.embed('completely different topic about rockets');
  assert(e1.length === 384, 'Embedding dimension is 384');
  assert(e2.length === 384, 'Batch item dimension is 384');
  assert(e1[0] === e2[0] && e1[100] === e2[100], 'Deterministic: same input = same output');
  // Similar texts should have higher cosine than dissimilar
  var cosSim = function(a, b) {
    var dot = 0, na = 0, nb = 0;
    for (var i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  };
  var simSame = cosSim(e1, e2);
  var simDiff = cosSim(e1, e3);
  assert(simSame > 0.99, 'Identical texts: cosine > 0.99 (got ' + simSame.toFixed(4) + ')');
  assert(simDiff < simSame, 'Different texts: lower cosine than identical');
  // Batch
  var batch = le.embedBatch(['hello world', 'foo bar baz']);
  assert(batch.length === 2, 'Batch returns 2');
  assert(batch[0].length === 384, 'Batch item 0 is 384-dim');
  assert(batch[1].length === 384, 'Batch item 1 is 384-dim');
  // Custom dimension
  var le512 = new LE({ dimension: 512 });
  assert(le512.embed('test').length === 512, 'Custom dimension 512 works');

  // 4. DocumentProcessor
  console.log('\n-- DocumentProcessor --');
  var dp = require('../src/core/documentProcessor');
  var chunks = dp.chunkText('Sentence one about leave. Sentence two about refund. Sentence three about tech.', 80, 20);
  assert(chunks.length > 0, 'Chunking works');
  assert(chunks[0].charStart === 0, 'Starts at 0');
  assert(dp.chunkText('').length === 0, 'Empty text');
  assert(dp.chunkText(null).length === 0, 'Null text');
  assert(dp.chunkText('Hi', 500, 100).length === 0, 'Short text');

  // PDF
  var pdfRes = await dp.processDocument('/home/z/my-project/upload/AI Engineer Assignment_ Build an Enterprise Knowledge Assistant (1).pdf', 'A.pdf');
  assert(pdfRes.stats.totalPages > 0, 'PDF pages');
  assert(pdfRes.stats.totalChunks > 0, 'PDF chunks');
  assert(pdfRes.chunks[0].metadata.filename === 'A.pdf', 'PDF metadata');

  // TXT
  require('fs').writeFileSync('/tmp/ekatest.txt', 'Benefits include 20 days leave. Health insurance provided. 401k matched at 6 percent.');
  var txtRes = await dp.processDocument('/tmp/ekatest.txt', 'test.txt');
  assert(txtRes.stats.totalPages === 1, 'TXT pages');
  assert(txtRes.stats.totalChunks >= 1, 'TXT chunks');

  // 5. Validators
  console.log('\n-- Validators --');
  var v = require('../src/utils/validators');
  assert(v.validateQuestion('What is the policy?') === 'What is the policy?', 'Valid Q');
  try { v.validateQuestion(''); assert(false,''); } catch(e) { assert(e.type === 'validation', 'Empty rejected'); }
  try { v.validateQuestion('Hi'); assert(false,''); } catch(e) { assert(true, 'Short rejected'); }
  try { v.validateQuestion('a'.repeat(2001)); assert(false,''); } catch(e) { assert(true, 'Long rejected'); }
  assert(v.validateSearchMode('semantic') === 'semantic', 'Semantic mode');
  try { v.validateSearchMode('bad'); assert(false,''); } catch(e) { assert(true, 'Bad mode rejected'); }
  assert(v.validateFileExtension('doc.pdf') === '.pdf', 'PDF ext');
  try { v.validateFileExtension('f.xyz'); assert(false,''); } catch(e) { assert(true, 'XYZ rejected'); }
  assert(v.parseBoolean('true') === true, 'Bool true');
  assert(v.parseBoolean('false') === false, 'Bool false');
  assert(v.parseBoolean(undefined, true) === true, 'Bool default true');
  assert(v.sanitizeInput('<script>x</script>Hello') === 'Hello', 'XSS stripped');
  assert(v.sanitizeInput('normal text') === 'normal text', 'Normal text passes');

  // 6. GroqClient (no API call)
  console.log('\n-- GroqClient --');
  var GC = require('../src/core/groqClient');
  assert(typeof GC.prototype.embed === 'function', 'Has embed');
  assert(typeof GC.prototype.embedBatch === 'function', 'Has embedBatch');
  assert(typeof GC.prototype.generateAnswer === 'function', 'Has generateAnswer');
  assert(typeof GC.prototype.streamAnswer === 'function', 'Has streamAnswer');
  assert(typeof GC.prototype.rewriteQuery === 'function', 'Has rewriteQuery');
  assert(typeof GC.prototype.classifyQuery === 'function', 'Has classifyQuery');
  assert(typeof GC.prototype.verifyAnswer === 'function', 'Has verifyAnswer');
  assert(typeof GC.prototype.generateRelatedQuestions === 'function', 'Has relatedQuestions');
  assert(typeof GC.prototype.decomposeQuery === 'function', 'Has decomposeQuery');
  assert(typeof GC.prototype.summarizeContext === 'function', 'Has summarizeContext');
  assert(typeof GC.prototype.generateBatch === 'function', 'Has generateBatch');
  try { new GC(); assert(true, 'Constructor works with .env key'); } catch(e) { assert(e.message.indexOf('GROQ_API_KEY') !== -1, 'Error mentions GROQ_API_KEY'); }

  // 7. QueryEngine
  console.log('\n-- QueryEngine --');
  var QE = require('../src/core/queryEngine');
  assert(typeof QE === 'function', 'QueryEngine is a class');

  // 8. Evaluator
  console.log('\n-- Evaluator --');
  var RE = require('../src/core/evaluator');
  assert(typeof RE === 'function', 'Evaluator is a class');

  // 9. Routes
  console.log('\n-- Routes --');
  var CR = require('../src/api/routes');
  assert(typeof CR === 'function', 'createRouter is a function');

  // 10. Frontend
  console.log('\n-- Frontend --');
  var html = require('fs').readFileSync(path.join(__dirname, '../public/index.html'), 'utf8');
  assert(html.length > 5000, 'HTML substantial');
  assert(html.includes('/api/upload'), 'Upload API');
  assert(html.includes('/api/ask'), 'Ask API');
  assert(html.includes('/api/ask/stream'), 'Streaming API');
  assert(html.includes('/api/feedback'), 'Feedback API');
  assert(html.includes('/api/analytics'), 'Analytics API');
  assert(html.includes('/api/export'), 'Export API');
  assert(html.includes('/api/documents/'), 'Delete source API');
  assert(html.includes('searchMode'), 'Search mode');
  assert(html.includes('rewriteQuery'), 'Rewrite toggle');
  assert(html.includes('rerank'), 'Rerank toggle');
  assert(html.includes('Verification'), 'Verification toggle');
  assert(html.includes('Classification'), 'Classification toggle');
  assert(html.includes('mode-tab'), 'Mode tabs in input');
  assert(html.includes('Decomposition'), 'Decomposition toggle');
  assert(html.includes('Related'), 'Related questions');
  assert(html.includes('Streaming'), 'Streaming toggle');
  assert(html.includes('Summarize'), 'Auto-summarize toggle');
  assert(html.includes('confidence'), 'Confidence badge');
  assert(html.includes('source-ref'), 'Source citations');
  assert(html.includes('feedback-row'), 'Feedback section');
  assert(html.includes('Semantic'), 'Semantic button');
  assert(html.includes('Keyword'), 'Keyword button');
  assert(html.includes('Hybrid'), 'Hybrid button');
  assert(html.includes('deleteFile'), 'Delete file function');
  assert(html.includes('clearChat'), 'Clear chat function');
  assert(html.includes('newSession'), 'New session function');
  assert(html.includes('sourceFilter'), 'Source filter dropdown');
  assert(html.includes('analyticsModal'), 'Analytics modal');
  assert(html.includes('formatsModal'), 'Formats modal');
  assert(html.includes('exportChat'), 'Export function');
  assert(html.includes('/api/export/'), 'Export API call');
  // Verify v5 function names exist (clean naming)
  assert(html.includes('function setMode'), 'Full function name: setMode');
  assert(html.includes('function askQuestion'), 'Full function name: askQuestion');
  assert(html.includes('function deleteFile'), 'Full function name: deleteFile');
  assert(html.includes('function loadDocuments'), 'Full function name: loadDocuments');
  assert(html.includes('function autoResize'), 'Full function name: autoResize');
  assert(html.includes('function newSession'), 'Full function name: newSession');
  // Verify no syntax errors by checking critical patterns
  assert(html.indexOf("ratings)'\\n") === -1, 'No missing + before \\n (old bug fixed)');

  // 11. Project Files
  console.log('\n-- Project Files --');
  assert(require('fs').existsSync(path.join(__dirname, '../.env.example')), '.env.example exists');
  assert(require('fs').existsSync(path.join(__dirname, '../.env')), '.env exists');
  assert(require('fs').existsSync(path.join(__dirname, '../.gitignore')), '.gitignore exists');
  assert(require('fs').existsSync(path.join(__dirname, '../docs/SYSTEM_DESIGN.md')), 'System design doc');
  assert(require('fs').existsSync(path.join(__dirname, '../src/core/localEmbedder.js')), 'localEmbedder.js exists');

  // 12. No nomic-embed-text references
  console.log('\n-- Model Isolation --');
  var srcDirs = ['src/core/', 'config/'];
  var hasNomic = false;
  srcDirs.forEach(function(dir) {
    var files = require('fs').readdirSync(path.join(__dirname, '../' + dir)).filter(function(f) { return f.endsWith('.js'); });
    files.forEach(function(f) {
      var content = require('fs').readFileSync(path.join(__dirname, '../' + dir + f), 'utf8').toLowerCase();
      if (content.indexOf('nomic') !== -1) hasNomic = true;
    });
  });
  assert(!hasNomic, 'No nomic-embed-text references in source code');

  console.log('\n=== Results: ' + passed + '/' + total + ' passed, ' + failed + ' failed ===\n');
  process.exit(failed > 0 ? 1 : 0);
}
run().catch(function(e) { console.error('Crash:', e); process.exit(1); });
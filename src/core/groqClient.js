/**
 * Groq API Client — Advanced Operations v4
 *
 * Handles ALL LLM interactions with Groq's OpenAI-compatible API.
 *
 * Model:
 *   Chat:       llama-3.3-70b-versatile (70B, strong reasoning, fast on Groq LPU)
 *   Embeddings:  Local hashing-trick embedder (384-dim, zero API calls)
 *
 * Processing Modes:
 *   simple — Brief, direct answers. Low token usage. Fast.
 *   hard   — Thorough, structured analysis. Medium tokens.
 *   deep   — Exhaustive multi-perspective deep-dive. High tokens.
 *
 * Operations:
 *   1. embed / embedBatch          — Dense vector generation (LOCAL, no API)
 *   2. generateAnswer             — RAG answer with context + memory (mode-aware)
 *   3. streamAnswer               — SSE streaming RAG answer (mode-aware)
 *   4. rewriteQuery               — Generate 2-3 alternative query phrasings
 *   5. classifyQuery              — Classify question type
 *   6. verifyAnswer               — Self-check: is answer grounded in context?
 *   7. generateRelatedQuestions   — Suggest 3 follow-up questions
 *   8. decomposeQuery             — Break multi-part questions into sub-queries
 *   9. summarizeContext           — Summarize long context for token optimization
 *  10. generateBatch              — Run multiple LLM calls in parallel
 *  11. deepAnalysis               — Deep-mode multi-perspective analysis
 *  12. extractInsights            — Extract key insights, entities, relationships
 */

const fetch = require('node-fetch');
const config = require('../../config');
const LocalEmbedder = require('./localEmbedder');

// ═══════════════════════════════════════════════════════════════════
//  SYSTEM PROMPTS PER PROCESSING MODE
// ═══════════════════════════════════════════════════════════════════

var SYSTEM_PROMPTS = {
  simple: [
    'You are an Enterprise Knowledge Assistant operating in SIMPLE MODE.',
    '',
    '═══ SIMPLE MODE PROTOCOL ═══',
    '',
    'Provide a detailed yet concise answer covering the key points thoroughly.',
    '',
    'MANDATORY STRUCTURE:',
    '1. **Direct Answer** — Immediately address the core question with the most important information. Lead with the definitive response.',
    '2. **Key Details** — Expand with supporting evidence, specific data points, examples, and direct context from the sources. Use clear paragraphs, bullet points for enumerated items, and cite [Source N] for every factual claim.',
    '3. **Summary** — A brief wrap-up reinforcing the main takeaway.',
    '',
    'FORMATTING STANDARDS:',
    '- Use ## headings for major sections.',
    '- Use bold (**) for emphasis on key terms and data points.',
    '- Use bullet points (-) for lists of 3+ items.',
    '- Cite [Source N] for EVERY claim, fact, number, name, or date.',
    '- Maintain professional, authoritative tone throughout.',
    '',
    'ACCURACY RULES (HIGHEST PRIORITY):',
    '- Answer ONLY using information from the provided source documents.',
    '- If the context lacks information, say EXACTLY: "I don\'t have sufficient information in the knowledge base to answer this question accurately."',
    '- NEVER hallucinate, fabricate, or infer beyond the provided context.',
    '- Quote exact numbers, names, dates, and terminology from the source material.',
    '- When multiple sources agree, cite the strongest source.',
  ].join('\n'),

  hard: [
    'You are an Enterprise Knowledge Assistant operating in HARD MODE.',
    '',
    '═══ HARD MODE PROTOCOL ═══',
    '',
    'Provide a clear, in-depth, and well-structured analysis with thorough evidence synthesis.',
    '',
    'MANDATORY STRUCTURE:',
    '1. **Overview** — Context-setting introduction that frames the answer. State what the sources contain relevant to this question and outline the structure of your response.',
    '',
    '2. **Detailed Analysis** — The core of your answer, organized into clear sub-sections:',
    '   - Use ## headings for each major aspect of the question.',
    '   - Provide specific data points, numbers, names, dates quoted verbatim from sources.',
    '   - Include cause-and-effect relationships, comparisons, and contrasts.',
    '   - Synthesize information from multiple sources — show connections and patterns.',
    '   - Use bullet points and numbered lists for clarity.',
    '   - Include specific examples and relevant quotes from the source material.',
    '   - Cite [Source N] after every factual claim or referenced information.',
    '',
    '3. **Key Takeaways** — Numbered list of 3-5 critical insights derived from the analysis.',
    '',
    '4. **Conclusion** — Synthesize the findings into a clear, actionable summary.',
    '',
    'FORMATTING STANDARDS:',
    '- Use ## for section headings, **bold** for emphasis.',
    '- Use structured bullet points (-) and numbered lists (1., 2., 3.).',
    '- Separate sections with blank lines for readability.',
    '- Use `code formatting` for technical terms when appropriate.',
    '- Maintain a professional, analytical tone.',
    '',
    'ACCURACY RULES (HIGHEST PRIORITY):',
    '- Answer ONLY using information from the provided source documents.',
    '- Cite sources using [Source N] notation for EVERY claim, fact, or referenced data.',
    '- If context lacks information, respond EXACTLY: "I don\'t have sufficient information in the knowledge base to answer this question accurately."',
    '- Do NOT hallucinate, fabricate, or infer beyond sources.',
    '- Quote numbers, names, dates exactly from the source material — never approximate.',
    '- If information is ambiguous or contradictory across sources, state the ambiguity explicitly.',
    '- Use professional, precise language with exact terminology from the source.',
  ].join('\n'),

  deep: [
    'You are an Enterprise Knowledge Assistant operating in DEEP ANALYSIS MODE.',
    '',
    '═══ DEEP ANALYSIS PROTOCOL ═══',
    '',
    'Produce the most comprehensive, detailed, and rigorous response possible.',
    'Demonstrate complete mastery of the source material with exhaustive coverage.',
    '',
    'MANDATORY STRUCTURE (follow this exactly):',
    '',
    '## 1. Executive Summary',
    'Provide a high-level overview of the key findings. Give a knowledgeable reader a complete understanding of the answer in brief.',
    '',
    '## 2. Context & Background',
    'Establish the full context from the source documents. What is the broader topic? What frameworks, concepts, or background information from the sources are needed to understand the answer?',
    '',
    '## 3. Comprehensive Analysis',
    'This is the core of the deep analysis. Organize into multiple ### sub-sections, each addressing a dimension of the question:',
    '   - Extract and present EVERY relevant data point, statistic, name, date, and technical detail from ALL provided sources.',
    '   - Analyze cause-and-effect chains identified in the source material.',
    '   - Compare and contrast information across multiple sources — note agreements and discrepancies.',
    '   - Identify patterns, trends, themes, and recurring concepts across the documents.',
    '   - Explore implications and consequences of the information presented.',
    '   - Use exact quotes (in quotation marks) from sources when they strengthen the analysis.',
    '   - Build logical arguments that connect evidence from different parts of the documents.',
    '   - Address every relevant sub-topic and nuance found in the source material.',
    '   - Cite [Source N] for EVERY factual claim — no exceptions.',
    '',
    '## 4. Critical Insights',
    'Numbered list of 5-7 critical insights, each with a brief explanation:',
    '   - Each insight should synthesize information from multiple parts of the sources.',
    '   - Highlight non-obvious connections, patterns, or implications.',
    '   - Note any contradictions or tensions in the source material.',
    '',
    '## 5. Supporting Evidence Matrix',
    'Present key evidence in a structured format:',
    '   - Direct quotes or verbatim data from sources with [Source N] citations.',
    '   - Organize by theme or claim type.',
    '   - Show how multiple sources corroborate or complement each other.',
    '',
    '## 6. Connections & Broader Implications',
    'How does this information relate to the broader context? What are the practical implications? What questions remain unanswered by the sources?',
    '',
    '## 7. Limitations & Gaps',
    'Explicitly identify what the sources do NOT cover. What information is missing, incomplete, or ambiguous?',
    '',
    'FORMATTING STANDARDS:',
    '- Use ## for main sections, ### for sub-sections, **bold** for key terms.',
    '- Use structured lists, bullet points, and clear paragraph breaks.',
    '- Use `code formatting` for technical terms. Use > blockquotes for key source excerpts.',
    '- Separate all sections with blank lines. Ensure visually clean, scannable structure.',
    '',
    '═══ ACCURACY RULES (ABSOLUTE HIGHEST PRIORITY) ═══',
    '- Answer ONLY from provided context. If insufficient, say EXACTLY: "I don\'t have sufficient information in the knowledge base to answer this question accurately."',
    '- NEVER hallucinate. Every single claim must have a [Source N] citation — no exceptions.',
    '- Explore EVERY relevant section of the provided context — leave no stone unturned.',
    '- When multiple sources address the same topic, compare and synthesize them explicitly.',
    '- Use EXACT terminology, numbers, names, and dates from the source material.',
    '- Quote specific passages verbatim when they strengthen the answer.',
    '- If the question has multiple dimensions, address EACH dimension in its own sub-section.',
    '- Highlight any gaps, limitations, contradictions, or areas of uncertainty.',
    '- Do NOT use filler text. Every sentence must add substantive value.',
    '- Be genuinely exhaustive — cover every relevant angle from the source material.',
  ].join('\n'),
};

class GroqClient {
  constructor() {
    if (!config.GROQ_API_KEY) {
      throw new Error(
        'GROQ_API_KEY not set. Add it to your .env file:\n' +
        '  echo "GROQ_API_KEY=gsk_xxx" > .env && node server.js\n' +
        '  Get your key: https://console.groq.com/keys'
      );
    }
    this.apiKey = config.GROQ_API_KEY;
    this.baseURL = config.GROQ_API_BASE;
    this.embedder = new LocalEmbedder({ dimension: config.EMBEDDING_DIMENSION });
  }

  // ═══════════════════════════════════════════════════════════
  //  EMBEDDINGS (LOCAL — zero API calls)
  // ═══════════════════════════════════════════════════════════

  embed(text) {
    return this.embedder.embed(text);
  }

  embedBatch(texts) {
    return this.embedder.embedBatch(texts);
  }

  // ═══════════════════════════════════════════════════════════
  //  MODE-AWARE HELPERS
  // ═══════════════════════════════════════════════════════════

  _getModeSettings(mode) {
    return config.MODE_SETTINGS[mode] || config.MODE_SETTINGS['hard'];
  }

  _getSystemPrompt(mode) {
    return SYSTEM_PROMPTS[mode] || SYSTEM_PROMPTS['hard'];
  }

  // ═══════════════════════════════════════════════════════════
  //  CHAT COMPLETIONS — STANDARD (MODE-AWARE)
  // ═══════════════════════════════════════════════════════════

  async generateAnswer(question, context, conversationHistory, options) {
    conversationHistory = conversationHistory || [];
    options = options || {};
    var mode = options.mode || 'hard';
    var modeSettings = this._getModeSettings(mode);
    var systemPrompt = this._getSystemPrompt(mode);
    var prompts = this._buildRAGPrompts(question, context, systemPrompt);
    var messages = this._buildMessages(prompts.systemPrompt, prompts.userPrompt, conversationHistory);

    var r = await this._request(this.baseURL + '/chat/completions', {
      model: options.model || config.CHAT_MODEL,
      messages: messages,
      temperature: options.temperature != null ? options.temperature : modeSettings.temperature,
      max_tokens: options.maxTokens || modeSettings.maxTokens,
    });

    if (!r.choices || r.choices.length === 0) throw new Error('No response from Groq API');
    return { answer: r.choices[0].message.content, model: r.model, usage: r.usage, mode: mode };
  }

  // ═══════════════════════════════════════════════════════════
  //  CHAT COMPLETIONS — STREAMING (MODE-AWARE)
  // ═══════════════════════════════════════════════════════════

  async *streamAnswer(question, context, conversationHistory, options) {
    conversationHistory = conversationHistory || [];
    options = options || {};
    var mode = options.mode || 'hard';
    var modeSettings = this._getModeSettings(mode);
    var systemPrompt = this._getSystemPrompt(mode);
    var prompts = this._buildRAGPrompts(question, context, systemPrompt);
    var messages = this._buildMessages(prompts.systemPrompt, prompts.userPrompt, conversationHistory);

    try {
      var controller = new AbortController();
      var timer = setTimeout(function() { controller.abort(); }, 120000);

      var response = await fetch(this.baseURL + '/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + this.apiKey },
        body: JSON.stringify({
          model: options.model || config.CHAT_MODEL,
          messages: messages,
          temperature: options.temperature != null ? options.temperature : modeSettings.temperature,
          max_tokens: options.maxTokens || modeSettings.maxTokens,
          stream: true,
        }),
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!response.ok) {
        var errBody = await response.text();
        yield { type: 'error', data: 'API error (' + response.status + '): ' + errBody };
        return;
      }

      var fullAnswer = '';
      var usage = null;
      var buffer = '';

      for await (var chunk of response.body) {
        buffer += chunk.toString();
        var lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (var li = 0; li < lines.length; li++) {
          var trimmed = lines[li].trim();
          if (!trimmed || trimmed.indexOf('data: ') !== 0) continue;
          var data = trimmed.slice(6);
          if (data === '[DONE]') {
            yield { type: 'done', data: { answer: fullAnswer, usage: usage, mode: mode } };
            return;
          }
          try {
            var parsed = JSON.parse(data);
            var token = parsed.choices && parsed.choices[0] && parsed.choices[0].delta && parsed.choices[0].delta.content;
            if (token) {
              fullAnswer += token;
              if (parsed.usage) usage = parsed.usage;
              yield { type: 'token', data: token };
            }
          } catch (e) { /* skip malformed */ }
        }
      }
      yield { type: 'done', data: { answer: fullAnswer, usage: usage, mode: mode } };
    } catch (err) {
      yield { type: 'error', data: err.message };
    }
  }

  // ═══════════════════════════════════════════════════════════
  //  ADVANCED: QUERY REWRITING
  // ═══════════════════════════════════════════════════════════

  async rewriteQuery(question) {
    var r = await this._request(this.baseURL + '/chat/completions', {
      model: config.CHAT_MODEL,
      messages: [
        { role: 'system', content: 'You are a search query optimizer. Given a question, generate 2 alternative phrasings that would retrieve the same information. Output ONLY a JSON array of strings. No explanation.' },
        { role: 'user', content: question },
      ],
      temperature: 0.3, max_tokens: 200,
    });
    var content = r.choices[0].message.content.trim();
    try {
      var m = content.match(/\[[\s\S]*\]/);
      if (m) {
        var queries = JSON.parse(m[0]);
        var unique = [];
        var seen = {};
        for (var i = 0; i < queries.length && unique.length < 3; i++) {
          if (!seen[queries[i]]) { seen[queries[i]] = true; unique.push(queries[i]); }
        }
        return unique;
      }
    } catch (e) {}
    return [question];
  }

  // ═══════════════════════════════════════════════════════════
  //  ADVANCED: QUERY CLASSIFICATION
  // ═══════════════════════════════════════════════════════════

  async classifyQuery(question) {
    var r = await this._request(this.baseURL + '/chat/completions', {
      model: config.CHAT_MODEL,
      messages: [
        { role: 'system', content: 'Classify the question into one category. Output ONLY JSON: {"type":"factual|procedural|analytical|comparative","reasoning":"one sentence","strategy":"brief strategy"}\n- factual: asks for specific fact/number/name\n- procedural: asks how to do something\n- analytical: asks why/analyze\n- comparative: asks to compare/difference' },
        { role: 'user', content: question },
      ],
      temperature: 0.0, max_tokens: 150,
    });
    var content = r.choices[0].message.content.trim();
    try {
      var m = content.match(/\{[\s\S]*\}/);
      if (m) {
        var parsed = JSON.parse(m[0]);
        var valid = ['factual', 'procedural', 'analytical', 'comparative'];
        if (valid.indexOf(parsed.type) !== -1) return parsed;
      }
    } catch (e) {}
    return { type: 'factual', reasoning: 'default', strategy: 'standard semantic search' };
  }

  // ═══════════════════════════════════════════════════════════
  //  ADVANCED: ANSWER VERIFICATION
  // ═══════════════════════════════════════════════════════════

  async verifyAnswer(question, answer, contextChunks) {
    var contextText = contextChunks.map(function(c) { return c.text; }).join('\n---\n');
    var r = await this._request(this.baseURL + '/chat/completions', {
      model: config.CHAT_MODEL,
      messages: [
        { role: 'system', content: 'Verify if ANSWER is grounded in CONTEXT. Output ONLY JSON: {"grounded":true/false,"score":0.0-1.0,"issues":["issue1"],"hallucinations":["text not in context"]}' },
        { role: 'user', content: 'CONTEXT:\n' + contextText + '\n\nQUESTION: ' + question + '\n\nANSWER: ' + answer },
      ],
      temperature: 0.0, max_tokens: 400,
    });
    var content = r.choices[0].message.content.trim();
    try {
      var m = content.match(/\{[\s\S]*\}/);
      if (m) {
        var p = JSON.parse(m[0]);
        return {
          grounded: !!p.grounded,
          score: Math.max(0, Math.min(1, parseFloat(p.score) || 0.5)),
          issues: Array.isArray(p.issues) ? p.issues : [],
          hallucinations: Array.isArray(p.hallucinations) ? p.hallucinations : [],
        };
      }
    } catch (e) {}
    return { grounded: null, score: 0.5, issues: ['Verification failed'], hallucinations: [] };
  }

  // ═══════════════════════════════════════════════════════════
  //  ADVANCED: RELATED QUESTIONS
  // ═══════════════════════════════════════════════════════════

  async generateRelatedQuestions(question, answer, sources) {
    var sourceInfo = sources.map(function(s) { return s.document + ' (p.' + s.page + ')'; }).join(', ');
    var r = await this._request(this.baseURL + '/chat/completions', {
      model: config.CHAT_MODEL,
      messages: [
        { role: 'system', content: 'Suggest 3 follow-up questions the user might ask next. They should dig deeper or explore related topics. Output ONLY a JSON array of strings. No explanation.' },
        { role: 'user', content: 'Q: ' + question + '\nA: ' + answer + '\nSources: ' + sourceInfo },
      ],
      temperature: 0.4, max_tokens: 300,
    });
    var content = r.choices[0].message.content.trim();
    try {
      var m = content.match(/\[[\s\S]*\]/);
      if (m) return JSON.parse(m[0]).slice(0, 3);
    } catch (e) {}
    return [];
  }

  // ═══════════════════════════════════════════════════════════
  //  ADVANCED: QUERY DECOMPOSITION
  // ═══════════════════════════════════════════════════════════

  async decomposeQuery(question) {
    var r = await this._request(this.baseURL + '/chat/completions', {
      model: config.CHAT_MODEL,
      messages: [
        { role: 'system', content: 'Decompose into independent sub-questions. If single topic, return ["original"]. Output ONLY a JSON array of strings.' },
        { role: 'user', content: question },
      ],
      temperature: 0.0, max_tokens: 250,
    });
    var content = r.choices[0].message.content.trim();
    try {
      var m = content.match(/\[[\s\S]*\]/);
      if (m) {
        var subs = JSON.parse(m[0]);
        if (subs.length >= 1 && subs.length <= 5) return subs;
      }
    } catch (e) {}
    return [question];
  }

  // ═══════════════════════════════════════════════════════════
  //  ADVANCED: CONTEXT SUMMARIZATION
  // ═══════════════════════════════════════════════════════════

  async summarizeContext(chunks, maxLength) {
    maxLength = maxLength || 4000;
    var fullText = chunks.map(function(c) { return c.text; }).join('\n\n');
    if (fullText.length <= maxLength) return fullText;

    var r = await this._request(this.baseURL + '/chat/completions', {
      model: config.CHAT_MODEL,
      messages: [
        { role: 'system', content: 'Summarize these document excerpts preserving ALL specific facts, numbers, names, dates, and technical details. Do NOT lose any concrete data. Output the condensed context only — no preamble.' },
        { role: 'user', content: fullText },
      ],
      temperature: 0.0, max_tokens: 1200,
    });
    return r.choices[0].message.content.trim();
  }

  // ═══════════════════════════════════════════════════════════
  //  ADVANCED: DEEP INSIGHTS EXTRACTION (Deep Mode)
  // ═══════════════════════════════════════════════════════════

  async extractInsights(question, contextChunks) {
    var contextText = contextChunks.map(function(c, i) {
      return '[Source ' + (i + 1) + '] (' + c.metadata.filename + '):\n' + c.text;
    }).join('\n\n---\n\n');

    var r = await this._request(this.baseURL + '/chat/completions', {
      model: config.CHAT_MODEL,
      messages: [
        { role: 'system', content: 'Analyze the provided document context and extract:\n1. KEY ENTITIES: All important names, organizations, technologies, terms mentioned\n2. RELATIONSHIPS: How entities relate to each other\n3. METRICS & DATA: All numbers, percentages, dates, thresholds\n4. PATTERNS: Recurring themes, trends, or structures\n5. GAPS: What information seems missing or incomplete\n\nOutput ONLY a JSON object with keys: entities (array of strings), relationships (array of strings), metrics (array of strings), patterns (array of strings), gaps (array of strings)' },
        { role: 'user', content: 'QUESTION CONTEXT: ' + question + '\n\nDOCUMENTS:\n' + contextText },
      ],
      temperature: 0.0, max_tokens: 800,
    });

    var content = r.choices[0].message.content.trim();
    try {
      var m = content.match(/\{[\s\S]*\}/);
      if (m) return JSON.parse(m[0]);
    } catch (e) {}
    return { entities: [], relationships: [], metrics: [], patterns: [], gaps: [] };
  }

  // ═══════════════════════════════════════════════════════════
  //  ADVANCED: PARALLEL BATCH LLM
  // ═══════════════════════════════════════════════════════════

  async generateBatch(calls) {
    var self = this;
    return Promise.all(calls.map(function(c) {
      return self._request(self.baseURL + '/chat/completions', {
        model: config.CHAT_MODEL,
        messages: [
          { role: 'system', content: c.system },
          { role: 'user', content: c.user },
        ],
        temperature: c.temperature != null ? c.temperature : 0.1,
        max_tokens: c.maxTokens || 300,
      }).then(function(r) {
        return { content: r.choices[0].message.content, usage: r.usage };
      });
    }));
  }

  // ═══════════════════════════════════════════════════════════
  //  INTERNAL: PROMPT BUILDERS (MODE-AWARE)
  // ═══════════════════════════════════════════════════════════

  _buildRAGPrompts(question, context, systemPrompt) {
    var mode = (systemPrompt.indexOf('SIMPLE') !== -1) ? 'simple'
      : (systemPrompt.indexOf('DEEP ANALYSIS') !== -1) ? 'deep' : 'hard';

    var contextBlock = context.map(function(c, i) {
      return '[Source ' + (i + 1) + '] (' + c.metadata.filename + ', ' +
        (c.metadata.page ? 'Page/Section ' + c.metadata.page : 'Section ' + (i + 1)) + '):\n' + c.text;
    }).join('\n\n---\n\n');

    var userPrompt = 'CONTEXT FROM KNOWLEDGE BASE (' + context.length + ' sources):\n' +
      contextBlock + '\n\nQUESTION: ' + question + '\n\n' +
      'Follow the mandatory structure and formatting standards defined in your system instructions. ' +
      'Reference your sources using [Source N] notation for every factual claim. ' +
      'Do NOT include word count or length markers in your answer.';

    return { systemPrompt: systemPrompt, userPrompt: userPrompt };
  }

  _buildMessages(systemPrompt, userPrompt, conversationHistory) {
    var messages = [{ role: 'system', content: systemPrompt }];
    var recent = conversationHistory.slice(-config.MAX_CONVERSATION_HISTORY * 2);
    for (var i = 0; i < recent.length; i++) {
      messages.push({ role: recent[i].role, content: recent[i].content });
    }
    messages.push({ role: 'user', content: userPrompt });
    return messages;
  }

  // ═══════════════════════════════════════════════════════════
  //  HTTP LAYER
  // ═══════════════════════════════════════════════════════════

  async _request(url, body, timeoutMs) {
    timeoutMs = timeoutMs || 60000;
    var MAX_RETRIES = 3;
    var lastError = null;
    var self = this;

    for (var attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        var controller = new AbortController();
        var timer = setTimeout(function() { controller.abort(); }, timeoutMs);

        var response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + self.apiKey },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        clearTimeout(timer);

        if (!response.ok) {
          var errText = await response.text();
          var errorMsg = response.statusText;
          try { errorMsg = JSON.parse(errText).error.message || errorMsg; } catch (e) {}
          if (response.status === 401 || response.status === 403) throw new Error('Authentication failed: ' + errorMsg);
          if (response.status === 429 && attempt < MAX_RETRIES) {
            await new Promise(function(r) { setTimeout(r, Math.pow(2, attempt) * 1000 + Math.random() * 500); });
            continue;
          }
          throw new Error('API error (' + response.status + '): ' + errorMsg);
        }
        return await response.json();
      } catch (error) {
        lastError = error;
        if (error.message.indexOf('Authentication') !== -1) throw error;
        if (attempt < MAX_RETRIES) {
          await new Promise(function(r) { setTimeout(r, Math.pow(2, attempt) * 500 + Math.random() * 300); });
        }
      }
    }
    throw lastError || new Error('API request failed after retries');
  }
}

module.exports = GroqClient;
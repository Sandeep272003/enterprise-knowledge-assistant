/**
 * Universal Document Processing Pipeline
 *
 * Pipeline: Detect Format → Extract Text → Chunk → [Ready for Embed]
 *
 * Supported formats (20+):
 *   Documents:     PDF, DOCX, TXT, MD, RTF
 *   Spreadsheets:  XLSX, XLS, CSV, TSV
 *   Presentations: PPTX (text from slides)
 *   Data/Config:   JSON, XML, YAML, TOML, INI, CFG, CONF
 *   Web:           HTML, HTM
 *   Logs:          LOG
 *
 * Extraction strategy per format:
 *   - PDF:     pdf-parse → per-page splitting with form-feed
 *   - DOCX:    mammoth → clean HTML-to-text
 *   - XLSX:    xlsx → each row as "Col1: val1, Col2: val2" per sheet
 *   - CSV/TSV: line-by-line → header-prefixed rows
 *   - JSON:    recursive key-value flattening with paths
 *   - HTML:    html-to-text → structured text with headings
 *   - XML:     regex strip tags → extract text content
 *   - YAML/INI/TOML/CFG: parsed as key-value text blocks
 *   - PPTX:    unzip → parse slide XML → extract text runs
 *   - RTF:     strip control words → extract visible text
 *   - TXT/MD/LOG: direct read, markdown headings preserved
 *
 * Chunking Strategy:
 *   - Fixed-size with overlap (configurable: default 500 chars, 100 overlap)
 *   - Sentence-boundary aware: breaks at '. ', '! ', '? ', '\n'
 *   - Minimum chunk length filter: skips fragments shorter than 30 chars
 *   - Page/section/sheet metadata preserved
 */

const fs = require('fs');
const path = require('path');
const config = require('../../config');

// ═══════════════════════════════════════════════════════════════════
//  FORMAT-SPECIFIC EXTRACTORS
// ═══════════════════════════════════════════════════════════════════

/**
 * PDF: per-page extraction via pdf-parse
 */
async function extractPDF(filePath) {
  const pdfParse = require('pdf-parse');
  const buf = fs.readFileSync(filePath);
  const data = await pdfParse(buf);
  const pages = data.text.split('\f');
  const pageTexts = [];
  for (let i = 0; i < pages.length; i++) {
    const cleaned = pages[i].replace(/\r\n/g, '\n').trim();
    if (cleaned.length > 0) pageTexts.push({ page: i + 1, text: cleaned });
  }
  return { fullText: data.text, pages: pageTexts, pageCount: data.numpages };
}

/**
 * DOCX: Microsoft Word via mammoth
 */
async function extractDOCX(filePath) {
  const mammoth = require('mammoth');
  const result = await mammoth.extractRawText({ path: filePath });
  const text = result.value;
  // Split into sections by double newlines (paragraph blocks)
  const sections = text.split(/\n\s*\n/).filter(function(s) { return s.trim().length > 0; });
  const pages = sections.map(function(s, i) {
    return { page: i + 1, text: s.trim() };
  });
  if (pages.length === 0) pages.push({ page: 1, text: text.trim() });
  return { fullText: text, pages: pages, pageCount: pages.length };
}

/**
 * XLSX: Microsoft Excel / LibreOffice Calc via xlsx library
 * Each sheet becomes a "page", each row is a key-value line.
 */
function extractXLSX(filePath) {
  const XLSX = require('xlsx');
  const workbook = XLSX.readFile(filePath);
  const allPages = [];
  let fullText = '';

  workbook.SheetNames.forEach(function(sheetName, sheetIdx) {
    const sheet = workbook.Sheets[sheetName];
    const rows = XLSX.utils.sheet_to_json(sheet, { defval: '' });
    if (rows.length === 0) return;

    const headers = Object.keys(rows[0]);
    let sheetText = 'Sheet: ' + sheetName + '\n';
    sheetText += 'Columns: ' + headers.join(', ') + '\n';
    sheetText += '---\n';

    rows.forEach(function(row, rowIdx) {
      const parts = [];
      headers.forEach(function(h) {
        if (row[h] !== undefined && row[h] !== null && String(row[h]).trim() !== '') {
          parts.push(h + ': ' + row[h]);
        }
      });
      if (parts.length > 0) {
        sheetText += 'Row ' + (rowIdx + 1) + ': ' + parts.join(' | ') + '\n';
      }
    });

    allPages.push({ page: sheetIdx + 1, text: sheetText.trim() });
    fullText += sheetText + '\n';
  });

  if (allPages.length === 0) allPages.push({ page: 1, text: '' });
  return { fullText: fullText.trim(), pages: allPages, pageCount: allPages.length };
}

/**
 * PPTX: PowerPoint — unzip and parse slide XML for text runs
 */
function extractPPTX(filePath) {
  const AdmZip = require('adm-zip');
  let zip;
  try {
    zip = new AdmZip(filePath);
  } catch (e) {
    // Fallback: try JSZip-like approach or return empty
    return { fullText: '', pages: [{ page: 1, text: '' }], pageCount: 1 };
  }

  const slideEntries = zip.getEntries().filter(function(e) {
    return e.entryName.indexOf('ppt/slides/slide') !== -1 && e.entryName.endsWith('.xml');
  });

  // Sort slides numerically
  slideEntries.sort(function(a, b) {
    var mA = a.entryName.match(/slide(\d+)/);
    var mB = b.entryName.match(/slide(\d+)/);
    var numA = mA ? parseInt(mA[1]) : 0;
    var numB = mB ? parseInt(mB[1]) : 0;
    return numA - numB;
  });

  const pages = [];
  let fullText = '';

  slideEntries.forEach(function(entry, idx) {
    var xml = entry.getData().toString('utf8');
    // Extract text between <a:t> tags
    var texts = [];
    var regex = /<a:t[^>]*>([\s\S]*?)<\/a:t>/g;
    var match;
    while ((match = regex.exec(xml)) !== null) {
      var t = match[1].replace(/<[^>]+>/g, '').trim();
      if (t) texts.push(t);
    }
    var slideText = texts.join(' ');
    if (slideText.trim().length > 0) {
      pages.push({ page: idx + 1, text: 'Slide ' + (idx + 1) + ': ' + slideText.trim() });
      fullText += slideText + '\n';
    }
  });

  if (pages.length === 0) pages.push({ page: 1, text: '' });
  return { fullText: fullText.trim(), pages: pages, pageCount: pages.length };
}

/**
 * CSV / TSV: Parse with header detection, each line as a record
 */
function extractCSV(filePath, delimiter) {
  delimiter = delimiter || ',';
  var text = fs.readFileSync(filePath, 'utf-8');
  var lines = text.split(/\r?\n/).filter(function(l) { return l.trim().length > 0; });
  if (lines.length === 0) return { fullText: '', pages: [{ page: 1, text: '' }], pageCount: 1 };

  var headers = lines[0].split(delimiter).map(function(h) { return h.trim().replace(/^"|"$/g, ''); });
  var records = [];
  for (var i = 1; i < lines.length; i++) {
    var vals = lines[i].split(delimiter).map(function(v) { return v.trim().replace(/^"|"$/g, ''); });
    var parts = [];
    headers.forEach(function(h, idx) {
      if (vals[idx]) parts.push(h + ': ' + vals[idx]);
    });
    if (parts.length > 0) records.push('Row ' + i + ': ' + parts.join(' | '));
  }

  var fullText = 'Columns: ' + headers.join(', ') + '\n' + records.join('\n');
  return { fullText: fullText, pages: [{ page: 1, text: fullText }], pageCount: 1 };
}

/**
 * JSON: Recursive key-value flattening with dot-notation paths
 */
function extractJSON(filePath) {
  var text = fs.readFileSync(filePath, 'utf-8');
  var data = JSON.parse(text);
  var lines = [];
  flattenJSON(data, '', lines);
  var fullText = lines.join('\n');
  return { fullText: fullText, pages: [{ page: 1, text: fullText }], pageCount: 1 };
}

function flattenJSON(obj, prefix, lines) {
  if (Array.isArray(obj)) {
    obj.forEach(function(item, idx) {
      flattenJSON(item, prefix + '[' + idx + ']', lines);
    });
  } else if (obj && typeof obj === 'object') {
    Object.keys(obj).forEach(function(key) {
      var newPrefix = prefix ? prefix + '.' + key : key;
      var val = obj[key];
      if (typeof val === 'object' && val !== null) {
        flattenJSON(val, newPrefix, lines);
      } else {
        lines.push(newPrefix + ': ' + val);
      }
    });
  } else {
    if (prefix) lines.push(prefix + ': ' + obj);
  }
}

/**
 * HTML / HTM: Convert to structured text via html-to-text
 */
function extractHTML(filePath) {
  var htmlToText = require('html-to-text');
  var html = fs.readFileSync(filePath, 'utf-8');
  var text = htmlToText.convert(html, {
    wordwrap: false,
    selectors: [
      { selector: 'h1', options: { uppercase: false, prefix: '# ' } },
      { selector: 'h2', options: { uppercase: false, prefix: '## ' } },
      { selector: 'h3', options: { uppercase: false, prefix: '### ' } },
      { selector: 'table', options: { uppercase: false } },
    ],
  });
  return { fullText: text, pages: [{ page: 1, text: text.trim() }], pageCount: 1 };
}

/**
 * XML: Strip tags, preserve text content with element paths
 */
function extractXML(filePath) {
  var text = fs.readFileSync(filePath, 'utf-8');
  var lines = [];
  // Extract text content with tag context
  var regex = /<([^!\/?][^>]*[^\/])>([\s\S]*?)<\/\1>/g;
  var match;
  while ((match = regex.exec(text)) !== null) {
    var tag = match[1].split(' ')[0];
    var content = match[2].replace(/<[^>]+>/g, '').trim();
    if (content.length > 0) lines.push(tag + ': ' + content);
  }
  // Also get self-closing and attribute hints
  if (lines.length === 0) {
    var plain = text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
    lines.push(plain);
  }
  var fullText = lines.join('\n');
  return { fullText: fullText, pages: [{ page: 1, text: fullText }], pageCount: 1 };
}

/**
 * YAML / TOML / INI / CFG / CONF: Key-value config files
 */
function extractConfigFile(filePath) {
  var text = fs.readFileSync(filePath, 'utf-8');
  return { fullText: text, pages: [{ page: 1, text: text.trim() }], pageCount: 1 };
}

/**
 * RTF: Strip RTF control words, extract visible text
 */
function extractRTF(filePath) {
  var text = fs.readFileSync(filePath, 'utf-8');
  // Remove RTF control words and groups
  var cleaned = text
    .replace(/\\[a-z]+\d*\s?/gi, ' ')
    .replace(/\\'[0-9a-fA-F]{2}/g, '')
    .replace(/[{}]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  // Remove RTF header
  var start = cleaned.indexOf('\\rtf');
  if (start !== -1) {
    cleaned = cleaned.substring(0, start) + cleaned.substring(start + 4);
  }
  return { fullText: cleaned, pages: [{ page: 1, text: cleaned }], pageCount: 1 };
}

/**
 * TXT / MD / LOG: Direct read
 */
function extractPlainText(filePath) {
  var text = fs.readFileSync(filePath, 'utf-8');
  return { fullText: text, pages: [{ page: 1, text: text.trim() }], pageCount: 1 };
}

// ═══════════════════════════════════════════════════════════════════
//  FORMAT ROUTER
// ═══════════════════════════════════════════════════════════════════

/**
 * Route to the correct extractor based on file extension
 */
function getExtractor(ext) {
  var extractors = {
    '.pdf':  extractPDF,
    '.docx': extractDOCX,
    '.xlsx': extractXLSX,
    '.pptx': extractPPTX,
    '.csv':  function(p) { return extractCSV(p, ','); },
    '.tsv':  function(p) { return extractCSV(p, '\t'); },
    '.json': extractJSON,
    '.html': extractHTML,
    '.htm':  extractHTML,
    '.xml':  extractXML,
    '.yaml': extractConfigFile,
    '.yml':  extractConfigFile,
    '.toml': extractConfigFile,
    '.ini':  extractConfigFile,
    '.cfg':  extractConfigFile,
    '.conf': extractConfigFile,
    '.rtf':  extractRTF,
    '.txt':  extractPlainText,
    '.md':   extractPlainText,
    '.log':  extractPlainText,
  };
  return extractors[ext] || null;
}

/**
 * Get human-readable format description
 */
function getFormatInfo(ext) {
  var info = {
    '.pdf':  'PDF Document (Adobe)',
    '.docx': 'Word Document (Microsoft)',
    '.xlsx': 'Excel Spreadsheet (Microsoft)',
    '.pptx': 'PowerPoint Presentation (Microsoft)',
    '.csv':  'Comma-Separated Values',
    '.tsv':  'Tab-Separated Values',
    '.json': 'JSON Data',
    '.html': 'HTML Web Page',
    '.htm':  'HTML Web Page',
    '.xml':  'XML Data',
    '.yaml': 'YAML Configuration',
    '.yml':  'YAML Configuration',
    '.toml': 'TOML Configuration',
    '.ini':  'INI Configuration',
    '.cfg':  'Configuration File',
    '.conf': 'Configuration File',
    '.rtf':  'Rich Text Format',
    '.txt':  'Plain Text',
    '.md':   'Markdown Document',
    '.log':  'Log File',
  };
  return info[ext] || 'Unknown Format';
}

// ═══════════════════════════════════════════════════════════════════
//  CHUNKING ENGINE
// ═══════════════════════════════════════════════════════════════════

/**
 * Smart text chunking with overlap and sentence-boundary awareness
 */
function chunkText(text, chunkSize, overlap) {
  chunkSize = chunkSize || config.CHUNK_SIZE;
  overlap = overlap || config.CHUNK_OVERLAP;
  if (!text || text.trim().length === 0) return [];

  var chunks = [];
  var start = 0;
  var step = Math.max(1, chunkSize - overlap);

  while (start < text.length) {
    var end = Math.min(start + chunkSize, text.length);

    if (end < text.length) {
      var searchStart = Math.max(start, end - 100);
      var region = text.substring(searchStart, end);
      var lastBreak = Math.max(
        region.lastIndexOf('. '),
        region.lastIndexOf('! '),
        region.lastIndexOf('? '),
        region.lastIndexOf('\n')
      );
      if (lastBreak !== -1) {
        var proposed = searchStart + lastBreak + 1;
        if (proposed > start + 50) end = proposed;
      }
    }

    var chunk = text.substring(start, end).trim();
    if (chunk.length >= config.MIN_CHUNK_LENGTH) {
      chunks.push({ text: chunk, charStart: start, charEnd: end });
    }

    var nextStart = end - overlap;
    if (nextStart <= start) {
      if (end >= text.length) break;
      start = end;
    } else {
      start = nextStart;
    }
  }

  return chunks;
}

// ═══════════════════════════════════════════════════════════════════
//  MAIN ENTRY POINT
// ═══════════════════════════════════════════════════════════════════

/**
 * Process any uploaded file into embeddable chunks.
 * Automatically detects format and routes to the correct extractor.
 *
 * @param {string} filePath - Path on disk
 * @param {string} originalName - Original filename
 * @returns {Promise<{chunks: Array, stats: object}>}
 */
async function processDocument(filePath, originalName) {
  var ext = path.extname(originalName).toLowerCase();
  var extractor = getExtractor(ext);

  if (!extractor) {
    throw new Error(
      'Unsupported file type: ' + ext + '. Supported: ' + config.ALLOWED_EXTENSIONS.join(', ')
    );
  }

  var formatInfo = getFormatInfo(ext);
  console.log('[EXTRACT] ' + originalName + ' → ' + formatInfo);

  var extracted;
  try {
    extracted = await extractor(filePath);
  } catch (err) {
    throw new Error('Failed to extract text from ' + originalName + ' (' + formatInfo + '): ' + err.message);
  }

  if (!extracted.fullText || extracted.fullText.trim().length === 0) {
    throw new Error('No text content could be extracted from "' + originalName + '". The file may be empty, corrupt, or contain only images.');
  }

  // Chunk each page/section independently to preserve metadata
  var allChunks = [];
  for (var i = 0; i < extracted.pages.length; i++) {
    var pageText = extracted.pages[i];
    var pageChunks = chunkText(pageText.text);
    for (var j = 0; j < pageChunks.length; j++) {
      allChunks.push({
        text: pageChunks[j].text,
        metadata: {
          filename: originalName,
          format: ext,
          formatLabel: formatInfo,
          page: pageText.page,
          chunkIndex: j,
          charStart: pageChunks[j].charStart,
          charEnd: pageChunks[j].charEnd,
          totalChunksInPage: pageChunks.length,
        },
      });
    }
  }

  return {
    chunks: allChunks,
    stats: {
      filename: originalName,
      format: ext,
      formatLabel: formatInfo,
      totalPages: extracted.pageCount,
      totalChunks: allChunks.length,
      totalCharacters: extracted.fullText.length,
      avgChunkSize: allChunks.length > 0
        ? Math.round(allChunks.reduce(function(s, c) { return s + c.text.length; }, 0) / allChunks.length)
        : 0,
    },
  };
}

module.exports = { processDocument, chunkText, getFormatInfo, getExtractor };

/**
 * Input Validation Utilities
 */

const config = require('../../config');

class ValidationError extends Error {
  constructor(message) {
    super(message);
    this.type = 'validation';
  }
}

function validateQuestion(question) {
  if (question === undefined || question === null) {
    throw new ValidationError('Question is required');
  }
  var q = String(question).trim();
  if (q.length === 0) throw new ValidationError('Question cannot be empty');
  if (q.length < config.MIN_QUESTION_LENGTH) {
    throw new ValidationError('Question too short. Minimum ' + config.MIN_QUESTION_LENGTH + ' characters required.');
  }
  if (q.length > config.MAX_QUESTION_LENGTH) {
    throw new ValidationError('Question too long. Maximum ' + config.MAX_QUESTION_LENGTH + ' characters allowed.');
  }
  return q;
}

function validateSearchMode(mode) {
  if (!mode) return 'semantic';
  var normalized = mode.toLowerCase();
  if (config.SEARCH_MODES.indexOf(normalized) === -1) {
    throw new ValidationError('Invalid search mode: "' + mode + '". Supported: ' + config.SEARCH_MODES.join(', '));
  }
  return normalized;
}

function validateProcessingMode(mode) {
  if (!mode) return 'hard';
  var normalized = mode.toLowerCase();
  if (config.PROCESSING_MODES.indexOf(normalized) === -1) {
    throw new ValidationError('Invalid processing mode: "' + mode + '". Supported: ' + config.PROCESSING_MODES.join(', '));
  }
  return normalized;
}

function validateFileExtension(filename) {
  var ext = '.' + filename.split('.').pop().toLowerCase();
  if (config.ALLOWED_EXTENSIONS.indexOf(ext) === -1) {
    throw new ValidationError('Unsupported file type: ' + ext + '. Supported: ' + config.ALLOWED_EXTENSIONS.join(', '));
  }
  return ext;
}

function parseBoolean(value, defaultValue) {
  if (defaultValue === undefined) defaultValue = false;
  if (value === undefined || value === null) return defaultValue;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    return ['true', '1', 'yes'].indexOf(value.toLowerCase()) !== -1;
  }
  return defaultValue;
}

function sanitizeInput(text) {
  return String(text)
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .trim();
}

module.exports = {
  ValidationError,
  validateQuestion,
  validateSearchMode,
  validateProcessingMode,
  validateFileExtension,
  parseBoolean,
  sanitizeInput,
};
/**
 * Global Error Handler Middleware
 * Catches all unhandled errors and returns consistent JSON responses.
 * Logs errors with context for debugging.
 */

function errorHandler(err, req, res, _next) {
  // Multer file size error
  if (err.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({
      error: `File too large. Maximum size is ${Math.round(50 * 1024 * 1024 / 1024 / 1024 * 100) / 100}MB.`,
    });
  }

  // Multer unexpected field
  if (err.code === 'LIMIT_UNEXPECTED_FILE') {
    return res.status(400).json({ error: 'Unexpected field in upload. Use field name "file".' });
  }

  // Multer file filter error
  if (err.message && err.message.startsWith('Unsupported file type')) {
    return res.status(400).json({ error: err.message });
  }

  // Request validation errors
  if (err.type === 'validation') {
    return res.status(400).json({ error: err.message });
  }

  // API errors
  if (err.message && (err.message.includes('API error') || err.message.includes('Authentication'))) {
    return res.status(502).json({ error: err.message });
  }

  // Default: Internal server error
  console.error(`[ERROR] ${new Date().toISOString()} ${req.method} ${req.path}:`, err.message);
  console.error(err.stack);

  res.status(500).json({
    error: process.env.NODE_ENV === 'development'
      ? err.message
      : 'An internal error occurred. Please try again.',
  });
}

module.exports = errorHandler;
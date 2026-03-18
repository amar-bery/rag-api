/**
 * server.js — HTTP API wrapper around the RAG core
 *
 * Endpoints:
 *   GET  /health  → { status: "ok" }
 *   POST /ask     → { answer: string, sources: string[] }
 *
 * Config (environment variables):
 *   OPENAI_API_KEY   (required)  — OpenAI secret key
 *   FAQ_DIR          (optional)  — path to FAQ directory (default: ./faqs)
 *   PORT             (optional)  — server port (default: 3000)
 *   COMPLETION_MODEL (optional)  — OpenAI chat model (default: gpt-4o-mini)
 *
 * The server refuses to start if OPENAI_API_KEY is missing (fail-fast).
 */

import 'dotenv/config';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { initOpenAI, ingestDirectory, ask } from './rag.js';

// ─── Fail-fast config validation ─────────────────────────────────────────────

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error('[startup] FATAL: OPENAI_API_KEY environment variable is not set.');
  process.exit(1);
}

const PORT    = parseInt(process.env.PORT || '3000', 10);
const FAQ_DIR = process.env.FAQ_DIR
  ? path.resolve(process.env.FAQ_DIR)
  : path.join(path.dirname(fileURLToPath(import.meta.url)), 'faqs');

const TOP_K_MIN = 1;
const TOP_K_MAX = 10;
const DEFAULT_TOP_K = 4;

// ─── Initialise RAG ──────────────────────────────────────────────────────────

initOpenAI(OPENAI_API_KEY);

let ready = false;

console.log(`[startup] Ingesting FAQ documents from: ${FAQ_DIR}`);
ingestDirectory(FAQ_DIR)
  .then(() => {
    ready = true;
    console.log('[startup] Ingestion complete. Server is ready.');
  })
  .catch((err) => {
    console.error('[startup] FATAL: Ingestion failed:', err.message);
    process.exit(1);
  });

// ─── Express setup ───────────────────────────────────────────────────────────

const app = express();
app.use(express.json());

// Request logger (minimal)
app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// ─── Routes ──────────────────────────────────────────────────────────────────

/**
 * GET /health
 * Simple liveness check. Also reports whether ingestion has completed.
 */
app.get('/health', (_req, res) => {
  res.status(200).json({ status: 'ok' });
});

/**
 * POST /ask
 * Body: { question: string, top_k?: number }
 * Returns: { answer: string, sources: string[] }
 */
app.post('/ask', async (req, res) => {
  // Guard: ingestion still in progress
  if (!ready) {
    return res.status(503).json({ error: 'Service is still initializing. Try again shortly.' });
  }

  const { question, top_k } = req.body;

  // ── Input validation ──────────────────────────────────────────────────────

  if (typeof question !== 'string' || question.trim().length === 0) {
    return res.status(400).json({
      error: 'Invalid input: "question" must be a non-empty string.',
    });
  }

  let topK = DEFAULT_TOP_K;
  if (top_k !== undefined) {
    if (!Number.isInteger(top_k) || top_k < TOP_K_MIN || top_k > TOP_K_MAX) {
      return res.status(400).json({
        error: `Invalid input: "top_k" must be an integer between ${TOP_K_MIN} and ${TOP_K_MAX}.`,
      });
    }
    topK = top_k;
  }

  // ── RAG pipeline ─────────────────────────────────────────────────────────

  try {
    const { answer, sources } = await ask(question.trim(), topK);
    // Deterministic key order: answer first, then sources
    return res.status(200).json({ answer, sources });
  } catch (err) {
    console.error('[/ask] Error:', err.message);
    return res.status(500).json({ error: 'Internal server error. Please try again.' });
  }
});

// 404 fallback
app.use((_req, res) => {
  res.status(404).json({ error: 'Not found.' });
});

// ─── Start ───────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[startup] HTTP server listening on http://localhost:${PORT}`);
});

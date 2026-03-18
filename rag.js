/**
 * rag.js — Core RAG logic
 *
 * Responsibilities:
 *   1. Chunk markdown files into ~200-char segments
 *   2. Embed chunks via OpenAI text-embedding-3-small
 *   3. Retrieve top-k chunks by cosine similarity at query time
 *   4. Generate an answer with GPT-4o-mini, instructing it to cite sources
 *
 * Design trade-offs:
 *   - In-memory vector store: fine for small corpora (<1000 chunks); swap for
 *     pgvector / Chroma / Pinecone at scale.
 *   - text-embedding-3-small: cheap and fast; text-embedding-3-large is more
 *     accurate but ~6x the cost.
 *   - Chunk size ~200 chars with a 40-char overlap so sentences aren't split
 *     cleanly across boundaries, preserving context.
 *   - Single-pass ingestion: call ingestDirectory() once at startup; no
 *     persistence — everything lives in the JS process. Add a JSON cache file
 *     if cold-start latency becomes a problem.
 */

import fs from 'fs/promises';
import path from 'path';
import OpenAI from 'openai';

const CHUNK_SIZE = 200;      // target chars per chunk
const CHUNK_OVERLAP = 40;    // overlap to preserve cross-boundary context
const DEFAULT_TOP_K = 4;

let openai;
/** @type {Array<{filename: string, text: string, embedding: number[]}>} */
let vectorStore = [];

// ─── Initialisation ──────────────────────────────────────────────────────────

export function initOpenAI(apiKey) {
  openai = new OpenAI({ apiKey });
}

// ─── Chunking ─────────────────────────────────────────────────────────────────

/**
 * Split text into overlapping chunks of ~CHUNK_SIZE chars.
 * Splitting on sentence boundaries ('. ') where possible keeps chunks coherent.
 */
function chunkText(text) {
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    let end = start + CHUNK_SIZE;

    // Don't overshoot
    if (end >= text.length) {
      chunks.push(text.slice(start).trim());
      break;
    }

    // Try to break on a sentence boundary within the last 60 chars of the window
    const window = text.slice(start, end);
    const lastPeriod = window.lastIndexOf('. ');
    if (lastPeriod > CHUNK_SIZE - 60) {
      end = start + lastPeriod + 2; // include the period + space
    }

    const chunk = text.slice(start, end).trim();
    if (chunk.length > 0) chunks.push(chunk);

    start = end - CHUNK_OVERLAP; // step back by overlap amount
  }

  return chunks;
}

// ─── Embedding ────────────────────────────────────────────────────────────────

/**
 * Embed a single string. Throws on OpenAI error (caught at call site).
 */
async function embed(text) {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  });
  return response.data[0].embedding;
}

/**
 * Embed multiple strings in one batched API call (cheaper, faster).
 */
async function embedBatch(texts) {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: texts,
  });
  // API returns embeddings in the same order as input
  return response.data.map((d) => d.embedding);
}

// ─── Ingestion ────────────────────────────────────────────────────────────────

/**
 * Read all .md files in `dir`, chunk them, embed all chunks in one batch call,
 * and populate the in-memory vector store.
 *
 * @param {string} dir  Absolute or relative path to the faqs/ directory
 */
export async function ingestDirectory(dir) {
  const files = (await fs.readdir(dir)).filter((f) => f.endsWith('.md'));

  if (files.length === 0) {
    throw new Error(`No .md files found in ${dir}`);
  }

  console.log(`[ingest] Found ${files.length} files: ${files.join(', ')}`);

  // Build a flat list of {filename, text} pairs for all chunks
  const allChunks = [];
  for (const file of files) {
    const content = await fs.readFile(path.join(dir, file), 'utf8');
    const chunks = chunkText(content);
    console.log(`[ingest]   ${file} → ${chunks.length} chunks`);
    for (const text of chunks) {
      allChunks.push({ filename: file, text });
    }
  }

  // Single batched embedding call for all chunks
  console.log(`[ingest] Embedding ${allChunks.length} chunks...`);
  const embeddings = await embedBatch(allChunks.map((c) => c.text));

  vectorStore = allChunks.map((chunk, i) => ({
    filename: chunk.filename,
    text: chunk.text,
    embedding: embeddings[i],
  }));

  console.log(`[ingest] Done. Vector store has ${vectorStore.length} entries.`);
}

// ─── Retrieval ────────────────────────────────────────────────────────────────

/**
 * Cosine similarity between two equal-length vectors.
 * Using the dot-product-over-magnitudes formula.
 */
function cosineSimilarity(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot  += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

/**
 * Retrieve the top-k most similar chunks to the query.
 * Returns chunks sorted descending by score.
 *
 * @param {string} query
 * @param {number} topK
 * @returns {Promise<Array<{filename: string, text: string, score: number}>>}
 */
async function retrieve(query, topK = DEFAULT_TOP_K) {
  const queryEmbedding = await embed(query);

  const scored = vectorStore.map((entry) => ({
    filename: entry.filename,
    text: entry.text,
    score: cosineSimilarity(queryEmbedding, entry.embedding),
  }));

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}

// ─── Generation ───────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are a helpful assistant that answers questions strictly using the provided context.
Rules:
- Base your answer only on the context below; do not invent facts.
- You MUST reference at least two distinct source filenames in your answer when they are available in the context.
- Cite sources inline like this: (source: filename.md).
- If the context does not contain enough information to answer, say so clearly.`;

/**
 * Full RAG pipeline: retrieve → generate → return answer + sources.
 *
 * @param {string} question
 * @param {number} topK
 * @returns {Promise<{answer: string, sources: string[]}>}
 */
export async function ask(question, topK = DEFAULT_TOP_K) {
  if (vectorStore.length === 0) {
    throw new Error('Vector store is empty. Run ingestDirectory() first.');
  }

  const chunks = await retrieve(question, topK);

  // Build context block, labelling each chunk with its source
  const contextBlock = chunks
    .map((c, i) => `[${i + 1}] (source: ${c.filename})\n${c.text}`)
    .join('\n\n');

  const userMessage = `Context:\n${contextBlock}\n\nQuestion: ${question}`;

  const completion = await openai.chat.completions.create({
    model: process.env.COMPLETION_MODEL || 'gpt-4o-mini',
    temperature: 0,          // deterministic answers
    max_tokens: 512,
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user',   content: userMessage },
    ],
  });

  const answer = completion.choices[0].message.content.trim();

  // Deduplicate sources from retrieved chunks (maintain retrieval rank order)
  const sources = [...new Set(chunks.map((c) => c.filename))];

  return { answer, sources };
}

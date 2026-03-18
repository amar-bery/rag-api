/**
 * ingest.js — Standalone ingestion script
 *
 * Run with:  node ingest.js
 *
 * Useful for verifying that your FAQ files chunk and embed correctly
 * without starting the HTTP server. Prints chunk count per file and
 * a sample retrieval result for a hardcoded test query.
 */

import 'dotenv/config';
import path from 'path';
import { fileURLToPath } from 'url';
import { initOpenAI, ingestDirectory, ask } from './rag.js';

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error('FATAL: OPENAI_API_KEY is not set.');
  process.exit(1);
}

const FAQ_DIR = process.env.FAQ_DIR
  ? path.resolve(process.env.FAQ_DIR)
  : path.join(path.dirname(fileURLToPath(import.meta.url)), 'faqs');

initOpenAI(OPENAI_API_KEY);

console.log('Starting ingestion test...\n');

await ingestDirectory(FAQ_DIR);

console.log('\n--- Sample retrieval test ---');
const testQuestion = 'How do I reset my password and enable two-factor authentication?';
console.log(`Question: ${testQuestion}\n`);

const result = await ask(testQuestion, 4);
console.log('Answer:\n', result.answer);
console.log('\nSources:', result.sources);

# 🧠 local-semantic-cache

[![NPM Version](https://img.shields.io/npm/v/local-semantic-cache.svg?style=flat-square&color=blue)](https://www.npmjs.com/package/local-semantic-cache)
[![Downloads](https://img.shields.io/npm/dm/local-semantic-cache.svg?style=flat-square)](https://www.npmjs.com/package/local-semantic-cache)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Ezeko/local-semantic-cache/ci.yml?branch=main&style=flat-square)](https://github.com/Ezeko/local-semantic-cache/actions)
[![License](https://img.shields.io/npm/l/local-semantic-cache.svg?style=flat-square)](https://github.com/Ezeko/local-semantic-cache/blob/main/LICENCE)
[![GitHub Stars](https://img.shields.io/github/stars/Ezeko/local-semantic-cache.svg?style=flat-square&label=stars&color=yellow)](https://github.com/Ezeko/local-semantic-cache/stargazers)

> **Ultra-fast, zero-dependency local semantic cache for LLM/AI prompt responses with built-in text vectorization, similarity search, and file persistence.**

Traditional caching (like key-value maps or standard Redis caches) requires exact string matches. A query like *"What is the capital of Nigeria?"* won't hit a cache keyed with *"Tell me the capital of Nigeria"*. 

`local-semantic-cache` resolves this by encoding prompt query keys into a **vector space** and using **cosine similarity** to retrieve cached responses for semantically identical queries.

```text
[Incoming Prompt] ──► Tokenize & Embed ──► Vector [0.2, 0.8, ...] 
                                                  │
                                                  ├──► Cosine Similarity (Dot Product)
                                                  │
[Cached Prompts]  ◄───────────────────────────────┴──► Hit >= 0.90 ? Serve Cached Response
```

---

## Key Features

*   **⚡ Sub-millisecond Execution**: Core vector search calculations are written in vanilla JS/TS, utilizing unit L2 normalized vectors where cosine similarity simplifies to a fast, raw dot-product multiplication.
*   **🔌 Pluggable Embeddings**: Integrates seamlessly with cloud embedding APIs (OpenAI, Gemini, Cohere) or local TensorFlow.js/Transformers pipelines.
*   **📦 Out-of-the-Box Fallback**: Includes a fast, zero-setup, built-in text vectorizer (using the Hashing Trick/Feature Hashing unigrams and bigrams) so the package functions instantly without requiring cloud keys.
*   **💾 File Persistence**: Auto-saves and loads cache collections to and from local disk storage.
*   **🛡️ Type Safe**: Built entirely in TypeScript with full type declarations.

---

## Installation

Install via your preferred package manager:

```bash
npm install local-semantic-cache
# or
yarn add local-semantic-cache
# or
pnpm add local-semantic-cache
# or
bun add local-semantic-cache
```

---

## Quick Start (Zero-Setup Local Mode)

By default, `local-semantic-cache` uses its built-in string vectorizer. No API keys are required:

```typescript
import { SemanticCache } from 'local-semantic-cache';

const cache = new SemanticCache({
  minSimilarity: 0.60,             // Cosine threshold between 0.0 and 1.0 (0.60 is recommended for sparse local vectorizer)
  filePath: './data/cache-db.json' // Path to persist logs on disk
});

// Seed the cache
await cache.set('What is the capital of Nigeria?', 'Abuja');

// Query with a semantically similar prompt
const response = await cache.get('Tell me the capital of Nigeria');
console.log(response); // "Abuja" (Cache Hit!)

// Query out of context
const invalidResponse = await cache.get('What is the temperature in Abuja?');
console.log(invalidResponse); // null (Cache Miss)
```

---

## Integration with LLM APIs

### 1. OpenAI Embeddings Example

```typescript
import { SemanticCache } from 'local-semantic-cache';
import OpenAI from 'openai';

const openai = new OpenAI();

const cache = new SemanticCache({
  dimensions: 1536, // text-embedding-3-small dimension
  minSimilarity: 0.92,
  embeddingFunction: async (text) => {
    const res = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: text
    });
    return res.data[0].embedding;
  }
});

await cache.set('Who wrote Hamlet?', 'William Shakespeare');

const result = await cache.get('Who is the author of Hamlet?');
console.log(result); // "William Shakespeare"
```

### 2. Google Gemini Embeddings Example

```typescript
import { SemanticCache } from 'local-semantic-cache';
import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI();

const cache = new SemanticCache({
  dimensions: 768, // text-embedding-004 dimension
  minSimilarity: 0.88,
  embeddingFunction: async (text) => {
    const res = await ai.models.embedContent({
      model: 'text-embedding-004',
      contents: text
    });
    return res.embedding.values;
  }
});
```

---

## API Reference

### `new SemanticCache(options)`
Creates a cache instance.
*   `options.embeddingFunction`: A sync/async function taking `(text: string)` and returning `number[]`.
*   `options.minSimilarity`: Similarity float cutoff value (default `0.90`).
*   `options.dimensions`: Hashed vector output slots length (default `128`).
*   `options.filePath`: Storage path to read/write JSON files on disk.

### `cache.set(prompt, response, metadata?)`
Saves a record to the cache.

### `cache.get(prompt)`
Returns the cached `response` string if similarity matches, or `null`.

### `cache.getDetailed(prompt)`
Returns detailed matching payload containing `{ response, similarity, metadata }` or `null`.

### `cache.clear()`
Wipes the cache memory pool and removes local storage files.

---

## Contributing

We welcome community contributions! Please read our [Contributing Guide](CONTRIBUTING.md) to get started.

---

## License

MIT © [Ezekiel Adejobi](https://github.com/ezeko)

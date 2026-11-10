# 🧠 local-semantic-cache

[![NPM Version](https://img.shields.io/npm/v/local-semantic-cache?style=flat-square)](https://www.npmjs.com/package/local-semantic-cache)
[![NPM Monthly Downloads](https://img.shields.io/npm/dm/local-semantic-cache?style=flat-square)](https://www.npmjs.com/package/local-semantic-cache)
[![License](https://img.shields.io/npm/l/local-semantic-cache?style=flat-square)](https://github.com/Ezeko/semanticcache/blob/main/LICENCE)
[![GitHub Stars](https://img.shields.io/github/stars/Ezeko/semanticcache?style=flat-square&logo=github)](https://github.com/Ezeko/semanticcache/stargazers)

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

*   **🚀 1-Line Drop-In OpenAI Wrapper**: Seamlessly wraps the official `OpenAI` client (`chat.completions` and `completions`). Zero pipeline refactoring — requests check the semantic cache first before reaching external APIs.
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

## 🚀 1-Line Drop-In OpenAI Wrapper

Wrap the official OpenAI client in one line without rewriting your existing pipelines. All calls to `openai.chat.completions.create` and legacy `openai.completions.create` automatically query the local semantic cache before hitting the external OpenAI API.

### Basic Usage

```typescript
import OpenAI from 'openai';
import { wrapOpenAI } from 'local-semantic-cache/openai';

// 1. Wrap your existing OpenAI client instance
const openai = wrapOpenAI(new OpenAI({ apiKey: process.env.OPENAI_API_KEY }));

// 2. Use your client exactly as you did before
const completion = await openai.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'What is the capital of France?' }],
});

console.log(completion.choices[0].message.content); // "Paris is the capital of France."

// 3. Second query with semantically similar wording hits cache instantly (< 1ms)!
const cached = await openai.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'Tell me the capital of France' }],
});

console.log(cached.choices[0].message.content); // Served from cache!
console.log((cached as any)._cached);           // true
console.log((cached as any)._similarity);       // 0.94
```

### Production Setup (Persistence & Observability Hooks)

You can pass configuration options or supply a pre-configured `SemanticCache` instance with disk persistence and telemetry hooks:

```typescript
import OpenAI from 'openai';
import { wrapOpenAI } from 'local-semantic-cache/openai';
import { SemanticCache } from 'local-semantic-cache';

// Pre-configure cache with local file storage
const persistentCache = new SemanticCache({
  minSimilarity: 0.85,
  filePath: './data/openai-cache.json', // Automatically saves to disk
});

const openai = wrapOpenAI(new OpenAI(), {
  cache: persistentCache,
  onCacheHit: (prompt, match) => {
    console.log(`[Cache Hit] Similarity: ${(match.similarity * 100).toFixed(1)}% | Prompt: "${prompt}"`);
  },
  onCacheMiss: (prompt) => {
    console.log(`[Cache Miss] Forwarding to OpenAI: "${prompt}"`);
  },
});
```

> **Note on Streaming**: Requests with `{ stream: true }` automatically bypass semantic caching and stream live chunks directly from the OpenAI API without interruption.

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

### `wrapOpenAI(client, options?)`
Wraps an OpenAI SDK client instance with transparent, zero-overhead semantic caching.
*   `client`: An instance of `new OpenAI()`.
*   `options.minSimilarity`: Similarity threshold (`0.0` to `1.0`) to trigger a hit (default `0.90`).
*   `options.cache`: Custom `SemanticCache` instance (optional, defaults to an in-memory instance).
*   `options.onCacheHit`: Optional callback invoked on hit: `(prompt, cachedEntry) => void`.
*   `options.onCacheMiss`: Optional callback invoked on miss: `(prompt) => void`.
*   `options.extractPrompt`: Optional custom parser function `(body) => string | null`.
*   `client.semanticCache`: Read-only property on the wrapped client exposing the underlying cache.

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

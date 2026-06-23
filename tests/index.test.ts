import { test } from 'node:test';
import * as assert from 'node:assert';
import * as path from 'path';
import * as fs from 'fs';
import { SemanticCache } from '../src/index';

test('semanticcache unit test suite', async (t) => {
  await t.test('retrieves exact prompt match responses', async () => {
    const cache = new SemanticCache({ minSimilarity: 0.95 });
    await cache.set('Who is CEO of Google?', 'Sundar Pichai');

    const result = await cache.get('Who is CEO of Google?');
    assert.strictEqual(result, 'Sundar Pichai');
  });

  await t.test('retrieves semantically similar prompt match responses', async () => {
    const cache = new SemanticCache({ minSimilarity: 0.60 });
    await cache.set('What is the capital of France?', 'Paris');

    // A similar query that matches semantically
    const result = await cache.get('Tell me the capital of France');
    assert.strictEqual(result, 'Paris');
  });

  await t.test('respects minSimilarity threshold filters', async () => {
    const cache = new SemanticCache({ minSimilarity: 0.80 });
    await cache.set('What is the capital of France?', 'Paris');

    // This should fail to hit the cache due to strict similarity rules
    const result = await cache.get('How far is Paris from London?');
    assert.strictEqual(result, null);
  });

  await t.test('supports custom embedding functions', async () => {
    const customEmbedding = (text: string) => {
      // Mock constant vector
      return [0.5, 0.5, 0.5, 0.5];
    };

    const cache = new SemanticCache({
      embeddingFunction: customEmbedding,
      dimensions: 4,
      minSimilarity: 0.99,
    });

    await cache.set('prompt 1', 'response A');
    const result = await cache.get('prompt 2'); // Will match because the embedding is constant
    assert.strictEqual(result, 'response A');
  });

  await t.test('persists cache to disk and loads it back', async () => {
    const filePath = path.join(__dirname, 'test-cache.json');
    
    // Clean up previous test files if present
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }

    const firstCache = new SemanticCache({ filePath, minSimilarity: 0.95 });
    await firstCache.set('Hello there!', 'General Kenobi!');

    // Initialize second cache instance with same file path to verify loading
    const secondCache = new SemanticCache({ filePath, minSimilarity: 0.95 });
    const result = await secondCache.get('Hello there!');
    assert.strictEqual(result, 'General Kenobi!');

    // Clean up
    secondCache.clear();
    assert.strictEqual(fs.existsSync(filePath), false);
  });
});

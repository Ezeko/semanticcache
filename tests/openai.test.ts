import { test } from 'node:test';
import * as assert from 'node:assert';
import { wrapOpenAI } from '../src/openai';
import { SemanticCache } from '../src/index';

test('wrapOpenAI drop-in wrapper test suite', async (t) => {
  await t.test('intercepts chat.completions.create on cache miss and caches response', async () => {
    let apiCallCount = 0;
    const mockOpenAI = {
      chat: {
        completions: {
          create: async (params: any) => {
            apiCallCount++;
            return {
              id: 'chatcmpl-test-123',
              object: 'chat.completion',
              created: 1677652288,
              model: params.model,
              choices: [
                {
                  index: 0,
                  message: {
                    role: 'assistant',
                    content: 'Paris is the capital of France.',
                  },
                  finish_reason: 'stop',
                },
              ],
              usage: { prompt_tokens: 10, completion_tokens: 8, total_tokens: 18 },
            };
          },
        },
      },
      models: {
        list: async () => [{ id: 'gpt-4o' }],
      },
    };

    let missCount = 0;
    let hitCount = 0;
    const cache = new SemanticCache({ minSimilarity: 0.60 });
    const client = wrapOpenAI(mockOpenAI, {
      cache,
      onCacheMiss: () => {
        missCount++;
      },
      onCacheHit: () => {
        hitCount++;
      },
    });

    // 1. Initial request (Cache Miss)
    const res1 = await client.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'What is the capital of France?' }],
    });

    assert.strictEqual(apiCallCount, 1);
    assert.strictEqual(missCount, 1);
    assert.strictEqual(hitCount, 0);
    assert.strictEqual(res1.choices[0].message.content, 'Paris is the capital of France.');

    // 2. Semantically similar request (Cache Hit)
    const res2 = await client.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'Tell me the capital of France' }],
    });

    // Should NOT have made a second API call
    assert.strictEqual(apiCallCount, 1, 'API call count should remain 1 on cache hit');
    assert.strictEqual(hitCount, 1);
    assert.strictEqual(res2.choices[0].message.content, 'Paris is the capital of France.');
    assert.strictEqual((res2 as any)._cached, true);
    assert.ok(typeof (res2 as any)._similarity === 'number' && (res2 as any)._similarity >= 0.60);
  });

  await t.test('handles multi-turn conversation messages seamlessly', async () => {
    let callCount = 0;
    const mockClient = {
      chat: {
        completions: {
          create: async (params: any) => {
            callCount++;
            return {
              id: 'chatcmpl-multi-turn',
              choices: [
                {
                  message: {
                    role: 'assistant',
                    content: 'TypeScript adds static type definitions to JavaScript.',
                  },
                },
              ],
            };
          },
        },
      },
    };

    const client = wrapOpenAI(mockClient, { minSimilarity: 0.85 });

    const messages = [
      { role: 'system', content: 'You are a programming tutor.' },
      { role: 'user', content: 'What is TypeScript?' },
    ];

    const first = await client.chat.completions.create({ model: 'gpt-4o', messages });
    assert.strictEqual(callCount, 1);
    assert.strictEqual(first.choices[0].message.content, 'TypeScript adds static type definitions to JavaScript.');

    // Same multi-turn conversation should hit cache
    const second = await client.chat.completions.create({ model: 'gpt-4o', messages });
    assert.strictEqual(callCount, 1);
    assert.strictEqual(second.choices[0].message.content, 'TypeScript adds static type definitions to JavaScript.');
    assert.strictEqual((second as any)._cached, true);
  });

  await t.test('handles legacy completions.create seamlessly', async () => {
    let callCount = 0;
    const mockClient = {
      completions: {
        create: async (params: any) => {
          callCount++;
          return {
            id: 'cmpl-legacy-1',
            choices: [
              {
                text: '42 is the answer to everything.',
                index: 0,
              },
            ],
          };
        },
      },
    };

    const client = wrapOpenAI(mockClient, { minSimilarity: 0.90 });

    const res1 = await client.completions.create({
      model: 'gpt-3.5-turbo-instruct',
      prompt: 'What is the answer to life, the universe, and everything?',
    });

    assert.strictEqual(callCount, 1);
    assert.strictEqual(res1.choices[0].text, '42 is the answer to everything.');

    const res2 = await client.completions.create({
      model: 'gpt-3.5-turbo-instruct',
      prompt: 'What is the answer to life, the universe, and everything?',
    });

    assert.strictEqual(callCount, 1);
    assert.strictEqual(res2.choices[0].text, '42 is the answer to everything.');
    assert.strictEqual((res2 as any)._cached, true);
  });

  await t.test('bypasses cache when stream: true is specified', async () => {
    let callCount = 0;
    const mockClient = {
      chat: {
        completions: {
          create: async (params: any) => {
            callCount++;
            return { streamData: true };
          },
        },
      },
    };

    const client = wrapOpenAI(mockClient);

    await client.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'Stream this' }],
      stream: true,
    });

    assert.strictEqual(callCount, 1);

    // Call again with stream: true - should bypass cache every time
    await client.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'Stream this' }],
      stream: true,
    });

    assert.strictEqual(callCount, 2);
  });

  await t.test('preserves non-chat methods and properties with correct this context', async () => {
    const mockClient = {
      apiKey: 'sk-mock-key-12345',
      organization: 'org-test',
      models: {
        list: async function () {
          return { data: [{ id: 'gpt-4o' }] };
        },
      },
    };

    const client = wrapOpenAI(mockClient);

    assert.strictEqual(client.apiKey, 'sk-mock-key-12345');
    assert.strictEqual(client.organization, 'org-test');

    const models = await client.models.list();
    assert.strictEqual(models.data[0].id, 'gpt-4o');
  });

  await t.test('supports custom extractPrompt hook', async () => {
    let callCount = 0;
    const mockClient = {
      chat: {
        completions: {
          create: async (params?: any) => {
            callCount++;
            return { choices: [{ message: { content: 'Custom output' } }] };
          },
        },
      },
    };

    const client = wrapOpenAI(mockClient, {
      extractPrompt: (body) => `custom-prefix:${body.customKey}`,
    });

    await client.chat.completions.create({ customKey: 'topic-alpha' });
    assert.strictEqual(callCount, 1);

    await client.chat.completions.create({ customKey: 'topic-alpha' });
    assert.strictEqual(callCount, 1);
  });

  await t.test('exposes semanticCache instance for cache operations', async () => {
    const mockClient = {
      chat: {
        completions: {
          create: async (params?: any) => ({
            choices: [{ message: { content: 'Response' } }],
          }),
        },
      },
    };

    const client = wrapOpenAI(mockClient);
    assert.ok(client.semanticCache instanceof SemanticCache);

    await client.chat.completions.create({
      messages: [{ role: 'user', content: 'test' }],
    });

    assert.strictEqual(client.semanticCache.getAll().length, 1);
    client.semanticCache.clear();
    assert.strictEqual(client.semanticCache.getAll().length, 0);
  });
});

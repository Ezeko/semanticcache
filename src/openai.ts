import { SemanticCache, SemanticCacheOptions } from './index';

/**
 * Configuration options for the OpenAI semantic cache wrapper.
 */
export interface WrapOpenAIOptions {
  /**
   * Dedicated SemanticCache instance. If omitted, a new in-memory
   * SemanticCache instance is instantiated with the provided options.
   */
  cache?: SemanticCache;

  /**
   * Minimum cosine similarity score threshold (0.0 to 1.0) to register a cache hit.
   * Defaults to 0.90.
   */
  minSimilarity?: number;

  /**
   * Custom prompt extractor hook to extract the query text from the request body.
   */
  extractPrompt?: (body: any) => string | null;

  /**
   * Hook called whenever a cached response is returned.
   */
  onCacheHit?: (prompt: string, cachedEntry: { response: string; similarity: number; metadata?: any }) => void;

  /**
   * Hook called whenever no matching cache entry is found and the request hits OpenAI.
   */
  onCacheMiss?: (prompt: string) => void;
}

/**
 * Extended properties exposed on a wrapped OpenAI client.
 */
export interface WrappedOpenAIClient {
  /**
   * Direct access to the active SemanticCache instance.
   */
  readonly semanticCache: SemanticCache;
}

/**
 * Extracts a prompt string from OpenAI Chat Completion request parameters.
 */
function extractPromptFromChat(body: any): string | null {
  if (!body || !Array.isArray(body.messages) || body.messages.length === 0) {
    return null;
  }

  // Fast path for single user message queries
  if (body.messages.length === 1 && body.messages[0].role === 'user') {
    return formatMessageContent(body.messages[0].content);
  }

  // For multi-turn chats, serialize conversation roles and content
  return body.messages
    .map((msg: any) => {
      const role = msg.role || 'user';
      const text = formatMessageContent(msg.content);
      return `${role}: ${text}`;
    })
    .join('\n');
}

/**
 * Handles string or structured text content arrays (e.g., multimodal text blocks).
 */
function formatMessageContent(content: any): string {
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .filter((part: any) => part && part.type === 'text' && typeof part.text === 'string')
      .map((part: any) => part.text)
      .join(' ');
  }
  return content ? String(content) : '';
}

/**
 * Extracts prompt text from legacy OpenAI text completions.
 */
function extractPromptFromCompletion(body: any): string | null {
  if (!body || !body.prompt) {
    return null;
  }
  if (typeof body.prompt === 'string') {
    return body.prompt;
  }
  if (Array.isArray(body.prompt)) {
    return body.prompt.join(' ');
  }
  return String(body.prompt);
}

/**
 * Wraps an OpenAI client instance with transparent semantic caching.
 *
 * Intercepts `chat.completions.create` and `completions.create` calls,
 * checking the semantic cache first before forwarding requests to the API.
 *
 * @example
 * ```ts
 * import OpenAI from 'openai';
 * import { wrapOpenAI } from 'local-semantic-cache/openai';
 *
 * const openai = wrapOpenAI(new OpenAI({ apiKey: process.env.OPENAI_API_KEY }));
 * const response = await openai.chat.completions.create({
 *   model: 'gpt-4o',
 *   messages: [{ role: 'user', content: 'What is the capital of France?' }],
 * });
 * ```
 */
export function wrapOpenAI<T extends object>(client: T, options: WrapOpenAIOptions = {}): T & WrappedOpenAIClient {
  const cache = options.cache ?? new SemanticCache({
    minSimilarity: options.minSimilarity ?? 0.90,
  });

  return new Proxy(client, {
    get(target: any, prop: string | symbol, receiver: any) {
      if (prop === 'semanticCache') {
        return cache;
      }

      if (prop === 'chat' && target.chat) {
        return wrapChatNamespace(target.chat, cache, options);
      }

      if (prop === 'completions' && target.completions) {
        return wrapLegacyCompletionsNamespace(target.completions, cache, options);
      }

      const value = Reflect.get(target, prop, receiver);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  }) as T & WrappedOpenAIClient;
}

/**
 * Intercepts the `chat` namespace to target `chat.completions`.
 */
function wrapChatNamespace(chat: any, cache: SemanticCache, options: WrapOpenAIOptions) {
  return new Proxy(chat, {
    get(chatTarget: any, chatProp: string | symbol, chatReceiver: any) {
      if (chatProp === 'completions' && chatTarget.completions) {
        return wrapChatCompletions(chatTarget.completions, cache, options);
      }
      const val = Reflect.get(chatTarget, chatProp, chatReceiver);
      return typeof val === 'function' ? val.bind(chatTarget) : val;
    },
  });
}

/**
 * Wraps `chat.completions.create` with semantic cache check and response storage.
 */
function wrapChatCompletions(completions: any, cache: SemanticCache, options: WrapOpenAIOptions) {
  const originalCreate = completions.create;

  return new Proxy(completions, {
    get(target: any, prop: string | symbol, receiver: any) {
      if (prop === 'create' && typeof originalCreate === 'function') {
        return async function create(body: any, requestOptions?: any) {
          // Streaming responses bypass the semantic cache
          if (body?.stream) {
            return originalCreate.call(completions, body, requestOptions);
          }

          const prompt = options.extractPrompt
            ? options.extractPrompt(body)
            : extractPromptFromChat(body);

          if (!prompt) {
            return originalCreate.call(completions, body, requestOptions);
          }

          // Check semantic cache for semantically similar cached answers
          const cached = await cache.getDetailed(prompt);
          if (cached) {
            const raw = cached.metadata?.raw;
            if (options.onCacheHit) {
              options.onCacheHit(prompt, cached);
            }

            return {
              id: raw?.id ? `cached-${raw.id}` : `chatcmpl-cached-${Date.now()}`,
              object: 'chat.completion',
              created: raw?.created || Math.floor(Date.now() / 1000),
              model: body?.model || raw?.model || 'cached-model',
              choices: [
                {
                  index: 0,
                  message: {
                    role: 'assistant',
                    content: cached.response,
                    refusal: null,
                  },
                  logprobs: null,
                  finish_reason: 'stop',
                },
              ],
              usage: raw?.usage || {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
              },
              _cached: true,
              _similarity: cached.similarity,
            };
          }

          // Cache miss: execute OpenAI API request
          if (options.onCacheMiss) {
            options.onCacheMiss(prompt);
          }

          const response = await originalCreate.call(completions, body, requestOptions);

          // Store response text in cache on success
          const assistantContent = response?.choices?.[0]?.message?.content;
          if (typeof assistantContent === 'string' && assistantContent.length > 0) {
            await cache.set(prompt, assistantContent, {
              raw: response,
              model: body?.model,
              cachedAt: Date.now(),
            });
          }

          return response;
        };
      }

      const val = Reflect.get(target, prop, receiver);
      return typeof val === 'function' ? val.bind(target) : val;
    },
  });
}

/**
 * Wraps legacy text `completions.create`.
 */
function wrapLegacyCompletionsNamespace(completions: any, cache: SemanticCache, options: WrapOpenAIOptions) {
  const originalCreate = completions.create;

  return new Proxy(completions, {
    get(target: any, prop: string | symbol, receiver: any) {
      if (prop === 'create' && typeof originalCreate === 'function') {
        return async function create(body: any, requestOptions?: any) {
          if (body?.stream) {
            return originalCreate.call(completions, body, requestOptions);
          }

          const prompt = options.extractPrompt
            ? options.extractPrompt(body)
            : extractPromptFromCompletion(body);

          if (!prompt) {
            return originalCreate.call(completions, body, requestOptions);
          }

          const cached = await cache.getDetailed(prompt);
          if (cached) {
            const raw = cached.metadata?.raw;
            if (options.onCacheHit) {
              options.onCacheHit(prompt, cached);
            }

            return {
              id: raw?.id ? `cached-${raw.id}` : `cmpl-cached-${Date.now()}`,
              object: 'text_completion',
              created: raw?.created || Math.floor(Date.now() / 1000),
              model: body?.model || raw?.model || 'cached-model',
              choices: [
                {
                  text: cached.response,
                  index: 0,
                  logprobs: null,
                  finish_reason: 'stop',
                },
              ],
              usage: raw?.usage || {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
              },
              _cached: true,
              _similarity: cached.similarity,
            };
          }

          if (options.onCacheMiss) {
            options.onCacheMiss(prompt);
          }

          const response = await originalCreate.call(completions, body, requestOptions);

          const choiceText = response?.choices?.[0]?.text;
          if (typeof choiceText === 'string' && choiceText.length > 0) {
            await cache.set(prompt, choiceText, {
              raw: response,
              model: body?.model,
              cachedAt: Date.now(),
            });
          }

          return response;
        };
      }

      const val = Reflect.get(target, prop, receiver);
      return typeof val === 'function' ? val.bind(target) : val;
    },
  });
}


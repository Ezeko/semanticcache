import * as fs from 'fs';
import * as path from 'path';
import { embed } from './vectorizer';

export interface CacheEntry {
  prompt: string;
  response: string;
  vector: number[];
  metadata?: any;
  timestamp: number;
}

export type EmbeddingFunction = (text: string) => number[] | Promise<number[]>;

export interface SemanticCacheOptions {
  /**
   * Pluggable custom embedding function (e.g. OpenAI or Gemini).
   * Defaults to local hashing vectorizer.
   */
  embeddingFunction?: EmbeddingFunction;
  /**
   * Vector dimensions length. Defaults to 128.
   */
  dimensions?: number;
  /**
   * Similarity score threshold threshold (between 0.0 and 1.0).
   * Cache hits are returned if similarity is greater than this value.
   * Defaults to 0.90.
   */
  minSimilarity?: number;
  /**
   * Local file path to persist cache entries.
   */
  filePath?: string;
}

export class SemanticCache {
  private cache: CacheEntry[] = [];
  private embeddingFunction: EmbeddingFunction;
  private dimensions: number;
  private minSimilarity: number;
  private filePath?: string;

  constructor(options: SemanticCacheOptions = {}) {
    this.dimensions = options.dimensions ?? 128;
    this.minSimilarity = options.minSimilarity ?? 0.90;
    this.filePath = options.filePath;

    // Use pluggable vectorizer or fall back to native TF-IDF hashing
    this.embeddingFunction = options.embeddingFunction ?? ((text: string) => embed(text, this.dimensions));

    // Load persisted cache if file path exists
    this.load();
  }

  /**
   * Store a prompt and its corresponding LLM response in the cache
   */
  async set(prompt: string, response: string, metadata?: any): Promise<void> {
    if (typeof prompt !== 'string' || typeof response !== 'string') {
      return;
    }

    const vector = await this.embeddingFunction(prompt);
    
    // Check if prompt already exists to avoid duplication
    const existingIndex = this.cache.findIndex((entry) => entry.prompt === prompt);
    const newEntry: CacheEntry = {
      prompt,
      response,
      vector,
      metadata,
      timestamp: Date.now(),
    };

    if (existingIndex !== -1) {
      this.cache[existingIndex] = newEntry;
    } else {
      this.cache.push(newEntry);
    }

    this.save();
  }

  /**
   * Retrieve response for prompt if it exceeds the similarity threshold
   */
  async get(prompt: string): Promise<string | null> {
    const match = await this.getDetailed(prompt);
    return match ? match.response : null;
  }

  /**
   * Retrieve response, similarity score, and metadata for a prompt
   */
  async getDetailed(prompt: string): Promise<{ response: string; similarity: number; metadata?: any } | null> {
    if (this.cache.length === 0 || typeof prompt !== 'string') {
      return null;
    }

    const queryVector = await this.embeddingFunction(prompt);
    let bestMatch: CacheEntry | null = null;
    let highestSimilarity = -1;

    for (const entry of this.cache) {
      const similarity = this.cosineSimilarity(queryVector, entry.vector);
      if (similarity > highestSimilarity) {
        highestSimilarity = similarity;
        bestMatch = entry;
      }
    }

    if (bestMatch && highestSimilarity >= this.minSimilarity) {
      return {
        response: bestMatch.response,
        similarity: highestSimilarity,
        metadata: bestMatch.metadata,
      };
    }

    return null;
  }

  /**
   * Reset vector cache pool and clear local storage file
   */
  clear(): void {
    this.cache = [];
    if (this.filePath && fs.existsSync(this.filePath)) {
      try {
        fs.unlinkSync(this.filePath);
      } catch (err) {
        // Safe check for locked files
      }
    }
  }

  /**
   * Returns all current cache entries
   */
  getAll(): CacheEntry[] {
    return [...this.cache];
  }

  /**
   * Calculate cosine similarity between two unit-normalized vectors.
   * Since vectors generated are already L2 normalized to unit length,
   * the cosine similarity is simply the dot product!
   */
  private cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (vecA.length !== vecB.length || vecA.length === 0) {
      return 0;
    }
    let dotProduct = 0;
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
    }
    return dotProduct;
  }

  /**
   * Read cache file from local disk
   */
  private load(): void {
    if (!this.filePath || !fs.existsSync(this.filePath)) {
      return;
    }

    try {
      const rawData = fs.readFileSync(this.filePath, 'utf8');
      const parsed = JSON.parse(rawData);
      if (Array.isArray(parsed)) {
        this.cache = parsed;
      }
    } catch (err) {
      // Gracefully handle corrupted files by starting with an empty cache
      this.cache = [];
    }
  }

  /**
   * Save cache file to local disk
   */
  private save(): void {
    if (!this.filePath) {
      return;
    }

    try {
      const dir = path.dirname(this.filePath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.writeFileSync(this.filePath, JSON.stringify(this.cache, null, 2), 'utf8');
    } catch (err) {
      // Fail silently to prevent cache saving failures from breaking business apps
    }
  }
}

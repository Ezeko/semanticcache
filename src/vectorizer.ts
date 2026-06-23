/**
 * Simple, zero-dependency, state-free text vectorizer using the Hashing Trick (Feature Hashing).
 * Maps arbitrary strings to a fixed-size vector space to calculate cosine similarity.
 */

/**
 * Fast string hash function
 */
function hashString(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = (hash * 33) ^ str.charCodeAt(i);
  }
  return Math.abs(hash);
}

/**
 * Tokenize string into alphanumeric lowercase terms
 */
export function tokenize(text: string): string[] {
  if (typeof text !== 'string') return [];
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .split(/\s+/)
    .filter((token) => token.length > 0);
}

/**
 * Generate a normalized vector from input text
 * @param text - The input text
 * @param dimensions - Target vector dimension size
 * @returns L2 normalized float array
 */
export function embed(text: string, dimensions: number = 128): number[] {
  const tokens = tokenize(text);
  const vector = new Array<number>(dimensions).fill(0);

  if (tokens.length === 0) {
    return vector;
  }

  // Accumulate term frequencies in hashed slots
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const index = hashString(token) % dimensions;
    vector[index] += 1.0;

    // Accumulate word bigrams to maintain context order
    if (i < tokens.length - 1) {
      const bigram = `${token}_${tokens[i + 1]}`;
      const bigramIndex = hashString(bigram) % dimensions;
      vector[bigramIndex] += 0.5;
    }
  }

  // L2 Norm normalization (converts vector to unit length)
  let sumSquare = 0;
  for (let i = 0; i < dimensions; i++) {
    sumSquare += vector[i] * vector[i];
  }

  const norm = Math.sqrt(sumSquare);
  if (norm > 0) {
    for (let i = 0; i < dimensions; i++) {
      vector[i] /= norm;
    }
  }

  return vector;
}

import { SemanticCache } from './src/index';

const green = (t: string) => `\x1b[32m${t}\x1b[0m`;
const red = (t: string) => `\x1b[31m${t}\x1b[0m`;

async function run() {
  const cache = new SemanticCache({
    minSimilarity: 0.60,
    dimensions: 128,
  });

  console.log('--- Seeding SemanticCache ---');
  await cache.set('What is the capital of Nigeria?', 'Abuja');
  await cache.set('Who wrote Hamlet?', 'William Shakespeare');

  console.log('\n--- Running Cache Queries ---');

  const query1 = 'Tell me the capital of Nigeria';
  console.log(`Query 1: "${query1}"`);
  const match1 = await cache.getDetailed(query1);
  if (match1) {
    console.log(`Result:  ${green('HIT')} ("${match1.response}", Similarity: ${(match1.similarity * 100).toFixed(1)}%)`);
  } else {
    console.log(`Result:  ${red('MISS')}`);
  }

  const query2 = 'Who is the author of Hamlet?';
  console.log(`\nQuery 2: "${query2}"`);
  const match2 = await cache.getDetailed(query2);
  if (match2) {
    console.log(`Result:  ${green('HIT')} ("${match2.response}", Similarity: ${(match2.similarity * 100).toFixed(1)}%)`);
  } else {
    console.log(`Result:  ${red('MISS')} (Expected for local vectorizer: requires neural embeddings like OpenAI/Gemini for synonym matching)`);
  }

  const query3 = 'What is the temperature in Abuja?';
  console.log(`\nQuery 3: "${query3}"`);
  const match3 = await cache.getDetailed(query3);
  if (match3) {
    console.log(`Result:  ${green('HIT')} ("${match3.response}", Similarity: ${(match3.similarity * 100).toFixed(1)}%)`);
  } else {
    console.log(`Result:  ${red('MISS')} (Correct: Different semantic intent)`);
  }
}

run();

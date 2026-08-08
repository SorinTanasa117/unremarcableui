const MAX_TOKENS = 262_144;
const CHARS_PER_TOKEN = 4;

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

export function getTokenPercentage(count: number): number {
  return Math.min((count / MAX_TOKENS) * 100, 100);
}

export function formatTokenCount(count: number): string {
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}k`;
  return String(count);
}

export { MAX_TOKENS };

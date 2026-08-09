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

export interface TokenRateSample {
  timestamp: number;
  input: number;
  output: number;
}

/**
 * Tracks prompt-prefill and generation speed from cumulative `tokens` SSE
 * events. Prompt speed uses run start through first output, so it can update
 * while prompt execution is still in progress. Generation speed uses output
 * deltas after first output.
 *
 * Ring keeps recent output samples bounded while preserving short-turn
 * measurements.
 */
export class TokenRateTracker {
  private samples: TokenRateSample[] = [];
  private startedAt = 0;
  private promptTokens = 0;
  private generationStartedAt = 0;
  private generationStartOutput = 0;

  constructor(private readonly maxSamples: number = 8) {}

  start(timestamp: number = Date.now()) {
    this.reset();
    this.startedAt = timestamp;
  }

  /**
   * Record a new reading. Drops the oldest sample when the ring is full so
   * the reported rate always reflects the last `maxSamples` SSE `tokens`
   * events regardless of how long the run took. Returns the freshly
   * computed rates so callers can update display without re-deriving them.
   */
  record(timestamp: number, input: number, output: number): { read: number; write: number; active: boolean } {
    if (!this.startedAt) this.startedAt = timestamp;
    this.promptTokens = Math.max(this.promptTokens, input);

    if (output > 0 && !this.generationStartedAt) {
      this.generationStartedAt = timestamp;
      this.generationStartOutput = output;
    }

    const last = this.samples[this.samples.length - 1];
    // Update timestamp of an unchanged count rather than creating a duplicate
    // sample — keeps the ring strictly monotone in (input, output) so a single
    // event cannot temporarily skew the rate to zero.
    if (last && last.input === input && last.output === output) {
      last.timestamp = timestamp;
      return this.rates();
    }
    this.samples.push({ timestamp, input, output });
    while (this.samples.length > this.maxSamples) this.samples.shift();
    return this.rates();
  }

  reset() {
    this.samples.length = 0;
    this.startedAt = 0;
    this.promptTokens = 0;
    this.generationStartedAt = 0;
    this.generationStartOutput = 0;
  }

  /**
   * Compute rates. Read remains live during prompt execution; write starts
   * after first output so prompt latency does not dilute generation speed.
   */
  rates(now: number = Date.now()): { read: number; write: number; active: boolean } {
    const last = this.samples[this.samples.length - 1];
    const readElapsed = (this.generationStartedAt || now) - this.startedAt;
    const read = this.promptTokens > 0 && readElapsed > 0
      ? this.promptTokens / (readElapsed / 1000)
      : 0;
    const writeElapsed = this.generationStartedAt && last
      ? last.timestamp - this.generationStartedAt
      : 0;
    const write = writeElapsed > 0 && last
      ? Math.max(0, (last.output - this.generationStartOutput) / (writeElapsed / 1000))
      : 0;
    return { read, write, active: read + write > 0 };
  }
}

/** Format a tokens-per-second value for compact display in the top bar. */
export function formatTokenRate(tokensPerSecond: number): string {
  if (!Number.isFinite(tokensPerSecond) || tokensPerSecond <= 0) return '0.0';
  if (tokensPerSecond >= 100) return `${Math.round(tokensPerSecond)}`;
  if (tokensPerSecond >= 10) return tokensPerSecond.toFixed(1);
  return tokensPerSecond.toFixed(2);
}

export { MAX_TOKENS };

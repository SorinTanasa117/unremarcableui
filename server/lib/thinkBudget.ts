/**
 * Pure decision helpers for the <think> reasoning budget.
 *
 * Extracted from server/routes/ollama.ts so the enforcement rules can be
 * tested without a live Ollama runtime.
 */

export interface ThinkBudgetTurnOptions {
  /** THINK_BUDGET_INITIAL env value (0 disables enforcement entirely). */
  initialBudget: number;
  /** THINK_BUDGET_FOLLOWUP_RATIO env value. */
  followupRatio: number;
  /** Thinking active for this run (user toggle AND not forced off by an earlier breach). */
  thinkEnabled: boolean;
  /** Runtime reports thinking support for this backend. */
  supportsThinking: boolean;
  isFirstTurn: boolean;
  isErrorTurn: boolean;
}

/**
 * Token budget a single turn may spend inside <think>. 0 = unenforced.
 * First turn and error-recovery turns get the full initial budget; follow-up
 * execution turns get followupRatio of it, floored at 100 tokens.
 */
export function computeThinkBudgetForTurn(opts: ThinkBudgetTurnOptions): number {
  if (opts.initialBudget <= 0 || !opts.thinkEnabled) return 0;
  if (!opts.supportsThinking) return 0;
  if (opts.isFirstTurn || opts.isErrorTurn) return opts.initialBudget;
  const clampedRatio = Math.min(Math.max(opts.followupRatio, 0.05), 1.0);
  return Math.max(100, Math.round(opts.initialBudget * clampedRatio));
}

export interface ThinkBreachRecoveryCheck {
  backendOllama: boolean;
  /** Stream was cut because hidden reasoning crossed the budget. */
  budgetExceededThisTurn: boolean;
  /** Any visible assistant text arrived before the cut. */
  hasVisibleContent: boolean;
  /** Any native or fallback-parsed tool calls arrived before the cut. */
  hasPendingToolCalls: boolean;
  /** Budget-interrupt recoveries already used in this run. */
  interruptionCount: number;
  /** Max recoveries allowed (shares MAX_EMPTY_TURN_RETRIES bound). */
  maxRetries: number;
  aborted: boolean;
}

/**
 * True when a budget-breach turn produced nothing usable and gets one retry
 * with thinking disabled and tools still available. False falls through to
 * the empty-turn ladder so a persistently mute model is still stopped.
 */
export function shouldRecoverAfterThinkBreach(c: ThinkBreachRecoveryCheck): boolean {
  return (
    c.backendOllama
    && c.budgetExceededThisTurn
    && !c.hasVisibleContent
    && !c.hasPendingToolCalls
    && c.interruptionCount < c.maxRetries
    && !c.aborted
  );
}

/**
 * Focused regression test for think-budget enforcement helpers.
 *
 * Run:  npx tsx server/scripts/testThinkBudget.ts
 *
 * Covers:
 *  - computeThinkBudgetForTurn (initial / follow-up / error / disabled paths)
 *  - shouldRecoverAfterThinkBreach (cut-thought retry vs fall-through)
 */
import {
  computeThinkBudgetForTurn,
  shouldRecoverAfterThinkBreach,
} from '../lib/thinkBudget.js';

let failures = 0;
function check(label: string, cond: boolean) {
  if (cond) {
    console.log(`  ok  - ${label}`);
  } else {
    failures += 1;
    console.error(`  FAIL- ${label}`);
  }
}

const base = {
  initialBudget: 1000,
  followupRatio: 0.3,
  thinkEnabled: true,
  supportsThinking: true,
  isFirstTurn: false,
  isErrorTurn: false,
};

// computeThinkBudgetForTurn
check('first turn gets full budget', computeThinkBudgetForTurn({ ...base, isFirstTurn: true }) === 1000);
check('error turn gets full budget', computeThinkBudgetForTurn({ ...base, isErrorTurn: true }) === 1000);
check('follow-up turn gets 30% floored at 100', computeThinkBudgetForTurn(base) === 300);
check(
  'follow-up floor applies for tiny budgets',
  computeThinkBudgetForTurn({ ...base, initialBudget: 150 }) === Math.max(100, Math.round(150 * 0.3)),
);
check('think disabled yields 0', computeThinkBudgetForTurn({ ...base, thinkEnabled: false }) === 0);
check('initialBudget=0 disables entirely', computeThinkBudgetForTurn({ ...base, initialBudget: 0 }) === 0);
check('runtime without thinking support yields 0', computeThinkBudgetForTurn({ ...base, supportsThinking: false }) === 0);

// Breached-run behavior is expressed by passing thinkEnabled=false next turn.
check(
  'breach forces no-think next turn (budget 0)',
  computeThinkBudgetForTurn({ ...base, thinkEnabled: false }) === 0,
);

// shouldRecoverAfterThinkBreach
const baseBreach = {
  backendOllama: true,
  budgetExceededThisTurn: true,
  hasVisibleContent: false,
  hasPendingToolCalls: false,
  interruptionCount: 0,
  maxRetries: 3,
  aborted: false,
};
check('pure cut-thought turn recovers', shouldRecoverAfterThinkBreach(baseBreach) === true);
check('visible content present does not recover', shouldRecoverAfterThinkBreach({ ...baseBreach, hasVisibleContent: true }) === false);
check('pending tool calls do not recover', shouldRecoverAfterThinkBreach({ ...baseBreach, hasPendingToolCalls: true }) === false);
check('retries exhausted falls through', shouldRecoverAfterThinkBreach({ ...baseBreach, interruptionCount: 3 }) === false);
check('non-ollama backend never recovers here', shouldRecoverAfterThinkBreach({ ...baseBreach, backendOllama: false }) === false);
check('aborted run never recovers', shouldRecoverAfterThinkBreach({ ...baseBreach, aborted: true }) === false);
check('no breach never recovers', shouldRecoverAfterThinkBreach({ ...baseBreach, budgetExceededThisTurn: false }) === false);

if (failures > 0) {
  console.error(`\n${failures} check(s) failed.`);
  process.exit(1);
}
console.log('\nAll think-budget checks passed.');

import { Router, Request, Response } from 'express';
import { spawn } from 'child_process';
import fsNative from 'fs';
import fs from 'fs/promises';
import path from 'path';
import { ContextManager, Message, ToolCall, estimateMessageTokens, compactToolCall } from '../lib/contextManager.js';
import { computeThinkBudgetForTurn, shouldRecoverAfterThinkBreach } from '../lib/thinkBudget.js';
import { runTool, ToolName, clearSessionToolState } from '../lib/toolRunner.js';
import { TOOL_DEFINITIONS, WORKSPACE_DIR } from '../lib/tools.js';
import * as novelStorage from '../lib/novel/storage.js';
import { inferenceDispatcher } from '../lib/inferenceDispatcher.js';
import {
  ingestAttachments,
  readImageAttachments,
  sessionAttachmentsDir,
  contentTypeForFile,
  type IncomingAttachment,
} from '../lib/attachments.js';

const router = Router();
const OLLAMA_URL = process.env.OLLAMA_URL ?? 'http://127.0.0.1:11434';
const LLAMACPP_URL = process.env.LLAMACPP_URL ?? 'http://127.0.0.1:8080';
// No session-token cap by default. The env override OLLAMA_NUM_CTX lets
// deployments tune this; Ollama itself enforces the model's own ceiling.
const OLLAMA_NUM_CTX = Number.parseInt(process.env.OLLAMA_NUM_CTX ?? '1048576', 10);
const LLAMACPP_NUM_CTX = Number.parseInt(process.env.LLAMACPP_NUM_CTX ?? '1048576', 10);
// Isolated story packets are capped well below this. A smaller KV cache avoids
// reserving a 262k-token window for every prose continuation.
const STORY_NUM_CTX = Number.parseInt(process.env.STORY_NUM_CTX ?? '16384', 10);
const OLLAMA_KEEP_ALIVE = process.env.OLLAMA_KEEP_ALIVE ?? '15m';
// A single write_file call must fit the whole file (filepath + content) plus
// any hidden reasoning tokens in one response. 1536 truncated mid-content on
// files larger than ~1500 tokens, dropping the trailing filepath argument and
// silently losing the file. 8192 comfortably covers typical single-file writes.
const parsedNumPredict = Number.parseInt(process.env.OLLAMA_NUM_PREDICT ?? '8192', 10);
const OLLAMA_NUM_PREDICT = Number.isFinite(parsedNumPredict) && parsedNumPredict > 0 ? parsedNumPredict : 8192;
// Thinking tokens are billed against num_predict, so a think=true turn can burn
// its whole budget inside <think> and be cut off (done_reason:'length') before
// emitting any visible content or a tool call. Reasoning turns get this extra
// ceiling on top of OLLAMA_NUM_PREDICT.
const parsedThinkOverhead = Number.parseInt(process.env.OLLAMA_THINKING_NUM_PREDICT_OVERHEAD ?? '16384', 10);
const THINKING_NUM_PREDICT_OVERHEAD = Number.isFinite(parsedThinkOverhead) && parsedThinkOverhead >= 0 ? parsedThinkOverhead : 16384;
// ── Thinking budget control ─────────────────────────────────────────────────
// Maximum tokens the model may spend inside <think> on the FIRST response turn.
// Subsequent (follow-up) execution turns are capped at THINK_BUDGET_FOLLOWUP_RATIO (30%)
// of this budget. A tool error restores the full initial budget for diagnostic reasoning.
// Set to 0 to disable thinking budget enforcement entirely.
const parsedThinkBudgetInitial = Number.parseInt(process.env.THINK_BUDGET_INITIAL ?? '1000', 10);
const THINK_BUDGET_INITIAL = Number.isFinite(parsedThinkBudgetInitial) && parsedThinkBudgetInitial >= 0
  ? parsedThinkBudgetInitial
  : 1000;
const parsedFollowupRatio = Number.parseFloat(process.env.THINK_BUDGET_FOLLOWUP_RATIO ?? '0.30');
const THINK_BUDGET_FOLLOWUP_RATIO = Number.isFinite(parsedFollowupRatio)
  ? Math.min(Math.max(parsedFollowupRatio, 0.05), 1.0)
  : 0.30;
const parsedTokensToChars = Number.parseFloat(process.env.THINK_TOKENS_TO_CHARS ?? '3.5');
const THINK_TOKENS_TO_CHARS = Number.isFinite(parsedTokensToChars) && parsedTokensToChars > 0
  ? parsedTokensToChars
  : 3.5;
const parsedNovelDraftNumPredict = Number.parseInt(process.env.NOVEL_DRAFT_NUM_PREDICT ?? '8192', 10);
const NOVEL_DRAFT_NUM_PREDICT = Number.isFinite(parsedNovelDraftNumPredict) && parsedNovelDraftNumPredict > 0
  ? parsedNovelDraftNumPredict
  : 8192;
const NOVEL_DRAFT_MAX_CONTINUATIONS = 3;
// No char limit on tool output fed back into context. Truncating silently
// dropped chunks of large file reads / build logs and broke long builds.
// Set TOOL_CONTEXT_CHAR_LIMIT only if you have a specific reason.
const parsedToolContextCharLimit = Number.parseInt(process.env.TOOL_CONTEXT_CHAR_LIMIT ?? '0', 10);
const TOOL_CONTEXT_CHAR_LIMIT = Number.isFinite(parsedToolContextCharLimit) && parsedToolContextCharLimit > 0 ? parsedToolContextCharLimit : Infinity;
const CONTEXT_SIZES = new Set([16_384, 32_768, 65_536, 131_072, 262_144]);
// Tool calls are unlimited — the agent may iterate as long as it keeps making
// progress. The guardrail is CONSECUTIVE failures: 3 in a row pauses for human
// guidance (ask_user), 5 in a row kills the run for human review. Any success
// resets the consecutive-failure counter.
const MAX_CONSECUTIVE_FAILURES = 5;
// Bounded retries for thinking-only / empty turns before the run is ended.
const MAX_EMPTY_TURN_RETRIES = 3;
// End-of-run code review: one bounded review round per run. The reviewer
// sidecar reads the files the agent changed and flags top issues; the coder
// gets one chance to fix them before the run completes.
const MAX_REVIEW_ROUNDS = 1;
const REVIEW_AT_END = process.env.REVIEW_AT_END !== 'false';

// ── Run watchdog (local runtimes) ──────────────────────────────────────────
// Kills runs that are provably dead instead of letting them hold VRAM forever,
// while NEVER cutting a model that is genuinely working slowly. Evidence model:
//   • any streamed token/thinking chunk  → decoding, alive
//   • llama.cpp /slots n_past advancing  → prompt-eval/decode progressing
//   • CPU burned by ollama*/llama-server → compute happening (prefill, paging)
// Termination rules (all env-tunable):
//   • whole run exceeds RUN_MAX_DURATION_MS          → cut (default 6h)
//   • ONE inference turn exceeds TURN_MAX_DURATION_MS → cut (default 1h)
//   • ONE tool call exceeds TOOL_MAX_DURATION_MS      → failed result (default 1h)
//   • response streaming but TWO consecutive probes find no tokens AND no
//     compute → proven hang, cut instantly. Inconclusive probes resolve to
//     "alive" — never kill blind.
const WATCHDOG_ENABLED = process.env.RUN_WATCHDOG !== 'false';
const parsedRunMaxMs = Number.parseInt(process.env.RUN_MAX_DURATION_MS ?? '21600000', 10);
const RUN_MAX_DURATION_MS = Number.isFinite(parsedRunMaxMs) && parsedRunMaxMs > 0 ? parsedRunMaxMs : 21_600_000;
const parsedTurnMaxMs = Number.parseInt(process.env.TURN_MAX_DURATION_MS ?? '3600000', 10);
const TURN_MAX_DURATION_MS = Number.isFinite(parsedTurnMaxMs) && parsedTurnMaxMs > 0 ? parsedTurnMaxMs : 3_600_000;
const parsedToolMaxMs = Number.parseInt(process.env.TOOL_MAX_DURATION_MS ?? '3600000', 10);
const TOOL_MAX_DURATION_MS = Number.isFinite(parsedToolMaxMs) && parsedToolMaxMs > 0 ? parsedToolMaxMs : 3_600_000;
// Long prefills on slow backends (iGPU/Vulkan, ~9 tok/s over an 8k+ prompt)
// stream zero tokens for many minutes while healthy. 120s false-killed those
// runs; 30 minutes of proven silence is the kill threshold instead.
const parsedStallQuietMs = Number.parseInt(process.env.STALL_QUIET_MS ?? '1800000', 10);
const STALL_QUIET_MS = Number.isFinite(parsedStallQuietMs) && parsedStallQuietMs > 0 ? parsedStallQuietMs : 1_800_000;
const parsedProbeIntervalMs = Number.parseInt(process.env.WATCHDOG_PROBE_INTERVAL_MS ?? '15000', 10);
const WATCHDOG_PROBE_INTERVAL_MS = Number.isFinite(parsedProbeIntervalMs) && parsedProbeIntervalMs > 0 ? parsedProbeIntervalMs : 15_000;
const parsedProbeGapMs = Number.parseInt(process.env.WATCHDOG_PROBE_SAMPLE_GAP_MS ?? '3000', 10);
const WATCHDOG_PROBE_SAMPLE_GAP_MS = Number.isFinite(parsedProbeGapMs) && parsedProbeGapMs > 0 ? parsedProbeGapMs : 3_000;
const parsedMinCpu = Number.parseFloat(process.env.WATCHDOG_MIN_CPU_TICKS_PER_SEC ?? '500000');
// Cumulative kernel+user CPU ticks (100ns units) burned per second by the
// runtime's processes. 500k ticks/s = 50ms of CPU per second ≈ 5% of one core.
// Below this during silence, the runtime is not working.
const WATCHDOG_MIN_CPU_TICKS_PER_SEC = Number.isFinite(parsedMinCpu) && parsedMinCpu >= 0 ? parsedMinCpu : 500_000;
const parsedIdleProbes = Number.parseInt(process.env.WATCHDOG_IDLE_PROBES_TO_KILL ?? '20', 10);
// 20 probes × 15s interval ≈ 5 minutes of consecutive conclusive-zero
// evidence before a hang verdict. A single bad CPU sample (iGPU offload,
// counter hiccup) can no longer kill a healthy run.
const WATCHDOG_IDLE_PROBES_TO_KILL = Number.isFinite(parsedIdleProbes) && parsedIdleProbes > 0 ? parsedIdleProbes : 20;
// Ollama-only fast liveness: poll the loaded-model list (`ollama ps` == GET
// /api/ps). On a 32 GB shared-memory laptop, memory pressure makes Ollama
// unload the model mid-run; the open request then hangs forever with zero
// tokens. An empty model list is unambiguous proof the runtime dropped the
// model, so we stop the run instead of waiting out the coarse quiet/turn
// budgets. A resident-but-slow prefill still lists the model, so this never
// false-kills a healthy long generation.
const MODEL_EVICTION_WATCH = process.env.MODEL_EVICTION_WATCH !== 'false';
const parsedPsIntervalMs = Number.parseInt(process.env.MODEL_PS_INTERVAL_MS ?? '60000', 10);
const MODEL_PS_INTERVAL_MS = Number.isFinite(parsedPsIntervalMs) && parsedPsIntervalMs > 0 ? parsedPsIntervalMs : 60_000;
const parsedGoneToKill = Number.parseInt(process.env.MODEL_GONE_CHECKS_TO_KILL ?? '2', 10);
// Two consecutive empty polls (≈2 min at the default 60s cadence) before the
// verdict, so a brief gap during a legitimate one-shot model swap can't trip it.
const MODEL_GONE_CHECKS_TO_KILL = Number.isFinite(parsedGoneToKill) && parsedGoneToKill > 0 ? parsedGoneToKill : 2;

const SESSIONS_DIR = path.resolve(WORKSPACE_DIR, '.sessions');
const OLLAMA_FAILURE_LOG_PATH = path.resolve(WORKSPACE_DIR, '.logs', 'ollama-failures.jsonl');
const MAX_OLLAMA_FAILURE_LOG_BYTES = 512 * 1024;
const OLLAMA_MODELS_DIR = process.env.OLLAMA_MODELS_DIR
  ? path.resolve(process.env.OLLAMA_MODELS_DIR)
  : path.resolve(process.env.USERPROFILE ?? '', '.ollama', 'models');
const OLLAMA_MANIFESTS_DIR = path.join(OLLAMA_MODELS_DIR, 'manifests', 'registry.ollama.ai');
const OLLAMA_BLOBS_DIR = path.join(OLLAMA_MODELS_DIR, 'blobs');
const LLAMACPP_PID_FILE = path.resolve(WORKSPACE_DIR, '.llamacpp-server.pid');
const LLAMACPP_START_LOG_PATH = path.resolve(WORKSPACE_DIR, '.logs', 'llamacpp-start.log');
const LLAMACPP_EXE_PATH = process.env.LLAMACPP_EXE
  ? path.resolve(process.env.LLAMACPP_EXE)
  : path.resolve(WORKSPACE_DIR, 'llama.cpp', 'build', 'bin', 'llama-server.exe');
const VSDEVCMD_PATH = process.env.VSDEVCMD_PATH
  ?? 'C:\\Program Files (x86)\\Microsoft Visual Studio\\18\\BuildTools\\Common7\\Tools\\VsDevCmd.bat';
const ONEAPI_SETVARS_PATH = process.env.ONEAPI_SETVARS_PATH
  ?? 'C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat';
const LLAMACPP_NGL = Number.parseInt(process.env.LLAMACPP_NGL ?? '24', 10);
const LLAMACPP_FA = (process.env.LLAMACPP_FA ?? 'on').toLowerCase() === 'off' ? 'off' : 'on';
const AUTO_START_LLAMACPP = process.env.AUTO_START_LLAMACPP !== 'false';
const LLAMACPP_START_TIMEOUT_MS = Number.parseInt(process.env.LLAMACPP_START_TIMEOUT_MS ?? '180000', 10);
let ollamaFailureLogQueue = Promise.resolve();

interface ModelCapabilities {
  supportsTools: boolean;
  supportsThinking: boolean;
  supportsVision: boolean;
  contextLength?: number;
  systemPromptType?: string;
}
type InferenceBackend = 'ollama' | 'llamacpp';
const modelCapabilitiesCache = new Map<string, ModelCapabilities>();

function resolveInferenceBackend(value: unknown): InferenceBackend {
  return value === 'llamacpp' ? 'llamacpp' : 'ollama';
}

function getBackendUrl(backend: InferenceBackend): string {
  return backend === 'llamacpp' ? LLAMACPP_URL : OLLAMA_URL;
}

const parsedLlamaCppUrl = (() => {
  try {
    return new URL(LLAMACPP_URL);
  } catch {
    return null;
  }
})();
const LLAMACPP_HOST = process.env.LLAMACPP_HOST ?? parsedLlamaCppUrl?.hostname ?? '127.0.0.1';
const LLAMACPP_PORT = Number.parseInt(
  process.env.LLAMACPP_PORT ?? parsedLlamaCppUrl?.port ?? '8080',
  10,
);
let llamaCppStartPromise: Promise<void> | null = null;
let ollamaDigestNameMapPromise: Promise<Map<string, string>> | null = null;
const unsupportedLlamaCppArchByDigest = new Map<string, string>();

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// ── Runtime health probes (watchdog evidence) ──────────────────────────────

interface CpuSample {
  seconds: number; // cumulative process CPU time (user+kernel), monotonic
}

/** Sample cumulative CPU-seconds of the runtime's processes via Get-Process
 *  (.CPU = TotalProcessorTime in seconds, monotonic). Returns null when no
 *  processes exist or the read fails — an inconclusive probe must never count
 *  as evidence of a hang. Unlike '% Processor Time' (an instantaneous rate),
 *  this is cumulative, so two samples a known interval apart yield true
 *  work-done-per-second. Emitted as integer milliseconds to stay independent
 *  of the shell's decimal-separator culture. */
async function sampleRuntimeCpuSeconds(backend: InferenceBackend): Promise<CpuSample | null> {
  // llama.cpp on SYCL still names its executable llama-server; Ollama runners
  // surface as ollama*. Include the generic llama* pattern for custom builds.
  const pattern = backend === 'llamacpp' ? 'llama*' : 'ollama*';
  const script = `$p = Get-Process -Name '${pattern}' -ErrorAction SilentlyContinue; `
    + `if (-not $p) { 'NONE' } else { [int64]((($p | Measure-Object -Property CPU -Sum).Sum) * 1000) }`;
  try {
    const result = await runCommandCapture('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script]);
    const out = result.stdout.trim();
    if (result.code !== 0 || !out || out === 'NONE') return null;
    const ms = Number.parseInt(out, 10);
    if (!Number.isFinite(ms)) return null;
    return { seconds: ms / 1000 };
  } catch {
    return null;
  }
}

/** Read llama.cpp slot progress. n_past advances while prompt-eval/decode runs,
 *  giving exact liveness evidence with no CPU sampling required. */
async function probeLlamaCppSlotProgress(): Promise<number | null> {
  try {
    const response = await fetch(`${LLAMACPP_URL}/slots`, {
      signal: AbortSignal.timeout(3_000),
      dispatcher: inferenceDispatcher,
    });
    if (!response.ok) return null;
    const payload = await response.json() as any;
    const slots = Array.isArray(payload) ? payload : [];
    let nPast = 0;
    for (const slot of slots) {
      const value = Number(slot?.prompt_progress?.n_past);
      if (Number.isFinite(value)) nPast += value;
    }
    return slots.length > 0 ? nPast : null;
  } catch {
    return null;
  }
}

/**
 * Decide whether a quiet runtime is actually computing. Self-contained: for the
 * CPU signal it takes two cumulative samples WATCHDOG_PROBE_SAMPLE_GAP_MS apart
 * and measures real work-done-per-second (no reliance on cross-tick state).
 * llama.cpp additionally uses slot n_past progress when available. Only a
 * successful measurement is `conclusive`; anything else must not count toward a
 * hang verdict.
 */
async function probeRuntimeComputing(
  backend: InferenceBackend,
  previousNPast: number | null,
): Promise<{ computing: boolean; conclusive: boolean; nPast: number | null }> {
  // WATCHDOG_MIN_CPU_TICKS_PER_SEC stays in legacy 100ns-tick units for
  // back-compat: 500000 ticks/s == 0.05 of one core.
  const minCoreFraction = WATCHDOG_MIN_CPU_TICKS_PER_SEC / 10_000_000;

  if (backend === 'llamacpp') {
    const nPast = await probeLlamaCppSlotProgress();
    if (nPast !== null && previousNPast !== null) {
      return { computing: nPast > previousNPast, conclusive: true, nPast };
    }
    // Slot data unavailable — fall back to the CPU measure so a missing /slots
    // endpoint can neither mask real compute nor fabricate one.
    const cpu = await measureCpuCoreFraction(backend);
    if (cpu === null) return { computing: false, conclusive: false, nPast };
    return { computing: cpu >= minCoreFraction, conclusive: true, nPast };
  }

  const cpu = await measureCpuCoreFraction(backend);
  if (cpu === null) return { computing: false, conclusive: false, nPast: null };
  return { computing: cpu >= minCoreFraction, conclusive: true, nPast: null };
}

/** Two cumulative CPU-seconds samples WATCHDOG_PROBE_SAMPLE_GAP_MS apart →
 *  average core utilization over that gap, or null if inconclusive. */
async function measureCpuCoreFraction(backend: InferenceBackend): Promise<number | null> {
  const first = await sampleRuntimeCpuSeconds(backend);
  if (first === null) return null;
  await sleep(WATCHDOG_PROBE_SAMPLE_GAP_MS);
  const second = await sampleRuntimeCpuSeconds(backend);
  if (second === null) return null;
  const gapSeconds = WATCHDOG_PROBE_SAMPLE_GAP_MS / 1000;
  if (gapSeconds <= 0) return null;
  // Clamp tiny negative jitter (counter reset / process restart) to 0.
  return Math.max(0, second.seconds - first.seconds) / gapSeconds;
}

/** Names of the models Ollama currently has resident — the HTTP equivalent of
 *  `ollama ps` (GET /api/ps). Returns null on an inconclusive read (endpoint
 *  down / bad JSON) so a transient failure is never mistaken for an eviction;
 *  an empty array means the runtime has unloaded every model. */
async function listLoadedOllamaModels(): Promise<string[] | null> {
  try {
    const resp = await fetch(`${OLLAMA_URL}/api/ps`, {
      signal: AbortSignal.timeout(3_000),
      dispatcher: inferenceDispatcher,
    });
    if (!resp.ok) return null;
    const data = await resp.json() as any;
    if (!Array.isArray(data?.models)) return null;
    return data.models
      .map((m: any) => String(m?.name ?? m?.model ?? ''))
      .filter((name: string) => name.length > 0);
  } catch {
    return null;
  }
}


async function runCommandCapture(command: string, args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = spawn(command, args, { windowsHide: true });
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.on('error', (error) => {
      resolve({ code: 1, stdout, stderr: error.message });
    });
    child.on('close', (code) => {
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}

// Compact text digest of a session for the summarizer sidecar. Keeps user
// goals, assistant prose (trimmed), and file operations; excludes the most
// recent two user turns (those stay in the window as raw context) and large
// file bodies (disk is source of truth). Capped so the summarizer stays fast.
function buildSessionDigest(messages: Message[]): string {
  const userIndexes = messages
    .map((m, i) => (m.role === 'user' ? i : -1))
    .filter((i) => i >= 0);
  const cutoff = userIndexes.length >= 2 ? userIndexes[userIndexes.length - 2] : messages.length;
  const older = messages.slice(0, cutoff);

  const lines: string[] = [];
  for (const m of older) {
    if (m.role === 'user') {
      lines.push(`USER: ${(m.content ?? '').slice(0, 800)}`);
    } else if (m.role === 'assistant') {
      const text = (m.content ?? '').trim();
      if (text) lines.push(`ASSISTANT: ${text.slice(0, 600)}`);
      for (const tc of m.tool_calls ?? []) {
        const name = tc.function?.name ?? 'tool';
        let filepath = '';
        try {
          const raw = tc.function?.arguments as unknown;
          const args = (typeof raw === 'string' ? JSON.parse(raw) : raw) as Record<string, unknown>;
          if (typeof args?.filepath === 'string') filepath = args.filepath;
          else if (typeof args?.command === 'string') filepath = String(args.command).slice(0, 120);
          else if (typeof args?.url === 'string') filepath = args.url;
          else if (typeof args?.query === 'string') filepath = String(args.query).slice(0, 120);
        } catch {}
        lines.push(`TOOL_CALL ${name}: ${filepath}`);
      }
    } else if (m.role === 'tool') {
      const out = (m.content ?? '').trim();
      if (out && !out.startsWith('[omitted')) lines.push(`TOOL_RESULT (${m.tool_name ?? ''}): ${out.slice(0, 200)}`);
    }
  }
  const digest = lines.join('\n');
  // Keep the summarizer input bounded regardless of history size.
  return digest.length > 24_000 ? digest.slice(-24_000) : digest;
}

function summarizeOllamaMessages(messages: Message[]) {
  const roleCounts: Record<string, number> = {};
  let totalContentChars = 0;
  let assistantToolCallCount = 0;
  const assistantToolCallNames: string[] = [];

  for (const message of messages) {
    roleCounts[message.role] = (roleCounts[message.role] ?? 0) + 1;
    totalContentChars += typeof message.content === 'string' ? message.content.length : 0;
    if (message.role === 'assistant' && message.tool_calls?.length) {
      assistantToolCallCount += message.tool_calls.length;
      for (const toolCall of message.tool_calls) {
        const name = toolCall.function?.name;
        if (typeof name === 'string') assistantToolCallNames.push(name);
      }
    }
  }

  return {
    count: messages.length,
    roleCounts,
    totalContentChars,
    lastRoles: messages.slice(-5).map(message => message.role),
    assistantToolCallNames: assistantToolCallNames.slice(-5),
    assistantToolCallCount,
  };
}

function appendOllamaFailureLog(entry: Record<string, unknown>): Promise<void> {
  const task = ollamaFailureLogQueue.then(async () => {
    await fs.mkdir(path.dirname(OLLAMA_FAILURE_LOG_PATH), { recursive: true });
    let existing = '';
    try {
      existing = await fs.readFile(OLLAMA_FAILURE_LOG_PATH, 'utf-8');
    } catch {}

    const lines = `${existing}${JSON.stringify(entry)}\n`.split('\n').filter(Boolean);
    while (Buffer.byteLength(lines.join('\n') + '\n', 'utf-8') > MAX_OLLAMA_FAILURE_LOG_BYTES) {
      lines.shift();
    }
    await fs.writeFile(OLLAMA_FAILURE_LOG_PATH, `${lines.join('\n')}\n`, 'utf-8');
  });
  ollamaFailureLogQueue = task.catch(() => undefined);
  return task.catch(() => undefined);
}

// Lazy-loaded model map for capability fallback
let modelMapData: any = null;
async function getModelMap(): Promise<any> {
  if (modelMapData) return modelMapData;
  try {
    const filePath = path.resolve(WORKSPACE_DIR, '../model_map.json');
    const content = await fs.readFile(filePath, 'utf-8');
    modelMapData = JSON.parse(content);
    return modelMapData;
  } catch {
    return null;
  }
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

// Role → model id map from model_map.json ("summarizer", "vision", "coder").
// Used for one-shot sidecar swaps: a small model summarizes context or reads an
// image, then unloads, so the primary coder keeps the RAM on a 32 GB machine.
async function getModelRoles(): Promise<Record<string, string>> {
  const map = await getModelMap();
  const roles = map?.roles;
  return roles && typeof roles === 'object' ? roles as Record<string, string> : {};
}

// One non-streaming call to an auxiliary model. keep_alive:0 unloads it right
// after so the primary coder can reload without exceeding shared RAM. Returns
// null on any failure (model not pulled, error) so callers can fall back.
async function runOneShotModel(
  model: string,
  messages: unknown[],
  signal: AbortSignal,
  opts: { numCtx?: number; numPredict?: number } = {},
): Promise<string | null> {
  try {
    const resp = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages,
        stream: false,
        think: false,
        options: { num_ctx: opts.numCtx ?? 8_192, num_predict: opts.numPredict ?? 1_024 },
        keep_alive: 0,
      }),
      signal,
      dispatcher: inferenceDispatcher,
    } as any);
    if (!resp.ok) return null;
    const data = await resp.json() as { message?: { content?: string } };
    const content = data?.message?.content;
    return typeof content === 'string' && content.trim() ? content.trim() : null;
  } catch {
    return null;
  }
}

async function listFilesRecursive(rootDir: string): Promise<string[]> {
  const files: string[] = [];
  const pending = [rootDir];

  while (pending.length > 0) {
    const current = pending.pop()!;
    let entries;
    try {
      entries = await fs.readdir(current, { withFileTypes: true, encoding: 'utf8' });
    } catch {
      continue;
    }

    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        pending.push(fullPath);
      } else if (entry.isFile()) {
        files.push(fullPath);
      }
    }
  }

  return files;
}

function extractDigestFromModelId(modelId: string): string | null {
  const match = modelId.match(/sha256[-:]([a-f0-9]{64})/i);
  return match?.[1]?.toLowerCase() ?? null;
}

function getUnsupportedArchitectureForModelPath(modelPath: string): string | undefined {
  const digest = extractDigestFromModelId(modelPath);
  if (!digest) return undefined;
  return unsupportedLlamaCppArchByDigest.get(digest);
}

function markUnsupportedArchitectureForModelPath(modelPath: string, architecture: string): void {
  const digest = extractDigestFromModelId(modelPath);
  if (!digest) return;
  unsupportedLlamaCppArchByDigest.set(digest, architecture);
}

function manifestPathToModelName(manifestPath: string): string {
  const relative = path.relative(OLLAMA_MANIFESTS_DIR, manifestPath);
  const segments = relative.split(path.sep).filter(Boolean);
  if (segments.length === 0) return path.basename(manifestPath);
  const tag = segments.pop()!;
  const namespaceAndName = segments.join('/').replace(/^library\//, '');
  return `${namespaceAndName}:${tag}`;
}

async function getOllamaDigestNameMap(): Promise<Map<string, string>> {
  if (ollamaDigestNameMapPromise) return ollamaDigestNameMapPromise;

  ollamaDigestNameMapPromise = (async () => {
    const digestMap = new Map<string, string>();
    const manifestFiles = await listFilesRecursive(OLLAMA_MANIFESTS_DIR);

    for (const manifestFile of manifestFiles) {
      try {
        const raw = await fs.readFile(manifestFile, 'utf-8');
        const parsed = JSON.parse(raw) as {
          layers?: Array<{ mediaType?: string; digest?: string }>;
        };
        const modelLayer = parsed.layers?.find((layer) =>
          layer.mediaType === 'application/vnd.ollama.image.model' && typeof layer.digest === 'string'
        );
        if (!modelLayer?.digest) continue;
        const digest = modelLayer.digest.replace(/^sha256:/, '').toLowerCase();
        if (!digest) continue;
        digestMap.set(digest, manifestPathToModelName(manifestFile));
      } catch {}
    }

    return digestMap;
  })();

  return ollamaDigestNameMapPromise;
}

async function listLocalOllamaManifestModels(): Promise<Array<{
  name: string;
  size: number;
  modified_at: string;
}>> {
  const manifestFiles = await listFilesRecursive(OLLAMA_MANIFESTS_DIR);
  const byName = new Map<string, { name: string; size: number; modified_at: string; modifiedTs: number }>();

  for (const manifestFile of manifestFiles) {
    const modelName = manifestPathToModelName(manifestFile);
    if (!modelName || !modelName.includes(':')) continue;

    let modifiedAt = new Date(0).toISOString();
    let modifiedTs = 0;
    try {
      const stat = await fs.stat(manifestFile);
      modifiedAt = stat.mtime.toISOString();
      modifiedTs = stat.mtimeMs;
    } catch {}

    const current = byName.get(modelName);
    if (!current || modifiedTs >= current.modifiedTs) {
      byName.set(modelName, {
        name: modelName,
        size: 0,
        modified_at: modifiedAt,
        modifiedTs,
      });
    }
  }

  return Array.from(byName.values())
    .sort((a, b) => b.modifiedTs - a.modifiedTs)
    .map(({ modifiedTs, ...model }) => model);
}

async function resolveFriendlyLlamaModel(modelId: string): Promise<{ displayName: string; mapId?: string }> {
  const digest = extractDigestFromModelId(modelId);
  if (digest) {
    const digestMap = await getOllamaDigestNameMap();
    const modelName = digestMap.get(digest);
    if (modelName) {
      return { displayName: `${modelName} (SYCL)`, mapId: modelName };
    }
  }

  return { displayName: path.basename(modelId) || modelId };
}

async function resolveBlobPathFromModelTag(modelTag: string): Promise<string | null> {
  const trimmed = modelTag.trim();
  if (!trimmed) return null;

  const colonIndex = trimmed.lastIndexOf(':');
  const baseName = colonIndex >= 0 ? trimmed.slice(0, colonIndex) : trimmed;
  const tag = colonIndex >= 0 ? trimmed.slice(colonIndex + 1) : 'latest';
  const parts = baseName.split('/').filter(Boolean);
  const manifestParts = parts.length === 1 ? ['library', parts[0]] : parts;
  const manifestPath = path.join(OLLAMA_MANIFESTS_DIR, ...manifestParts, tag);
  if (!(await fileExists(manifestPath))) return null;

  try {
    const parsed = JSON.parse(await fs.readFile(manifestPath, 'utf-8')) as {
      layers?: Array<{ mediaType?: string; digest?: string }>;
    };
    const modelLayer = parsed.layers?.find((layer) =>
      layer.mediaType === 'application/vnd.ollama.image.model' && typeof layer.digest === 'string'
    );
    if (!modelLayer?.digest) return null;
    const digest = modelLayer.digest.replace(/^sha256:/, '').toLowerCase();
    const blobPath = path.join(OLLAMA_BLOBS_DIR, `sha256-${digest}`);
    return (await fileExists(blobPath)) ? blobPath : null;
  } catch {
    return null;
  }
}

async function resolveDefaultLlamaCppModelPath(): Promise<string | null> {
  const configuredPath = process.env.LLAMACPP_MODEL?.trim();
  if (configuredPath) {
    const resolved = path.resolve(configuredPath);
    if (await fileExists(resolved)) return resolved;
  }

  const modelMap = await getModelMap();
  const defaultModel = typeof modelMap?.default_model === 'string' ? modelMap.default_model : '';
  if (defaultModel) {
    const blobPath = await resolveBlobPathFromModelTag(defaultModel);
    if (blobPath) return blobPath;
  }

  const digestMap = await getOllamaDigestNameMap();
  for (const digest of digestMap.keys()) {
    const candidate = path.join(OLLAMA_BLOBS_DIR, `sha256-${digest}`);
    if (await fileExists(candidate)) return candidate;
  }

  return null;
}

async function readLlamaCppPid(): Promise<number | null> {
  try {
    const raw = (await fs.readFile(LLAMACPP_PID_FILE, 'utf-8')).trim();
    const pid = Number.parseInt(raw, 10);
    return Number.isFinite(pid) && pid > 0 ? pid : null;
  } catch {
    return null;
  }
}

function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error: any) {
    return error?.code === 'EPERM';
  }
}

async function fetchLlamaCppModelsPayload(timeoutMs = 2_500): Promise<{ status: number | null; payload?: any }> {
  try {
    const response = await fetch(`${LLAMACPP_URL}/v1/models`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) {
      return { status: response.status };
    }
    return { status: response.status, payload: await response.json() };
  } catch {
    return { status: null };
  }
}

async function getLlamaCppRuntimeInfo(): Promise<{ status: number | null; modelId: string | null; nCtx: number | null }> {
  const { status, payload } = await fetchLlamaCppModelsPayload();
  if (status !== 200 || !payload) {
    return { status, modelId: null, nCtx: null };
  }

  const firstData = Array.isArray(payload.data) ? payload.data[0] : undefined;
  const firstModel = Array.isArray(payload.models) ? payload.models[0] : undefined;

  const modelId = firstData?.id
    ? String(firstData.id)
    : firstModel?.name
      ? String(firstModel.name)
      : null;

  const rawCtx = firstData?.meta?.n_ctx ?? firstModel?.details?.n_ctx;
  const parsedCtx = typeof rawCtx === 'number'
    ? rawCtx
    : typeof rawCtx === 'string'
      ? Number.parseInt(rawCtx, 10)
      : NaN;

  return {
    status,
    modelId,
    nCtx: Number.isFinite(parsedCtx) ? parsedCtx : null,
  };
}

async function getLlamaCppModelsStatus(): Promise<number | null> {
  return (await fetchLlamaCppModelsPayload()).status;
}

async function getLlamaCppLoadedModelId(): Promise<string | null> {
  return (await getLlamaCppRuntimeInfo()).modelId;
}

function modelReferenceMatches(modelId: string | null, expectedModelPath: string): boolean {
  if (!modelId) return false;
  const expectedPath = path.resolve(expectedModelPath).toLowerCase();
  const normalizedModelId = path.resolve(modelId).toLowerCase();
  if (normalizedModelId === expectedPath) return true;

  const modelDigest = extractDigestFromModelId(modelId);
  const expectedDigest = extractDigestFromModelId(expectedModelPath);
  return Boolean(modelDigest && expectedDigest && modelDigest === expectedDigest);
}

async function waitForLlamaCppReady(
  timeoutMs: number,
  expectedModelPath?: string,
  expectedPid?: number,
  expectedContextSize?: number,
): Promise<void> {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const runtime = await getLlamaCppRuntimeInfo();
    if (runtime.status === 200) {
      if (!expectedModelPath) return;
      const contextMatches = !expectedContextSize || !runtime.nCtx || runtime.nCtx === expectedContextSize;
      if (modelReferenceMatches(runtime.modelId, expectedModelPath) && contextMatches) return;
    }
    if (expectedPid && !isPidAlive(expectedPid)) {
      throw new Error(`llama.cpp runtime exited before becoming ready. Check ${LLAMACPP_START_LOG_PATH}`);
    }
    await sleep(2_000);
  }

  if (expectedModelPath) {
    throw new Error(`llama.cpp runtime did not become ready with ${path.basename(expectedModelPath)} before timeout`);
  }
  throw new Error('llama.cpp runtime did not become ready before timeout');
}

async function detectUnsupportedArchitectureFromLog(modelPath: string): Promise<string | null> {
  try {
    const content = await fs.readFile(LLAMACPP_START_LOG_PATH, 'utf-8');
    const marker = `starting model=${modelPath}`;
    const markerIndex = content.lastIndexOf(marker);
    const section = markerIndex >= 0 ? content.slice(markerIndex) : content;
    const matches = [...section.matchAll(/unknown model architecture:\s*'([^']+)'/gi)];
    if (matches.length === 0) return null;
    return matches[matches.length - 1][1].toLowerCase();
  } catch {
    return null;
  }
}

async function findListeningPidForPort(port: number): Promise<number | null> {
  const result = await runCommandCapture('netstat', ['-ano', '-p', 'tcp']);
  if (result.code !== 0) return null;

  const lines = result.stdout.split(/\r?\n/);
  for (const line of lines) {
    if (!line.includes('LISTENING')) continue;
    const parts = line.trim().split(/\s+/);
    if (parts.length < 5) continue;
    const localAddress = parts[1];
    const localPortRaw = localAddress.split(':').pop();
    const localPort = localPortRaw ? Number.parseInt(localPortRaw, 10) : NaN;
    if (!Number.isFinite(localPort) || localPort !== port) continue;
    const pid = Number.parseInt(parts[4], 10);
    if (Number.isFinite(pid) && pid > 0) return pid;
  }

  return null;
}

async function getProcessImageName(pid: number): Promise<string | null> {
  const result = await runCommandCapture('tasklist', ['/FI', `PID eq ${pid}`, '/FO', 'CSV', '/NH']);
  if (result.code !== 0) return null;
  const firstLine = result.stdout.split(/\r?\n/).find((line) => line.trim());
  if (!firstLine || firstLine.startsWith('INFO:')) return null;
  const match = firstLine.match(/^"([^"]+)"/);
  return match?.[1] ?? null;
}

async function killPidTree(pid: number): Promise<boolean> {
  const result = await runCommandCapture('taskkill', ['/PID', String(pid), '/T', '/F']);
  return result.code === 0;
}

async function terminateTrackedLlamaCppServer(): Promise<boolean> {
  let killed = false;
  const pid = await readLlamaCppPid();
  if (pid && isPidAlive(pid)) {
    killed = await killPidTree(pid);
  }
  await fs.rm(LLAMACPP_PID_FILE, { force: true });

  if (killed) return true;

  const portPid = await findListeningPidForPort(LLAMACPP_PORT);
  if (!portPid || !isPidAlive(portPid)) return false;

  const imageName = (await getProcessImageName(portPid))?.toLowerCase() ?? '';
  if (!imageName.includes('llama-server') && imageName !== 'cmd.exe') return false;

  return killPidTree(portPid);
}

async function resolveLlamaCppModelPath(selectedModel?: string): Promise<string | null> {
  const selected = selectedModel?.trim();
  if (!selected) return resolveDefaultLlamaCppModelPath();

  const selectedAsPath = path.resolve(selected);
  if (await fileExists(selectedAsPath)) {
    return selectedAsPath;
  }

  const blobPath = await resolveBlobPathFromModelTag(selected);
  if (blobPath) return blobPath;

  const digest = extractDigestFromModelId(selected);
  if (digest) {
    const digestPath = path.join(OLLAMA_BLOBS_DIR, `sha256-${digest}`);
    if (await fileExists(digestPath)) return digestPath;
  }

  return null;
}

async function spawnLlamaCppServer(modelPath: string, contextSize: number, ngl: number, fa: 'on' | 'off'): Promise<number> {
  if (!(await fileExists(LLAMACPP_EXE_PATH))) {
    throw new Error(`llama-server binary not found at ${LLAMACPP_EXE_PATH}`);
  }
  if (!(await fileExists(VSDEVCMD_PATH))) {
    throw new Error(`VS dev cmd not found at ${VSDEVCMD_PATH}`);
  }
  if (!(await fileExists(ONEAPI_SETVARS_PATH))) {
    throw new Error(`oneAPI setvars not found at ${ONEAPI_SETVARS_PATH}`);
  }

  await fs.mkdir(path.dirname(LLAMACPP_START_LOG_PATH), { recursive: true });
  await fs.appendFile(
    LLAMACPP_START_LOG_PATH,
    `\n[${new Date().toISOString()}] starting model=${modelPath} ctx=${contextSize} ngl=${ngl} fa=${fa}\n`,
    'utf-8',
  );
  const launchScriptPath = path.resolve(
    WORKSPACE_DIR,
    '.logs',
    `llamacpp-launch-${Date.now()}.cmd`,
  );
  const launchScript = [
    '@echo off',
    'setlocal',
    `call "${VSDEVCMD_PATH}" -arch=x64 >nul`,
    'if errorlevel 1 (',
    '  echo Failed to run VS developer environment script.',
    '  exit /b 100',
    ')',
    `call "${ONEAPI_SETVARS_PATH}" >nul`,
    'if errorlevel 1 (',
    '  echo Failed to run oneAPI setvars script.',
    '  exit /b 101',
    ')',
    `"${LLAMACPP_EXE_PATH}" -m "${modelPath}" -c ${contextSize} -ngl ${ngl} -fa ${fa} --host ${LLAMACPP_HOST} --port ${LLAMACPP_PORT}`,
    'set EXIT_CODE=%ERRORLEVEL%',
    'echo llama-server exited with code %EXIT_CODE%',
    'exit /b %EXIT_CODE%',
    '',
  ].join('\r\n');
  await fs.writeFile(launchScriptPath, launchScript, 'utf-8');
  const logFd = fsNative.openSync(LLAMACPP_START_LOG_PATH, 'a');
  const child = spawn('cmd.exe', ['/d', '/c', launchScriptPath], {
    detached: false,
    stdio: ['ignore', logFd, logFd],
    windowsHide: true,
  });
  fsNative.closeSync(logFd);
  child.unref();

  await fs.writeFile(LLAMACPP_PID_FILE, String(child.pid), 'utf-8');
  return child.pid ?? 0;
}

async function ensureLlamaCppServerReady(selectedModel?: string, requestedContextSize?: number): Promise<string> {
  const targetModelPath = await resolveLlamaCppModelPath(selectedModel);
  if (!targetModelPath) {
    throw new Error(`Unable to resolve model "${selectedModel ?? ''}" for llama.cpp runtime.`);
  }
  const targetContextSize = typeof requestedContextSize === 'number' && Number.isFinite(requestedContextSize) && requestedContextSize > 0
    ? Math.trunc(requestedContextSize)
    : LLAMACPP_NUM_CTX;
  const knownUnsupportedArch = getUnsupportedArchitectureForModelPath(targetModelPath);
  if (knownUnsupportedArch) {
    throw new Error(`Model architecture '${knownUnsupportedArch}' is not supported by this llama.cpp build. Use Ollama runtime for this model.`);
  }

  const trackedPid = await readLlamaCppPid();
  if (trackedPid && !isPidAlive(trackedPid)) {
    await fs.rm(LLAMACPP_PID_FILE, { force: true });
  }

  if (llamaCppStartPromise) {
    await llamaCppStartPromise;
  }

  const runtime = await getLlamaCppRuntimeInfo();
  if (runtime.status === 200) {
    const contextMatches = !runtime.nCtx || runtime.nCtx === targetContextSize;
    if (modelReferenceMatches(runtime.modelId, targetModelPath) && contextMatches) {
      return targetModelPath;
    }
    const stopped = await terminateTrackedLlamaCppServer();
    if (!stopped) {
      throw new Error(`llama.cpp runtime is already running with different model/context (${runtime.modelId ?? 'unknown'}, ctx=${runtime.nCtx ?? 'unknown'}). Use lightning eject button, or stop external llama-server process manually.`);
    }
  } else if (runtime.status !== null) {
    try {
      await waitForLlamaCppReady(45_000, targetModelPath, undefined, targetContextSize);
      return targetModelPath;
    } catch {
      const stopped = await terminateTrackedLlamaCppServer();
      if (!stopped) throw new Error('llama.cpp runtime is stuck loading and was not started by this app.');
    }
  }

  if (llamaCppStartPromise) {
    await llamaCppStartPromise;
    return targetModelPath;
  }

  llamaCppStartPromise = (async () => {
    const profiles: Array<{ ngl: number; fa: 'on' | 'off'; label: string }> = [
      { ngl: LLAMACPP_NGL, fa: LLAMACPP_FA, label: 'configured' },
    ];

    const addProfile = (ngl: number, fa: 'on' | 'off', label: string) => {
      if (profiles.some((profile) => profile.ngl === ngl && profile.fa === fa)) return;
      profiles.push({ ngl, fa, label });
    };

    if (LLAMACPP_NGL > 24) addProfile(24, 'on', 'fallback-ngl24-faon');
    if (LLAMACPP_NGL > 16) addProfile(16, 'on', 'fallback-ngl16-faon');
    if (LLAMACPP_NGL > 8) addProfile(8, 'on', 'fallback-ngl8-faon');
    addProfile(0, 'off', 'fallback-cpu');

    let lastError: Error | null = null;
    for (const profile of profiles) {
      const pid = await spawnLlamaCppServer(targetModelPath, targetContextSize, profile.ngl, profile.fa);
      console.log(`[llama.cpp] started pid=${pid} model=${targetModelPath} profile=${profile.label} ctx=${targetContextSize} ngl=${profile.ngl} fa=${profile.fa}`);
      try {
        await waitForLlamaCppReady(LLAMACPP_START_TIMEOUT_MS, targetModelPath, pid, targetContextSize);
        return;
      } catch (error: any) {
        lastError = error instanceof Error ? error : new Error(String(error));
        const unsupportedArch = await detectUnsupportedArchitectureFromLog(targetModelPath);
        if (unsupportedArch) {
          markUnsupportedArchitectureForModelPath(targetModelPath, unsupportedArch);
          throw new Error(`Model architecture '${unsupportedArch}' is not supported by this llama.cpp build. Use Ollama runtime for this model.`);
        }
        console.warn(`[llama.cpp] startup failed profile=${profile.label}: ${lastError.message}`);
        await terminateTrackedLlamaCppServer();
      }
    }

    throw lastError ?? new Error(`llama.cpp startup failed. Check ${LLAMACPP_START_LOG_PATH}`);
  })().finally(() => {
    llamaCppStartPromise = null;
  });

  await llamaCppStartPromise;
  return targetModelPath;
}

// Ensure sessions directory exists
await fs.mkdir(SESSIONS_DIR, { recursive: true });

// Per-session context managers in-memory cache
export const sessions = new Map<string, ContextManager>();

export interface AskUserRequest {
  kind: 'install' | 'guidance';
  prompt: string;
  command?: string;
  options?: string[];
}

export interface ActiveRun {
  model: string;
  status: 'thinking' | 'tool' | 'paused';
  statusText: string;
  partialContent: string;
  hasStartedGenerating: boolean;
  startedAt: number;
  updatedAt: number;
  provider?: string;
  /** Present while the run is paused waiting for the user to approve/guide. */
  pausedAskUser?: AskUserRequest;
}

export const activeRuns = new Map<string, ActiveRun>();

interface ExtractedEvidence {
  url: string;
  title: string;
  publisherAndDate: string;
  claims: string[];
  quotes: string[];
  relevanceScore: number;
  credibilityScore: number;
}

function extractEvidenceHeuristically(url: string, rawText: string): ExtractedEvidence {
  const lines = rawText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
  const title = lines[0]?.slice(0, 100) || 'Unknown Title';

  let publisherAndDate = 'Unknown';
  try {
    const parsedUrl = new URL(url);
    const domain = parsedUrl.hostname.replace('www.', '');
    publisherAndDate = domain.charAt(0).toUpperCase() + domain.slice(1);

    const datePattern = /\b(19|20)\d{2}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b/i;
    const dateMatch = rawText.match(datePattern);
    if (dateMatch) {
      publisherAndDate += ` / ${dateMatch[0]}`;
    }
  } catch {}

  const sentences: string[] = [];
  const quotes: string[] = [];

  for (const line of lines) {
    if (line.includes('{') || line.includes('}') || line.includes('class=') || line.includes('div>')) continue;

    const parts = line.split(/[.!?]\s+/);
    for (const part of parts) {
      const clean = part.trim().replace(/\s+/g, ' ');
      if (clean.length > 45 && clean.length < 250) {
        if (clean.includes('"') || clean.includes('“') || clean.includes('”') || clean.startsWith('-') || clean.startsWith('—')) {
          quotes.push(clean);
        } else {
          sentences.push(clean);
        }
      }
    }
  }

  const uniqueClaims = Array.from(new Set(sentences)).slice(0, 4);
  const uniqueQuotes = Array.from(new Set(quotes)).slice(0, 3);

  if (uniqueClaims.length === 0) {
    uniqueClaims.push(lines.find(l => l.length > 50)?.slice(0, 150) || 'Webpage was browsed successfully.');
  }

  return {
    url,
    title,
    publisherAndDate,
    claims: uniqueClaims,
    quotes: uniqueQuotes,
    relevanceScore: 4,
    credibilityScore: 4,
  };
}

async function extractEvidenceFromPage(
  model: string,
  url: string,
  rawText: string,
  signal: AbortSignal
): Promise<ExtractedEvidence> {
  // To keep speed and protect shared memory (LPDDR5) from choking, we bypass LLM sub-calls
  // for page evidence extraction unless explicitly disabled. Calling the same LLM inside the
  // tool-loop evicts/invalidates the main session's context/prompt cache, forcing a complete
  // context re-evaluation on every browse step. Fast heuristic extraction runs in 0ms and keeps
  // the main prompt cache 100% warm.
  if (process.env.FAST_HEURISTIC_EXTRACTION !== 'false') {
    return extractEvidenceHeuristically(url, rawText);
  }

  const systemPrompt = `You are a precise data extraction agent. Analyze the provided webpage text and extract structured evidence as JSON.
Respond ONLY with a JSON object in this format:
{
  "title": "Page Title",
  "publisherAndDate": "Publisher / Publication Date",
  "claims": ["Claim 1", "Claim 2", "Claim 3"],
  "quotes": ["Quote 1", "Quote 2"],
  "relevanceScore": 5, // 1 to 5 scale
  "credibilityScore": 5 // 1 to 5 scale
}`;

  const userPrompt = `URL: ${url}\n\nWebpage Text:\n${rawText.slice(0, 10000)}`;

  try {
    const body = {
      model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ],
      stream: false,
      options: { num_ctx: 12288 },
      format: 'json',
    };

    const response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
      dispatcher: inferenceDispatcher,
    });

    if (!response.ok) {
      throw new Error(`Ollama extraction failed: ${response.statusText}`);
    }

    const data = await response.json() as { message: { content: string } };
    const parsed = JSON.parse(data.message.content.trim());
    return {
      url,
      title: parsed.title || 'Unknown Title',
      publisherAndDate: parsed.publisherAndDate || 'Unknown Publisher/Date',
      claims: Array.isArray(parsed.claims) ? parsed.claims.map(String) : [],
      quotes: Array.isArray(parsed.quotes) ? parsed.quotes.map(String) : [],
      relevanceScore: typeof parsed.relevanceScore === 'number' ? parsed.relevanceScore : 3,
      credibilityScore: typeof parsed.credibilityScore === 'number' ? parsed.credibilityScore : 3,
    };
  } catch (error) {
    console.warn(`[extractEvidenceFromPage] failed, using fallback extraction:`, error);
    const titleMatch = rawText.match(/^([^\n]+)/);
    const title = titleMatch ? titleMatch[1].trim().slice(0, 100) : 'Unknown Title';
    const cleanLines = rawText.split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 50 && !line.includes('{') && !line.includes('}'))
      .slice(0, 3);
    return {
      url,
      title,
      publisherAndDate: 'Unknown',
      claims: cleanLines.length ? cleanLines : ['Failed to parse structured claims from page.'],
      quotes: [],
      relevanceScore: 3,
      credibilityScore: 3,
    };
  }
}

async function appendToResearchLedger(
  sessionId: string,
  url: string,
  record: ExtractedEvidence
) {
  const sessionLedgerPath = path.join(SESSIONS_DIR, `${sessionId}-ledger.json`);
  const activeLedgerPath = path.resolve(WORKSPACE_DIR, 'research-ledger.json');
  
  let currentLedger: any[] = [];
  try {
    const data = await fs.readFile(sessionLedgerPath, 'utf-8');
    currentLedger = JSON.parse(data);
    if (!Array.isArray(currentLedger)) {
      currentLedger = [];
    }
  } catch {
    currentLedger = [];
  }

  // Remove existing entries for the same URL to prevent duplication
  currentLedger = currentLedger.filter((entry: any) => entry.url !== url);
  currentLedger.push({
    ...record,
    timestamp: Date.now()
  });

  const content = JSON.stringify(currentLedger, null, 2);
  await fs.writeFile(sessionLedgerPath, content, 'utf-8');
  await fs.writeFile(activeLedgerPath, content, 'utf-8');
}

function formatCompactedSummary(extracted: ExtractedEvidence): string {
  const claimsStr = extracted.claims.map(c => `- ${c}`).join('\n');
  const quotesStr = extracted.quotes.map(q => `- "${q}"`).join('\n');
  return `[Webpage Browsed successfully]
Source: ${extracted.title} (${extracted.url})
Publisher/Date: ${extracted.publisherAndDate}
Relevance: ${extracted.relevanceScore}/5, Credibility: ${extracted.credibilityScore}/5

Extracted Claims:
${claimsStr || '- No specific claims extracted.'}

Key Quotes:
${quotesStr || '- No direct quotes extracted.'}`;
}

function truncateToolOutputForContext(content: string): string {
  if (content.length <= TOOL_CONTEXT_CHAR_LIMIT) return content;
  return `${content.slice(0, TOOL_CONTEXT_CHAR_LIMIT)}\n\n[Tool output truncated to ${TOOL_CONTEXT_CHAR_LIMIT} chars for faster context reuse.]`;
}


/**
 * Some reasoning models emit their chain of thought in `content` rather than
 * Ollama's separate `thinking` field, often without an explicit opening <think> tag.
 * Remove those tags and thinking text from answer content while forwarding their
 * inner text through the thinking callback.
 */
function createVisibleContentFilter(
  onThinking?: (content: string) => void,
  startInsideThink = false,
) {
  let insideThink = startInsideThink;
  let nativeThinkingSeen = false;
  let pending = '';
  const OPEN_TAGS = ['<think>', '<thought>'];
  const CLOSE_TAGS = ['</think>', '</thought>'];

  const filter = (chunk: string): string => {
    if (nativeThinkingSeen) {
      let cleaned = pending + chunk;
      pending = '';
      for (const tag of [...OPEN_TAGS, ...CLOSE_TAGS]) {
        cleaned = cleaned.replaceAll(new RegExp(tag, 'gi'), '');
      }
      return cleaned;
    }

    let input = pending + chunk;
    pending = '';
    let visible = '';

    while (input) {
      const lower = input.toLowerCase();

      if (insideThink) {
        let earliestCloseIdx = -1;
        let matchedCloseTag = '';
        for (const tag of CLOSE_TAGS) {
          const idx = lower.indexOf(tag);
          if (idx >= 0 && (earliestCloseIdx === -1 || idx < earliestCloseIdx)) {
            earliestCloseIdx = idx;
            matchedCloseTag = tag;
          }
        }

        let earliestOpenIdx = -1;
        let matchedOpenTag = '';
        for (const tag of OPEN_TAGS) {
          const idx = lower.indexOf(tag);
          if (idx >= 0 && (earliestOpenIdx === -1 || idx < earliestOpenIdx)) {
            earliestOpenIdx = idx;
            matchedOpenTag = tag;
          }
        }

        if (earliestOpenIdx >= 0 && (earliestCloseIdx === -1 || earliestOpenIdx < earliestCloseIdx)) {
          if (earliestOpenIdx > 0) {
            onThinking?.(input.slice(0, earliestOpenIdx));
          }
          input = input.slice(earliestOpenIdx + matchedOpenTag.length);
          continue;
        }

        if (earliestCloseIdx < 0) {
          const maxTagLen = 10;
          const safeLength = Math.max(0, input.length - (maxTagLen - 1));
          if (safeLength > 0) onThinking?.(input.slice(0, safeLength));
          pending = input.slice(safeLength);
          return visible;
        }

        if (earliestCloseIdx > 0) onThinking?.(input.slice(0, earliestCloseIdx));
        input = input.slice(earliestCloseIdx + matchedCloseTag.length);
        insideThink = false;
        continue;
      }

      let earliestOpenIdx = -1;
      let matchedOpenTag = '';
      for (const tag of OPEN_TAGS) {
        const idx = lower.indexOf(tag);
        if (idx >= 0 && (earliestOpenIdx === -1 || idx < earliestOpenIdx)) {
          earliestOpenIdx = idx;
          matchedOpenTag = tag;
        }
      }

      let earliestCloseIdx = -1;
      let matchedCloseTag = '';
      for (const tag of CLOSE_TAGS) {
        const idx = lower.indexOf(tag);
        if (idx >= 0 && (earliestCloseIdx === -1 || idx < earliestCloseIdx)) {
          earliestCloseIdx = idx;
          matchedCloseTag = tag;
        }
      }

      if (earliestCloseIdx >= 0 && (earliestOpenIdx === -1 || earliestCloseIdx < earliestOpenIdx)) {
        if (earliestCloseIdx > 0) {
          onThinking?.(input.slice(0, earliestCloseIdx));
        }
        input = input.slice(earliestCloseIdx + matchedCloseTag.length);
        insideThink = false;
        continue;
      }

      if (earliestOpenIdx < 0) {
        const maxTagLen = 10;
        const safeLength = Math.max(0, input.length - (maxTagLen - 1));
        visible += input.slice(0, safeLength);
        pending = input.slice(safeLength);
        return visible;
      }

      visible += input.slice(0, earliestOpenIdx);
      input = input.slice(earliestOpenIdx + matchedOpenTag.length);
      insideThink = true;
    }
    return visible;
  };

  filter.markNativeThinkingSeen = () => {
    nativeThinkingSeen = true;
    insideThink = false;
  };

  filter.flush = (): string => {
    if (insideThink) {
      if (pending) onThinking?.(pending);
      pending = '';
      return '';
    }
    const flushed = pending;
    pending = '';
    return flushed;
  };
  return filter;
}

const FALLBACK_TOOL_NAMES = new Set<ToolName>([
  'web_search',
  'browse_url',
  'write_file',
  'edit_file',
  'read_file',
  'run_terminal',
]);

function decodeXmlEntities(value: string): string {
  return value
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, '&')
    .replace(/&#(\d+);/g, (_, code: string) => String.fromCharCode(Number(code)));
}

interface GemmaFallbackCall {
  name: string;
  argumentsText: string;
  start: number;
  end: number;
}

// Gemma 4 sometimes emits its internal call syntax as visible text when native
// tool parsing is unavailable. Extract balanced argument objects so code/file
// content containing braces does not truncate the call.
function extractGemmaFallbackCalls(content: string): GemmaFallbackCall[] {
  const calls: GemmaFallbackCall[] = [];
  // Require the protocol at the start of its own line. This avoids executing
  // examples such as `the model emitted call:write_file{...}` from prose/code.
  const callPattern = /^[ \t]*call:([A-Za-z_][\w-]*)\s*\{/gim;
  let match: RegExpExecArray | null;

  while ((match = callPattern.exec(content)) !== null) {
    const openBrace = callPattern.lastIndex - 1;
    let depth = 0;
    let quote = '';
    let escaped = false;
    let end = -1;

    for (let i = openBrace; i < content.length; i += 1) {
      if (content.startsWith('<|"|>', i)) {
        quote = quote === '"' ? '' : '"';
        i += 4;
        continue;
      }
      const char = content[i];
      if (quote) {
        if (escaped) {
          escaped = false;
        } else if (char === '\\') {
          escaped = true;
        } else if (char === quote) {
          quote = '';
        }
        continue;
      }
      if (char === '"' || char === "'") {
        quote = char;
      } else if (char === '{') {
        depth += 1;
      } else if (char === '}') {
        depth -= 1;
        if (depth === 0) {
          end = i + 1;
          break;
        }
      }
    }

    if (end < 0) continue;
    const lineEnd = content.indexOf('\n', end);
    const trailing = content.slice(end, lineEnd < 0 ? content.length : lineEnd);
    if (trailing.trim()) continue;
    calls.push({
      name: match[1],
      argumentsText: content.slice(openBrace, end).replace(/<\|"\|>/g, '"'),
      start: match.index,
      end,
    });
    callPattern.lastIndex = end;
  }
  return calls;
}

function parseGemmaFallbackArguments(value: string): Record<string, unknown> | null {
  // Ollama's Gemma parser failure commonly emits unquoted property names and
  // sometimes `=` separators: {filepath:"x",limit=0}.
  let repaired = '';
  let quote = '';
  let escaped = false;
  for (let i = 0; i < value.length; i += 1) {
    const char = value[i];
    repaired += char;
    if (quote) {
      if (escaped) {
        escaped = false;
      } else if (char === '\\') {
        escaped = true;
      } else if (char === quote) {
        quote = '';
      }
      continue;
    }
    if (char === '"' || char === "'") {
      quote = char;
      continue;
    }
    if (char !== '{' && char !== ',') continue;

    let cursor = i + 1;
    let whitespace = '';
    while (cursor < value.length && /\s/.test(value[cursor])) {
      whitespace += value[cursor];
      cursor += 1;
    }
    const keyMatch = value.slice(cursor).match(/^([A-Za-z_][\w-]*)/);
    if (!keyMatch) continue;
    const key = keyMatch[1];
    cursor += key.length;
    let afterKeyWhitespace = '';
    while (cursor < value.length && /\s/.test(value[cursor])) {
      afterKeyWhitespace += value[cursor];
      cursor += 1;
    }
    if (value[cursor] !== ':' && value[cursor] !== '=') continue;

    repaired += `${whitespace}"${key}"${afterKeyWhitespace}:`;
    i = cursor;
  }
  try {
    const parsed = JSON.parse(repaired);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

// Parse tool calls emitted as visible model content.
function parseFallbackToolCalls(content: string): ToolCall[] {
  const calls: ToolCall[] = [];
  const seenCalls = new Set<string>();
  let nextId = 0;
  const canonicalize = (value: unknown): string => {
    if (Array.isArray(value)) return `[${value.map(canonicalize).join(',')}]`;
    if (value && typeof value === 'object') {
      const record = value as Record<string, unknown>;
      return `{${Object.keys(record).sort().map((key) =>
        `${JSON.stringify(key)}:${canonicalize(record[key])}`).join(',')}}`;
    }
    return JSON.stringify(value) ?? 'undefined';
  };
  const addCall = (name: string, args: Record<string, unknown>) => {
    if (!FALLBACK_TOOL_NAMES.has(name as ToolName)) return;
    const signature = `${name}:${canonicalize(args)}`;
    if (seenCalls.has(signature)) return;
    seenCalls.add(signature);
    calls.push({
      id: `fallback-${nextId++}`,
      type: 'function',
      function: { name, arguments: JSON.stringify(args) },
    });
  };

  const blocks = [...content.matchAll(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/gi)];
  for (const block of blocks) {
    const body = block[1].trim();
    if (body.startsWith('{')) {
      try {
        const parsed = JSON.parse(body);
        if (parsed?.name && parsed.arguments && typeof parsed.arguments === 'object') {
          addCall(String(parsed.name), parsed.arguments);
        }
      } catch {}
      continue;
    }
    for (const match of body.matchAll(/<function=([A-Za-z_][\w-]*)(?:\s*>)?([\s\S]*?)<\/function>/gi)) {
      const args: Record<string, string> = {};
      for (const parameter of match[2].matchAll(/<parameter=([^>\s]+)>([\s\S]*?)<\/parameter>/gi)) {
        args[decodeXmlEntities(parameter[1])] = decodeXmlEntities(parameter[2]);
      }
      addCall(match[1], args);
    }
  }

  const withoutBlocks = content.replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, '');
  for (const match of withoutBlocks.matchAll(/<function=([A-Za-z_][\w-]*)(?:\s*>)?([\s\S]*?)<\/function>/gi)) {
    const args: Record<string, string> = {};
    for (const parameter of match[2].matchAll(/<parameter=([^>\s]+)>([\s\S]*?)<\/parameter>/gi)) {
      args[decodeXmlEntities(parameter[1])] = decodeXmlEntities(parameter[2]);
    }
    addCall(match[1], args);
  }
  for (const call of extractGemmaFallbackCalls(content)) {
    const args = parseGemmaFallbackArguments(call.argumentsText);
    if (args) addCall(call.name, args);
  }
  return calls;
}

function removeFallbackToolCallMarkup(content: string): string {
  const withoutXml = content
    .replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, '')
    .replace(/<function=[A-Za-z_][\w-]*(?:\s*>)?[\s\S]*?<\/function>/gi, '')
    .replace(/<\/tool_call>/gi, '');
  const gemmaCalls = extractGemmaFallbackCalls(withoutXml);
  if (gemmaCalls.length === 0) return withoutXml;

  let cursor = 0;
  let cleaned = '';
  for (const call of gemmaCalls) {
    cleaned += withoutXml.slice(cursor, call.start);
    cursor = call.end;
  }
  return cleaned + withoutXml.slice(cursor);
}

function sanitizeNoToolsAssistantContent(content: string): string {
  if (!content) return '';
  const withoutThink = content
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<\/?think>/gi, '');
  const withoutMarkup = removeFallbackToolCallMarkup(withoutThink);
  const toolCallLine = /^\s*(web_search|browse_url|write_file|edit_file|read_file|run_terminal)\s*\(.+\)\s*$/i;
  const filteredLines = withoutMarkup
    .split('\n')
    .filter((line) => !toolCallLine.test(line.trim()));
  return filteredLines.join('\n').trim();
}

async function fetchOllamaChat(body: unknown, signal: AbortSignal): Promise<Awaited<ReturnType<typeof fetch>>> {
  let lastError: unknown;
  const retryDelaysMs = [0, 2_000, 5_000];

  for (let attempt = 0; attempt < retryDelaysMs.length; attempt += 1) {
    if (retryDelaysMs[attempt] > 0) {
      await new Promise((resolve) => setTimeout(resolve, retryDelaysMs[attempt]));
    }

    try {
      const response = await fetch(`${OLLAMA_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal,
        dispatcher: inferenceDispatcher,
      });
      if (response.ok || response.status < 500 || attempt === retryDelaysMs.length - 1) return response;
      lastError = new Error(`Ollama returned ${response.status} ${response.statusText}`);
      console.warn(`[Ollama chat] transient ${response.status}; retrying (${attempt + 1}/${retryDelaysMs.length})`);
    } catch (err: any) {
      if (err.name === 'AbortError') throw err;
      lastError = err;
      console.warn(`[Ollama chat] connection failed; retrying (${attempt + 1}/${retryDelaysMs.length}):`, err.cause?.message || err.message);
    }
  }

  throw lastError instanceof Error ? lastError : new Error('Unable to connect to Ollama');
}

async function fetchLlamaCppChat(body: unknown, signal: AbortSignal): Promise<Awaited<ReturnType<typeof fetch>>> {
  return fetch(`${LLAMACPP_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
    dispatcher: inferenceDispatcher,
  });
}

async function getModelCapabilities(model: string, signal: AbortSignal, backend: InferenceBackend): Promise<ModelCapabilities> {
  const cacheKey = `${backend}:${model}`;
  const cached = modelCapabilitiesCache.get(cacheKey);
  if (cached !== undefined) return cached;

  // Use model_map.json as a baseline / fallback for capability detection.
  // Ollama's /api/show only reports capabilities for models it explicitly tags;
  // many GGUF models have tool-call support but Ollama does not list it.
  const modelMap = await getModelMap();
  const mapDef = modelMap?.models?.find((m: any) => m.id === model);
  const mapSupportsTools = mapDef?.capabilities?.function_calling === true;
  const mapSupportsThinking = mapDef?.capabilities?.thinking_mode === true;
  const mapSupportsVision = mapDef?.capabilities?.vision === true;
  const mapContextLength: number | undefined = mapDef?.max_context;
  const mapSystemPromptType: string | undefined = mapDef?.system_prompt_type;

  if (backend === 'llamacpp') {
    const capabilities = {
      supportsTools: false,
      supportsThinking: false,
      supportsVision: false,
      contextLength: mapContextLength ?? (Number.isFinite(LLAMACPP_NUM_CTX) && LLAMACPP_NUM_CTX > 0 ? LLAMACPP_NUM_CTX : 32_768),
      systemPromptType: mapSystemPromptType,
    };
    modelCapabilitiesCache.set(cacheKey, capabilities);
    return capabilities;
  }

  try {
    const response = await fetch(`${OLLAMA_URL}/api/show`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: model }),
      signal,
    });
    if (!response.ok) {
      const capabilities = {
        supportsTools: mapSupportsTools,
        supportsThinking: mapSupportsThinking,
        supportsVision: mapSupportsVision,
        contextLength: mapContextLength,
        systemPromptType: mapSystemPromptType,
      };
      modelCapabilitiesCache.set(cacheKey, capabilities);
      return capabilities;
    }
    const data = await response.json() as {
      capabilities?: string[];
      model_info?: Record<string, number>;
    };
    const capabilities = {
      // OR Ollama's report with model_map so either source can enable tools
      supportsTools: (data.capabilities?.includes('tools') ?? false) || mapSupportsTools,
      supportsThinking: (data.capabilities?.includes('thinking') ?? false) || mapSupportsThinking,
      // Vision is authoritative from Ollama's live report: it reflects whether
      // an image projector (mmproj) is actually present. A model_map vision:true
      // must NOT override a live "no vision" — e.g. a VL-derived chat template
      // shipped without a projector — or we send images the runtime rejects
      // ("image input is not supported"). Fall back to model_map only when
      // Ollama returns no capabilities list at all.
      supportsVision: Array.isArray(data.capabilities)
        ? data.capabilities.includes('vision')
        : mapSupportsVision,
      // Prefer Ollama's value; fall back to model_map max_context
      contextLength: data.model_info?.['llama.context_length'] ?? mapContextLength,
      systemPromptType: mapSystemPromptType,
    };
    modelCapabilitiesCache.set(cacheKey, capabilities);
    return capabilities;
  } catch (error: any) {
    if (error.name === 'AbortError') throw error;
    const capabilities = {
      supportsTools: mapSupportsTools,
      supportsThinking: mapSupportsThinking,
      supportsVision: mapSupportsVision,
      contextLength: mapContextLength,
      systemPromptType: mapSystemPromptType,
    };
    modelCapabilitiesCache.set(cacheKey, capabilities);
    return capabilities;
  }
}

function normalizeOllamaMessages(messages: Message[], systemPromptType?: string): Message[] {
  const outboundMessages = messages.map((message) => {
    if (message.role !== 'assistant' || !message.tool_calls?.length) return message;

    const toolCalls = message.tool_calls.map((toolCall) => {
      if (typeof toolCall.function.arguments !== 'string') return toolCall;
      try {
        const parsed: unknown = JSON.parse(toolCall.function.arguments);
        if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
          return {
            ...toolCall,
            function: { ...toolCall.function, arguments: parsed },
          };
        }
      } catch {}
      return toolCall;
    });
    return { ...message, tool_calls: toolCalls as unknown as Message['tool_calls'] };
  });

  // Mistral: embed all system content into the first user message (no system role).
  if (systemPromptType?.toLowerCase() === 'mistral') {
    const systemContent = outboundMessages
      .filter((message) => message.role === 'system')
      .map((message) => message.content)
      .join('\n\n');
    if (!systemContent) return outboundMessages;

    const nonSystemMessages = outboundMessages.filter((message) => message.role !== 'system');
    const firstUserIndex = nonSystemMessages.findIndex((message) => message.role === 'user');
    if (firstUserIndex < 0) {
      return [{ role: 'user', content: systemContent }, ...nonSystemMessages];
    }

    return nonSystemMessages.map((message, index) => index === firstUserIndex
        ? { ...message, content: `${systemContent}\n\n${message.content}` }
        : message);
  }

  // General: hoist all system messages to the front as a single merged block.
  // Required for models (e.g. Qwen) whose Jinja template enforces system-first.
  // Preserves order of non-system messages; no-ops when already in correct order.
  const systemMessages = outboundMessages.filter((m) => m.role === 'system');
  if (systemMessages.length <= 1 && (systemMessages.length === 0 || outboundMessages[0].role === 'system')) {
    return outboundMessages;
  }

  const mergedSystemContent = systemMessages.map((m) => m.content).join('\n\n');
  const nonSystem = outboundMessages.filter((m) => m.role !== 'system');
  return [{ role: 'system' as const, content: mergedSystemContent }, ...nonSystem];
}

// Default persona fallbacks in case file read fails
const FALLBACK_PERSONAS: Record<string, string> = {
  coder: `You are an elite software engineering assistant. Your goal is to write correct, idiomatic, and highly optimized code. Provide complete, production-ready solutions and explain your structural choices concisely. You have full access to workspace file utilities and terminal commands. Use them to write, test, and verify your implementations.`,
  researcher: `You are a systematic data intelligence researcher. Your goal is to search, verify, and compile accurate information using web search and URL browsing tools. Always cross-reference facts, outline sources, and present findings in clean, structured markdown tables and summaries. Prioritize objective data and analytical depth. For a 4–7 source request, make at most 3 broad searches, then inspect 4–7 valid result URLs and write the report. Only browse URLs returned by a successful search or explicitly supplied by the user. If a search or page request fails, do not guess URLs or repeat the failed request; clearly report the limitation and provide only verifiable partial findings.`,
  creative: `You are a creative writer specializing in atmospheric dark fantasy, gothic mystery, and dramatic literature. Your style is rich, poetic, and immersive, focusing on dark aesthetics, emotional complexity, suspense, and vivid sensory details. Create compelling narratives, dramatic dialogue, and atmospheric settings while adhering to standard creative writing limits.`,
  system: `You are a general-purpose AI assistant and local model advisor. Answer user questions directly and honestly without hallucinating. Recommend the best installed model (Ollama or GPU-accelerated) for their task. Research unknown topics using web search and browsing tools when your knowledge is insufficient. Admit uncertainty and verify facts rather than guessing.`
};

// Dynamic markdown parser for personas.md
export async function getPersonaPrompt(personaKey: string): Promise<string> {
  try {
    const filePath = path.resolve(WORKSPACE_DIR, '../personas.md');
    const content = await fs.readFile(filePath, 'utf-8');
    
    // Split by header: "# Title"
    const sections = content.split(/^#\s+/m);
    const parsed: Record<string, string> = {};

    for (const section of sections) {
      const lines = section.split('\n');
      const title = lines[0].trim().toLowerCase();
      if (!title) continue;

      const body = lines.slice(1).join('\n').trim();
      if (body) {
        parsed[title] = body;
      }
    }

    const persona = parsed[personaKey] || FALLBACK_PERSONAS[personaKey] || FALLBACK_PERSONAS.coder;
    let rule = '';
    if (personaKey === 'creative') {
      rule = '\n\nResponse rule: Begin directly with the requested final answer. Creative Mode is prose-only: write the requested text in the chat response. Never call tools, write files, read files, or describe file operations.';
    } else if (personaKey === 'system') {
      rule = '\n\nTool-use guidance: You have access to web_search, browse_url, and read_file tools. Use them proactively to research unknown topics, verify uncertain claims, and recommend models based on current knowledge. For model recommendations, consider task complexity, speed requirements, and available VRAM. Admit when you don\'t know rather than guessing.';
    } else {
      rule = '\n\nTool-use mandate: You MUST use the provided tools (write_file, edit_file, run_terminal, read_file) to perform all file and directory operations. NEVER run package installation or initialization commands (npm install, npm init, npx tailwindcss init, pip install, etc.) in the terminal—write all configuration files (package.json, tailwind.config.js, vite.config.js, etc.) and code files directly using write_file. Provide install and run instructions for the user at the end in your final response. To FIX or CHANGE part of an existing file, use edit_file (replace a unique old_str with new_str) instead of rewriting the whole file with write_file. In each write_file call put "filepath" before "content". If you support reasoning/thinking, keep internal planning inside <think>...</think> strictly concise and bounded. Never draft whole file contents inside thoughts or meander; close </think> promptly and emit the tool call.';
    }
    return `${persona}${rule}`;
  } catch (err) {
    const persona = FALLBACK_PERSONAS[personaKey] || FALLBACK_PERSONAS.coder;
    let rule = '';
    if (personaKey === 'creative') {
      rule = '\n\nResponse rule: Begin directly with the requested final answer. Creative Mode is prose-only: write the requested text in the chat response. Never call tools, write files, read files, or describe file operations.';
    } else if (personaKey === 'system') {
      rule = '\n\nTool-use guidance: You have access to web_search, browse_url, and read_file tools. Use them proactively to research unknown topics, verify uncertain claims, and recommend models based on current knowledge. For model recommendations, consider task complexity, speed requirements, and available VRAM. Admit when you don\'t know rather than guessing.';
    } else {
      rule = '\n\nTool-use mandate: You MUST use the provided tools (write_file, edit_file, run_terminal, read_file) to perform all file and directory operations. NEVER run package installation or initialization commands (npm install, npm init, npx tailwindcss init, pip install, etc.) in the terminal—write all configuration files (package.json, tailwind.config.js, vite.config.js, etc.) and code files directly using write_file. Provide install and run instructions for the user at the end in your final response. To FIX or CHANGE part of an existing file, use edit_file (replace a unique old_str with new_str) instead of rewriting the whole file with write_file. In each write_file call put "filepath" before "content". If you support reasoning/thinking, keep internal planning inside <think>...</think> strictly concise and bounded. Never draft whole file contents inside thoughts or meander; close </think> promptly and emit the tool call.';
    }
    return `${persona}${rule}`;
  }
}

export async function getCtx(
  sessionId: string,
  fallbackPersona: string = 'coder',
  opts: { preferDisk?: boolean } = {},
): Promise<{ ctx: ContextManager; persona: string; model: string }> {
  // preferDisk bypasses the in-memory cache so readers (session history UI)
  // see the on-disk transcript even if a stale empty context was cached by
  // the brand-new-session race below. Active runs keep using their cache.
  if (sessions.has(sessionId) && !opts.preferDisk) {
    let currentPersona = fallbackPersona;
    let currentModel = '';
    let hasSavedSession = false;
    try {
      const filePath = path.join(SESSIONS_DIR, `${sessionId}.json`);
      const fileContent = await fs.readFile(filePath, 'utf-8');
      const data = JSON.parse(fileContent);
      hasSavedSession = true;
      if (data && data.persona) currentPersona = data.persona;
      if (data && typeof data.model === 'string') currentModel = data.model;
    } catch {}
    // The history endpoint may have initialized a brand-new session using its
    // default persona. Replace that temporary context with the persona chosen
    // by a quick-action button before the first message is sent.
    if (!hasSavedSession) {
      const ctx = new ContextManager(await getPersonaPrompt(fallbackPersona));
      sessions.set(sessionId, ctx);
      return { ctx, persona: fallbackPersona, model: '' };
    }
    return { ctx: sessions.get(sessionId)!, persona: currentPersona, model: currentModel };
  }

  const filePath = path.join(SESSIONS_DIR, `${sessionId}.json`);
  let loadedPersona = fallbackPersona;
  let loadedModel = '';
  let messagesToLoad: Message[] = [];

  try {
    const fileContent = await fs.readFile(filePath, 'utf-8');
    const data = JSON.parse(fileContent);
    if (data) {
      if (data.persona) loadedPersona = data.persona;
      if (typeof data.model === 'string') loadedModel = data.model;
      if (Array.isArray(data.messages)) messagesToLoad = data.messages;
    }
  } catch {}

  // Resolve system prompt dynamically from personas.md
  const systemPrompt = await getPersonaPrompt(loadedPersona);
  const ctx = new ContextManager(systemPrompt);

  if (messagesToLoad.length > 0) {
    ctx.reset();
    for (const msg of messagesToLoad) {
      if (msg.role !== 'system') {
        ctx.push(msg);
      }
    }
  }

  sessions.set(sessionId, ctx);
  return { ctx, persona: loadedPersona, model: loadedModel };
}

export async function saveSessionToFile(
  sessionId: string,
  ctx: ContextManager,
  persona: string,
  model: string,
  title?: string,
  provider?: string,
) {
  const filePath = path.join(SESSIONS_DIR, `${sessionId}.json`);
  const messages = ctx.getMessages();
  
  let sessionTitle = title;
  if (!sessionTitle) {
    const firstUser = messages.find((m) => m.role === 'user');
    sessionTitle = firstUser ? firstUser.content.slice(0, 30) + (firstUser.content.length > 30 ? '...' : '') : 'New Chat';
  }

  const payload = {
    sessionId,
    title: sessionTitle,
    persona,
    model,
    ...(provider ? { provider } : {}),
    updatedAt: Date.now(),
    messages,
    run: activeRuns.get(sessionId),
  };
  await fs.writeFile(filePath, JSON.stringify(payload, null, 2), 'utf-8');
}

function normalizeUrl(value: string): string {
  try {
    const parsed = new URL(value);
    parsed.hash = '';
    return parsed.toString().replace(/\/$/, '');
  } catch {
    return '';
  }
}

/** Reject model-invented URLs before a browser request is made. */
export function isBrowseUrlAllowed(url: string, messages: Message[]): boolean {
  const normalized = normalizeUrl(url);
  if (!normalized) return false;

  const urlPattern = /https?:\/\/[^\s)\]"']+/gi;
  return messages.some((message) => {
    // The user may explicitly supply a page to inspect.
    if (message.role !== 'user' && !(message.role === 'tool' && message.tool_name === 'web_search')) return false;
    return (message.content.match(urlPattern) ?? []).some((candidate) => normalizeUrl(candidate) === normalized);
  });
}

// GET /api/ollama/loaded — returns currently loaded model(s) resident in memory
router.get('/loaded', async (req: Request, res: Response) => {
  const backend = resolveInferenceBackend(req.query.backend);
  try {
    if (backend === 'llamacpp') {
      const runtimeLoadedModel = await getLlamaCppLoadedModelId();
      const runtimeInfo = await getLlamaCppRuntimeInfo();
      if (runtimeInfo.status === 200 && runtimeLoadedModel) {
        const runtimeFriendly = await resolveFriendlyLlamaModel(runtimeLoadedModel);
        res.json({
          backend: 'llamacpp',
          loaded: true,
          models: [
            {
              name: runtimeFriendly.mapId || runtimeLoadedModel,
              displayName: runtimeFriendly.displayName,
              model: runtimeLoadedModel,
              nCtx: runtimeInfo.nCtx,
            },
          ],
        });
        return;
      }
      res.json({ backend: 'llamacpp', loaded: false, models: [] });
      return;
    }

    const resp = await fetch(`${OLLAMA_URL}/api/ps`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!resp.ok) {
      res.json({ backend: 'ollama', loaded: false, models: [] });
      return;
    }
    const data = (await resp.json()) as any;
    const loadedList = Array.isArray(data?.models) ? data.models : [];
    res.json({
      backend: 'ollama',
      loaded: loadedList.length > 0,
      models: loadedList.map((m: any) => ({
        name: m.name || m.model,
        model: m.model,
        size: m.size,
        sizeVram: m.size_vram,
        expiresAt: m.expires_at,
      })),
    });
  } catch (err: any) {
    res.json({ backend, loaded: false, models: [], error: err.message });
  }
});

// GET /api/ollama/models
router.get('/models', async (req: Request, res: Response) => {
  const backend = resolveInferenceBackend(req.query.backend);
  const backendUrl = getBackendUrl(backend);
  try {
    if (backend === 'llamacpp') {
      const runtimeLoadedModel = await getLlamaCppLoadedModelId();
      const runtimeFriendly = runtimeLoadedModel ? await resolveFriendlyLlamaModel(runtimeLoadedModel) : null;
      const localModels = await listLocalOllamaManifestModels();
      const models = await Promise.all(localModels.map(async (model) => {
        const isLoaded = Boolean(runtimeFriendly?.mapId && runtimeFriendly.mapId === model.name);
        const blobPath = await resolveBlobPathFromModelTag(model.name);
        const unsupportedArchitecture = blobPath ? getUnsupportedArchitectureForModelPath(blobPath) : undefined;
        const syclAvailable = blobPath !== null && !unsupportedArchitecture;
        return {
          ...model,
          map_id: model.name,
          sycl_available: syclAvailable,
          sycl_reason: unsupportedArchitecture
            ? `unsupported architecture: ${unsupportedArchitecture}`
            : (blobPath ? undefined : 'no local GGUF blob found'),
          display_name: isLoaded ? `${model.name} (loaded SYCL)` : `${model.name} (SYCL)`,
          is_loaded: isLoaded,
        };
      }));

      if (runtimeLoadedModel && !models.some((model) =>
        model.name === runtimeLoadedModel || model.map_id === runtimeFriendly?.mapId
      )) {
        const loadedArchUnsupported = getUnsupportedArchitectureForModelPath(runtimeLoadedModel);
        models.unshift({
          name: runtimeLoadedModel,
          size: 0,
          modified_at: new Date(0).toISOString(),
          map_id: runtimeFriendly?.mapId ?? runtimeLoadedModel,
          sycl_available: !loadedArchUnsupported,
          sycl_reason: loadedArchUnsupported ? `unsupported architecture: ${loadedArchUnsupported}` : undefined,
          display_name: `${runtimeFriendly?.displayName ?? runtimeLoadedModel} (loaded SYCL)`,
          is_loaded: true,
        });
      }

      res.json({ models });
      return;
    }

    const [tagsResp, psResp] = await Promise.allSettled([
      fetch(`${OLLAMA_URL}/api/tags`, { signal: AbortSignal.timeout(5000) }),
      fetch(`${OLLAMA_URL}/api/ps`, { signal: AbortSignal.timeout(3000) }),
    ]);

    if (tagsResp.status !== 'fulfilled' || !tagsResp.value.ok) {
      const detail = tagsResp.status === 'fulfilled' ? await tagsResp.value.text() : tagsResp.reason?.message;
      res.status(502).json({ error: detail || 'Ollama unreachable' });
      return;
    }

    const tagsData = (await tagsResp.value.json()) as any;
    let loadedModelNames: string[] = [];
    if (psResp.status === 'fulfilled' && psResp.value.ok) {
      try {
        const psData = (await psResp.value.json()) as any;
        if (Array.isArray(psData?.models)) {
          loadedModelNames = psData.models.map((m: any) => m.name || m.model);
        }
      } catch {}
    }

    const models = (tagsData?.models || []).map((m: any) => {
      const isLoaded = loadedModelNames.some((loadedName) => loadedName === m.name || loadedName.startsWith(`${m.name}:`));
      return {
        ...m,
        is_loaded: isLoaded,
      };
    });

    res.json({ models });
  } catch (err: any) {
    console.error(`❌ Failed to connect to backend at ${backendUrl}:`, err.message);
    res.status(502).json({ error: err.message });
  }
});

// GET /api/ollama/model-map
router.get('/model-map', async (_req: Request, res: Response) => {
  try {
    const filePath = path.resolve(WORKSPACE_DIR, '../model_map.json');
    const content = await fs.readFile(filePath, 'utf-8');
    res.json(JSON.parse(content));
  } catch (err: any) {
    res.status(500).json({ error: `Failed to load model map: ${err.message}` });
  }
});

// POST /api/ollama/session — create an empty, named chat session
router.post('/session', async (req: Request, res: Response) => {
  const { sessionId, title, persona, model } = req.body as {
    sessionId?: string;
    title?: string;
    persona?: string;
    model?: string;
  };
  if (!sessionId?.trim()) {
    res.status(400).json({ error: 'Missing sessionId' });
    return;
  }

  try {
    const sessionPath = path.join(SESSIONS_DIR, `${sessionId}.json`);
    try {
      await fs.access(sessionPath);
      res.status(409).json({ error: 'Chat session already exists' });
      return;
    } catch {}

    const { ctx } = await getCtx(sessionId, persona || 'novelist');
    await saveSessionToFile(
      sessionId,
      ctx,
      persona || 'novelist',
      model || '',
      title?.trim() || 'Novel Outline',
    );
    res.json({ sessionId });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/ollama/sessions — list persistent sessions
router.get('/sessions', async (_req: Request, res: Response) => {
  try {
    const files = await fs.readdir(SESSIONS_DIR);
    const list = [];

    for (const file of files) {
      if (!file.endsWith('.json') || file.endsWith('-ledger.json')) continue;
      try {
        const content = await fs.readFile(path.join(SESSIONS_DIR, file), 'utf-8');
        const parsed = JSON.parse(content);
        if (typeof parsed.sessionId !== 'string' || !parsed.sessionId.trim()) continue;
        list.push({
          sessionId: parsed.sessionId,
          title: parsed.title || 'Untitled Chat',
          persona: parsed.persona || 'coder',
          model: typeof parsed.model === 'string' ? parsed.model : '',
          updatedAt: parsed.updatedAt || 0,
        });
      } catch {}
    }

    list.sort((a, b) => b.updatedAt - a.updatedAt);
    res.json(list);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/ollama/session?sessionId=... — load single session messages
router.get('/session', async (req: Request, res: Response) => {
  const sessionId = req.query.sessionId as string;
  if (!sessionId) { res.status(400).json({ error: 'Missing sessionId' }); return; }
  try {
    // While a run is active, serve the live in-memory context. Otherwise
    // prefer disk so a stale cached empty context can never hide history.
    const liveRun = activeRuns.get(sessionId);
    const { ctx, persona, model } = await getCtx(sessionId, 'coder', {
      preferDisk: !liveRun,
    });
    if (liveRun) {
      if (persona === 'researcher') {
        const sessionLedgerPath = path.join(SESSIONS_DIR, `${sessionId}-ledger.json`);
        const activeLedgerPath = path.resolve(WORKSPACE_DIR, 'research-ledger.json');
        try {
          const content = await fs.readFile(sessionLedgerPath, 'utf-8');
          await fs.writeFile(activeLedgerPath, content, 'utf-8');
        } catch {
          await fs.rm(activeLedgerPath, { force: true });
        }
      }
    }
    res.json({ messages: ctx.getMessages(), persona, model, run: activeRuns.get(sessionId) });
  } catch (err: any) {
    res.status(404).json({ error: err.message });
  }
});

// DELETE /api/ollama/session?sessionId=... — delete session file
router.delete('/session', async (req: Request, res: Response) => {
  const sessionId = req.query.sessionId as string;
  if (!sessionId) { res.status(400).json({ error: 'Missing sessionId' }); return; }
  try {
    sessions.delete(sessionId);
    const filePath = path.join(SESSIONS_DIR, `${sessionId}.json`);
    await fs.rm(filePath, { force: true });
    
    // Clean up ledger files
    const sessionLedgerPath = path.join(SESSIONS_DIR, `${sessionId}-ledger.json`);
    await fs.rm(sessionLedgerPath, { force: true });
    const activeLedgerPath = path.resolve(WORKSPACE_DIR, 'research-ledger.json');
    await fs.rm(activeLedgerPath, { force: true });

    res.json({ ok: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/ollama/stop
export const abortControllers = new Map<string, AbortController>();

// ── Human-in-the-loop pause (ask_user) ─────────────────────────────────────
// When the agent hits a wall — a side-effecting install it wants to run, or a
// tool-failure budget near exhaustion — the run pauses and surfaces an
// approval/guidance prompt to the user. The in-flight chat request awaits the
// matching resumeDeferred; the separate /resume route resolves it.
//
// Motivation: a weak model (e.g. gemma4-uncensored) was observed retrying an
// identical failing command (`npx tailwindcss init -p`, which Tailwind v4
// removed) four times until the failure budget forced a final report, then
// kept emitting tool calls after tools were disabled. Pausing for the human
// before the budget is burned lets the user approve the correct install
// (e.g. `npm install -D @tailwindcss/vite`) or steer the agent off the loop.
// 'timeout' kept as a type variant for backward compatibility but is no longer emitted —
// pauses wait indefinitely until the user responds or aborts the run.
export interface ResumeDecision {
  decision: 'approve' | 'deny' | 'guidance' | 'timeout' | 'approve_all';
  message?: string;
  command?: string;
}
export const resumeDeferred = new Map<string, { resolve: (d: ResumeDecision) => void; reject: (e: Error) => void }>();

// Commands that mutate the user's environment (install packages/tools). These
// pause for explicit approval before the agent is allowed to run them. Once a
// command is approved, its normalized form is allowlisted for the rest of the
// run so the agent's re-issue is not re-intercepted.
const SIDE_EFFECTING_INSTALL_RE = /(^|&&|;|\|\|)\s*(npm\s+(?:install|i|ci|install-ci|add)|pnpm\s+(?:add|install|i)|yarn\s+(?:add|install)|pip3?\s+install|python\s+-m\s+pip\s+install|pipx\s+install|uv\s+(?:pip\s+)?install|winget\s+install|choco\s+install|scoop\s+install|cargo\s+add|go\s+get|gem\s+install|apt(?:-get)?\s+install|brew\s+install|dotnet\s+(?:add|tool\s+install)|rustup\s+(?:component\s+)?add)\b/i;
function isSideEffectingInstall(command: string): boolean {
  return SIDE_EFFECTING_INSTALL_RE.test(command);
}
function normalizeCommand(command: string): string {
  return command.trim().replace(/\s+/g, ' ').toLowerCase();
}

router.post('/resume', (req: Request, res: Response) => {
  const sessionId: string | undefined = req.body?.sessionId;
  const deferred = sessionId ? resumeDeferred.get(sessionId) : undefined;
  if (!deferred) {
    res.status(404).json({ error: 'No pending approval for this session.' });
    return;
  }
  const decision: ResumeDecision = {
    decision: req.body?.decision === 'approve' ? 'approve'
      : req.body?.decision === 'approve_all' ? 'approve_all'
      : req.body?.decision === 'deny' ? 'deny'
      : req.body?.decision === 'guidance' ? 'guidance'
      : 'guidance',
    ...(typeof req.body?.message === 'string' && req.body.message.trim() ? { message: req.body.message.trim() } : {}),
    ...(typeof req.body?.command === 'string' && req.body.command.trim() ? { command: req.body.command.trim() } : {}),
  };
  deferred.resolve(decision);
  res.json({ ok: true });
});
router.post('/stop', (req: Request, res: Response) => {
  const { sessionId } = req.body as { sessionId: string };
  abortControllers.get(sessionId)?.abort();
  res.json({ ok: true });
});

// POST /api/ollama/unload — stop active work for a model and release it from memory.
router.post('/unload', async (req: Request, res: Response) => {
  const { model, inferenceBackend } = req.body as { model?: string; inferenceBackend?: InferenceBackend };
  if (!model) { res.status(400).json({ error: 'Missing model' }); return; }
  const backend = resolveInferenceBackend(inferenceBackend);

  for (const [sessionId, run] of activeRuns) {
    if (run.model === model) abortControllers.get(sessionId)?.abort();
  }

  if (backend === 'llamacpp') {
    const stopped = await terminateTrackedLlamaCppServer();
    if (stopped) {
      res.json({ ok: true, model, stopped: true });
      return;
    }
    const status = await getLlamaCppModelsStatus();
    if (status === 200) {
      res.status(409).json({ error: 'llama.cpp runtime is running but was not managed by this app. Stop external llama-server process manually.' });
      return;
    }
    res.json({ ok: true, model, stopped: false, detail: 'llama.cpp runtime was not running.' });
    return;
  }

  try {
    const response = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, keep_alive: 0, stream: false }),
    });
    if (!response.ok) {
      const detail = await response.text();
      res.status(response.status).json({ error: detail || response.statusText });
      return;
    }
    res.json({ ok: true, model });
  } catch (err: any) {
    res.status(502).json({ error: err.message });
  }
});

// POST /api/ollama/reset
router.post('/reset', async (req: Request, res: Response) => {
  const { sessionId } = req.body as { sessionId: string };
  sessions.delete(sessionId);
  clearSessionToolState(sessionId);
  const filePath = path.join(SESSIONS_DIR, `${sessionId}.json`);
  await fs.rm(filePath, { force: true });
  res.json({ ok: true });
});

// POST /api/ollama/truncate — drop every message at/after fromTimestamp so a
// prompt can be rerun from that point. System prompt (no created_at) survives.
router.post('/truncate', async (req: Request, res: Response) => {
  const { sessionId, fromTimestamp } = req.body as { sessionId: string; fromTimestamp: number };
  if (!sessionId || typeof fromTimestamp !== 'number') {
    res.status(400).json({ error: 'sessionId and fromTimestamp are required' });
    return;
  }
  try {
    abortControllers.get(sessionId)?.abort();
    activeRuns.delete(sessionId);
    sessions.delete(sessionId);

    const filePath = path.join(SESSIONS_DIR, `${sessionId}.json`);
    const data = JSON.parse(await fs.readFile(filePath, 'utf-8'));
    const kept = (data.messages as any[]).filter((m) => (m.created_at ?? 0) < fromTimestamp);
    data.messages = kept;
    data.updatedAt = Date.now();
    delete data.run;
    await fs.writeFile(filePath, JSON.stringify(data, null, 2), 'utf-8');
    res.json({ ok: true, remaining: kept.length });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/ollama/tokens
router.get('/tokens', async (req: Request, res: Response) => {
  const sessionId = req.query.sessionId as string;
  const { ctx } = await getCtx(sessionId);
  res.json(ctx.getTokenUsage());
});

// GET /api/ollama/attachment?sessionId=..&file=..
// Serves a stored attachment for display on session reload. Used by both the
// local and cloud UIs since sessions share one attachments layout on disk.
router.get('/attachment', async (req: Request, res: Response) => {
  const sessionId = String(req.query.sessionId ?? '');
  const file = path.basename(String(req.query.file ?? ''));
  if (!sessionId || !file || file === '.' || file === '..') {
    res.status(400).json({ error: 'sessionId and file are required.' });
    return;
  }
  try {
    const target = path.join(sessionAttachmentsDir(SESSIONS_DIR, sessionId), file);
    const buf = await fs.readFile(target);
    res.setHeader('Content-Type', contentTypeForFile(file));
    res.setHeader('Cache-Control', 'private, max-age=86400');
    res.send(buf);
  } catch {
    res.status(404).json({ error: 'Attachment not found.' });
  }
});

// POST /api/ollama/chat
router.post('/chat', async (req: Request, res: Response) => {
  const { sessionId, model, message, pendingBrowse, persona, isolated, numCtx, think, numThread, caveman, inferenceBackend, attachments, autopilotInstalls: autopilotInstallsBody } = req.body as {
    sessionId: string;
    model: string;
    message?: string;
    pendingBrowse?: string;
    persona?: string;
    isolated?: boolean;
    numCtx?: number;
    think?: boolean;
    numThread?: number;
    caveman?: boolean;
    inferenceBackend?: InferenceBackend;
    attachments?: IncomingAttachment[];
    /** When true, side-effecting installs run without per-command approval
     *  for the rest of the run (user opted in at session start, or flipped
     *  on via the "Allow all installs" button during a paused install). */
    autopilotInstalls?: boolean;
  };
  // Mutable so the "Allow all installs" button in the install alert can flip
  // it on at runtime, suppressing all future install prompts for this run.
  let autopilotInstalls = Boolean(autopilotInstallsBody);
  const backend = resolveInferenceBackend(inferenceBackend);
  const backendUrl = getBackendUrl(backend);
  let llamaCppModelIdForRequest: string | null = null;

  if (!model?.trim()) {
    res.status(400).json({ error: 'Select a model before sending a message.' });
    return;
  }

  const selectedPersona = persona || 'coder';
  const requestedContext = numCtx ?? (isolated ? STORY_NUM_CTX : OLLAMA_NUM_CTX);
  const requestedNumThread = typeof numThread === 'number' ? Math.trunc(numThread) : undefined;
  // No hard session-token cap. Context size is whatever the client sends (or
  // the OLLAMA_NUM_CTX default). Ollama enforces the model's own ceiling
  // server-side; we don't second-guess it here so large multi-file builds
  // (social media sites, monorepos) can ride the full available window.
  if (requestedContext < 1024) {
    res.status(400).json({ error: 'Context size too small (min 1024).' });
    return;
  }
  if (requestedNumThread !== undefined && (requestedNumThread < 1 || requestedNumThread > 64)) {
    res.status(400).json({ error: 'Thread count must be between 1 and 64.' });
    return;
  }
  if (think && backend === 'ollama') {
    const capabilities = await getModelCapabilities(model, new AbortController().signal, backend);
    if (!capabilities.supportsThinking) {
      res.status(400).json({ error: `Thinking Mode is not supported by ${model}.` });
      return;
    }
  }
  // Research runs repeatedly feed source material back to the model. A 32k
  // context is ample for the bounded source budget and avoids huge CPU KV
  // caches (the UI can otherwise request 131k+).
  // Otherwise respect the exact context size requested by the user.
  const activeContextSize = selectedPersona === 'researcher'
    ? Math.min(requestedContext, 32_768)
    : requestedContext;

  if (backend === 'llamacpp') {
    const requestedModelPath = await resolveLlamaCppModelPath(model);
    if (!requestedModelPath) {
      res.status(400).json({ error: `No local GGUF/blob found for "${model}". Select a local model for llama.cpp runtime.` });
      return;
    }
    const unsupportedArchitecture = getUnsupportedArchitectureForModelPath(requestedModelPath);
    if (unsupportedArchitecture) {
      res.status(400).json({ error: `Model architecture '${unsupportedArchitecture}' is not supported by current llama.cpp build. Use Ollama runtime for this model.` });
      return;
    }

    if (AUTO_START_LLAMACPP) {
      try {
        llamaCppModelIdForRequest = await ensureLlamaCppServerReady(model, activeContextSize);
      } catch (err: any) {
        res.status(502).json({ error: `Failed to start llama.cpp runtime: ${err.message}` });
        return;
      }
    } else {
      const runtime = await getLlamaCppRuntimeInfo();
      if (!runtime.modelId || runtime.status !== 200) {
        res.status(502).json({ error: 'llama.cpp runtime is offline. Start llama-server manually or set AUTO_START_LLAMACPP=true.' });
        return;
      }
      if (!modelReferenceMatches(runtime.modelId, requestedModelPath)) {
        res.status(400).json({ error: `llama.cpp runtime loaded a different model (${runtime.modelId}). Enable AUTO_START_LLAMACPP or switch runtime model manually.` });
        return;
      }
      if (runtime.nCtx && runtime.nCtx !== activeContextSize) {
        res.status(400).json({ error: `llama.cpp runtime context is ${runtime.nCtx}. Restart it with context ${activeContextSize}, or enable AUTO_START_LLAMACPP.` });
        return;
      }
      llamaCppModelIdForRequest = runtime.modelId;
    }
  }

  const { ctx } = await getCtx(sessionId, selectedPersona);
  // Story requests include their own compact continuity packet. Their full
  // exchange is still retained in the session, but never reused as inference
  // context for the following isolated story continuation.
  const requestCtx = isolated
    ? new ContextManager(await getPersonaPrompt(selectedPersona))
    : ctx;

  // Add the user message to the short-lived inference context and, for an
  // isolated Story Mode request, to the persistent session transcript too.
  let userMessage: Message | undefined;
  let novelDraftContext: string | null = null;
  let novelDraftTargetWords = 1500; // fallback; overwritten from outline entry
  
  // Check for novel draft command: [NOVEL_DRAFT:novelId:chapterNum]
  const novelDraftMatch = message?.match(/^\[NOVEL_DRAFT:([^:]+):(\d+)\]$/);
  const isNovelDraft = Boolean(novelDraftMatch && selectedPersona === 'novelist');
  // outputNumPredict is a let so it can be increased after we know the chapter
  // word target and whether thinking is enabled (thinking tokens count against
  // num_predict, so a generous overhead is needed when think=true).
  let outputNumPredict = isNovelDraft ? NOVEL_DRAFT_NUM_PREDICT : OLLAMA_NUM_PREDICT;
  if (novelDraftMatch && selectedPersona === 'novelist') {
    const [, novelId, chapterNumStr] = novelDraftMatch;
    const chapterNum = parseInt(chapterNumStr, 10);
    
    try {
      const novel = await novelStorage.getNovel(novelId);
      if (novel) {
        const outlineEntry = novel.outline.find(o => o.number === chapterNum);
        const rollingContext = await novelStorage.getRollingContext(novelId, chapterNum);
        const bible = novel.bible;
        
        // Build novel context
        let contextContent = '';
        
        // Partition Bible Characters by chapter roster to prevent early appearance of future characters
        const rosterNames = [
          outlineEntry?.pov_character,
          ...(outlineEntry?.characters_involved ?? []),
          ...(outlineEntry?.characters ?? []),
        ].filter(Boolean).map(c => c!.toLowerCase().trim());

        const presentChars = bible.characters.filter(c => {
          const charNameLower = c.name.toLowerCase();
          return rosterNames.some(r => charNameLower.includes(r) || r.includes(charNameLower));
        });

        const futureChars = bible.characters.filter(c => {
          const charNameLower = c.name.toLowerCase();
          return !rosterNames.some(r => charNameLower.includes(r) || r.includes(charNameLower));
        });

        // 1. Character & World Bible
        if (bible.pov || bible.characters.length > 0 || bible.locations.length > 0 || bible.facts.length > 0 || bible.style_notes) {
          contextContent += '## Character & World Bible\n';
          if (bible.pov) contextContent += `POV Style: ${bible.pov}\n`;
          if (presentChars.length > 0) {
            contextContent += '### Approved Characters For This Chapter\n';
            for (const char of presentChars) {
              contextContent += `- **${char.name}**: ${char.description}`;
              if (char.traits?.length) contextContent += ` (Traits: ${char.traits.join(', ')})`;
              if (char.relationships?.length) contextContent += ` (Relationships: ${char.relationships.join(', ')})`;
              contextContent += '\n';
            }
          }
          if (bible.locations?.length > 0) {
            contextContent += '### Key Locations\n';
            for (const loc of bible.locations) {
              contextContent += `- **${loc.name}**: ${loc.description}\n`;
            }
          }
          if (bible.facts?.length > 0) contextContent += `World Facts: ${bible.facts.join('; ')}\n`;
          if (bible.style_notes) contextContent += `Style & Tone: ${bible.style_notes}\n`;
          contextContent += '\n';
        }
        
        // 2. Author Steering & Course Correction Directives (MANDATORY OVERRIDES)
        if (bible.steering_notes && bible.steering_notes.length > 0) {
          contextContent += '## AUTHOR MANDATORY DIRECTIVES & COURSE CORRECTIONS\n';
          for (const note of bible.steering_notes) {
            contextContent += `- ${note}\n`;
          }
          contextContent += '\n';
        }

        // 3. Arc summary
        if (rollingContext.arcSummaries.length > 0) {
          contextContent += '## Story So Far\n';
          for (const arc of rollingContext.arcSummaries) {
            contextContent += `${arc.summary}\n`;
          }
          contextContent += '\n';
        }
        
        // 4. Recent chapter summaries
        if (rollingContext.recentSummaries.length > 0) {
          contextContent += '## Recent Chapters\n';
          for (const sum of rollingContext.recentSummaries) {
            contextContent += `Ch ${sum.chapter}: ${sum.summary}\n`;
          }
          contextContent += '\n';
        }
        
        // 5. Strict Chapter Outline & Scope
        if (outlineEntry) {
          contextContent += `## STRICT CHAPTER OUTLINE & SCOPE\n`;
          contextContent += `Chapter ${outlineEntry.number}: ${outlineEntry.title}\n`;
          contextContent += `POV Character: ${outlineEntry.pov_character}\n`;
          contextContent += `Description / Beats: ${outlineEntry.description || outlineEntry.beat_summary}\n`;
          contextContent += `Approved Roster: ${presentChars.map(c => c.name).join(', ') || (outlineEntry.characters_involved ?? []).join(', ') || 'POV character only'}\n\n`;

          if (futureChars.length > 0) {
            contextContent += `⚠️ ABSOLUTE CHARACTER RESTRICTION:\n`;
            contextContent += `DO NOT include, introduce, mention, or feature the following characters in Chapter ${chapterNum} (they belong to LATER chapters and are NOT present in this scene): ${futureChars.map(c => c.name).join(', ')}.\n\n`;
          }

          contextContent += `STRICT OUTLINE ADHERENCE MANDATE:\n`;
          contextContent += `1. Write strictly what is described in this chapter's outline beat.\n`;
          contextContent += `2. Do NOT introduce future characters or jump ahead to future outline events.\n`;
          contextContent += `3. Obey all Author Mandatory Directives above strictly.\n\n`;

          const requestedWords = Math.max(
            outlineEntry.target_words ?? 0,
            (outlineEntry.target_pages ?? 8) * (novel.manifest.words_per_page || 250),
          ) || 1500;
          novelDraftTargetWords = requestedWords;
          contextContent += `Target: approximately ${requestedWords} words.\n`;
          contextContent += `Write this chapter as immersive narrative prose. Include complete scenes with sensory detail, dialogue, and character action. Do not summarize or truncate. Write until the chapter's arc is complete, then end with the exact text [END_OF_CHAPTER] on its own line.\n`;
          contextContent += '\n';
        }
        
        // Raw tail for voice continuity
        if (rollingContext.rawTail) {
          contextContent += `## Previous Chapter Ending\n...${rollingContext.rawTail}\n\n`;
        }
        
        contextContent += `Now write Chapter ${chapterNum}. When complete, end with [END_OF_CHAPTER] on its own line.`;
        novelDraftContext = contextContent;
        
        // Replace the user message with a cleaner version
        userMessage = { role: 'user', content: `Draft Chapter ${chapterNum}: ${outlineEntry?.title ?? 'Untitled'}`, created_at: Date.now() };
      }
    } catch (err) {
      console.error('[Novel draft context assembly failed]', err);
    }
  }
  
  // Recalculate token budget now that we know novelDraftTargetWords and think.
  // Thinking tokens are counted against num_predict in Ollama, so we need a
  // large overhead when think=true to avoid the budget running out before the
  // model finishes writing prose.  At ~1.5 tokens/word:
  //   no thinking: targetWords × 1.5  + 1 500 safety margin  (min 8 192)
  //   with thinking: targetWords × 1.5 + 20 000 thinking overhead
  if (isNovelDraft) {
    const thinkingOverhead = think ? 20000 : 1500;
    outputNumPredict = Math.max(
      NOVEL_DRAFT_NUM_PREDICT,
      Math.ceil(novelDraftTargetWords * 1.5) + thinkingOverhead,
    );
    console.log(`[Novel draft] target=${novelDraftTargetWords} words, think=${Boolean(think)}, num_predict=${outputNumPredict}`);
  } else if (think && backend === 'ollama') {
    // Non-novel reasoning turn: grant the extra thinking ceiling so the model
    // doesn't exhaust its budget inside <think> and get cut off before any
    // visible content or tool call (mirrors the novel-draft overhead above).
    outputNumPredict = OLLAMA_NUM_PREDICT + THINKING_NUM_PREDICT_OVERHEAD;
  }

  // Persist any attachments to this session's folder and fold text files into
  // the prompt. Image metadata rides along on the user message; the bytes stay
  // on disk and are re-read when the model payload is built below.
  const { metas: attachmentMetas, textBlocks: attachmentTextBlocks } =
    await ingestAttachments(SESSIONS_DIR, sessionId, attachments);

  if (!userMessage) {
    if (pendingBrowse) {
      userMessage = { role: 'user', content: `[User approved browse: ${pendingBrowse}]`, created_at: Date.now() };
    } else if (message || attachmentMetas.length > 0) {
      userMessage = {
        role: 'user',
        content: `${message ?? ''}${attachmentTextBlocks}`,
        created_at: Date.now(),
        ...(attachmentMetas.length > 0 ? { attachments: attachmentMetas } : {}),
      };
    }
  }
  
  // If we have novel context, add it as a system message
  if (novelDraftContext) {
    requestCtx.push({ role: 'system', content: novelDraftContext, created_at: Date.now() });
  }
  
  if (userMessage) {
    requestCtx.push(userMessage);
    if (isolated) ctx.push(userMessage);
  }

  // Save state immediately after user prompt
  await saveSessionToFile(sessionId, ctx, selectedPersona, model);

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const send = (event: string, data: unknown) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  let generationStarted = false;
  const markGenerationStarted = () => {
    const run = activeRuns.get(sessionId);
    if (run) run.hasStartedGenerating = true;
    if (!generationStarted) {
      generationStarted = true;
      send('thinking_started', {});
    }
  };

  // The saved session can be much larger than the active runtime context.
  // Track the estimated outbound window separately so the UI meter reflects
  // what this request will send, not every historical token kept on disk.
  const initialOutputReserve = Math.min(
    Math.max(2_048, outputNumPredict),
    Math.floor(activeContextSize * 0.25),
  );
  let activePromptEstimatedTokens = requestCtx.getModelContext(
    Math.max(1_024, activeContextSize - initialOutputReserve - 2_048),
  ).estimatedTokens;
  const getLiveTokenUsage = (pendingMessages: Message[] = []) => {
    const usage = requestCtx.getTokenUsage(pendingMessages);
    return {
      ...usage,
      contextUsed: Math.min(
        activeContextSize,
        activePromptEstimatedTokens + estimateMessageTokens(pendingMessages),
      ),
      contextLimit: activeContextSize,
    };
  };
  const sendTokenUsage = (pendingMessages: Message[] = []) => {
    send('tokens', getLiveTokenUsage(pendingMessages));
  };
  sendTokenUsage();

  const ac = new AbortController();
  abortControllers.set(sessionId, ac);

  // ── Vision ingest ────────────────────────────────────────────────────────
  // If the user attached images but the active model can't see them, describe
  // them ONCE with the vision-role sidecar and inject that text. The image
  // bytes are then dropped from the message so the coder never re-encodes them
  // each turn (saving ~2k tokens/turn) and no large vision primary is needed.
  const imageAttachments = userMessage?.attachments?.filter((a) => a.kind === 'image') ?? [];
  if (backend === 'ollama' && imageAttachments.length > 0) {
    const activeCaps = await getModelCapabilities(model, ac.signal, backend);
    if (!activeCaps.supportsVision) {
      const roles = await getModelRoles();
      const visionModel = roles.vision;
      if (visionModel) {
        send('status', { message: `Active model has no vision; describing ${imageAttachments.length} image(s) with ${visionModel}.` });
        const imgs = await readImageAttachments(SESSIONS_DIR, sessionId, imageAttachments);
        const description = imgs.length > 0
          ? await runOneShotModel(
              visionModel,
              [{
                role: 'user',
                content: 'Describe every attached image in full, exhaustive detail: visible text, UI elements, layout, colors, code, diagrams, error messages — everything an engineer would need to act without seeing it. Output only the description.',
                images: imgs.map((i) => i.base64),
              }],
              ac.signal,
              { numCtx: 8_192, numPredict: 1_536 },
            )
          : null;
        if (description) {
          const note: Message = {
            role: 'system',
            content: `[Image description from vision model ${visionModel}; the primary model cannot see the raw image(s)]\n${description}`,
            created_at: Date.now(),
          };
          requestCtx.push(note);
          if (isolated) ctx.push(note);
          if (userMessage) {
            const remaining = (userMessage.attachments ?? []).filter((a) => a.kind !== 'image');
            if (remaining.length > 0) userMessage.attachments = remaining;
            else delete userMessage.attachments;
          }
          await saveSessionToFile(sessionId, ctx, selectedPersona, model);
          send('status', { message: 'Image description injected into context.' });
        } else {
          send('status', { message: 'Vision model unavailable or returned nothing; continuing without an image description.' });
        }
      }
    }
  }

  // ── ask_user pause primitives (closure-scoped to this run) ───────────────
  // approvedCommands: normalized commands the user has already approved this
  // run, so the agent's re-issue of an install isn't re-intercepted.
  const approvedCommands = new Set<string>();
  const NEAR_FAILURE_THRESHOLD = 3; // pause for guidance at 3 consecutive failures

  // Pause the run and surface an approval/guidance prompt to the user. Resolves
  // with the user's decision, or null if aborted, or {decision:'timeout'} after
  // PAUSE_TIMEOUT_MS. Sends a 5s heartbeat so the SSE stream and the client
  // stall detector don't time out while waiting.
  const pauseForUser = async (request: AskUserRequest): Promise<ResumeDecision | null> => {
    send('ask_user', { ...request, sessionId });
    const run = activeRuns.get(sessionId);
    if (run) {
      run.status = 'paused';
      run.statusText = 'Waiting for user approval…';
      run.pausedAskUser = request;
      run.updatedAt = Date.now();
    }
    await saveSessionToFile(sessionId, ctx, selectedPersona, model);

    return await new Promise<ResumeDecision | null>((resolve) => {
      let settled = false;
      let heartbeat: ReturnType<typeof setInterval> | undefined;
      const finish = (decision: ResumeDecision | null) => {
        if (settled) return;
        settled = true;
        if (heartbeat) clearInterval(heartbeat);
        ac.signal.removeEventListener('abort', onAbort);
        resumeDeferred.delete(sessionId);
        const r = activeRuns.get(sessionId);
        if (r) { r.pausedAskUser = undefined; r.updatedAt = Date.now(); }
        // A user decision ends a legitimate quiet period; reset the watchdog's
        // stall clock so post-resume inference gets a full quiet window.
        lastStreamActivityAtMs = Date.now();
        consecutiveIdleProbes = 0;
        resolve(decision);
      };
      const onAbort = () => finish(null);
      heartbeat = setInterval(() => send('status', { message: 'Waiting for user approval…' }), 5_000);
      ac.signal.addEventListener('abort', onAbort, { once: true });
      resumeDeferred.set(sessionId, {
        resolve: (d) => finish(d),
        reject: () => finish(null),
      });
    });
  };
  activeRuns.set(sessionId, {
    model,
    status: 'thinking',
    statusText: 'Thinking…',
    partialContent: '',
    hasStartedGenerating: false,
    startedAt: Date.now(),
    updatedAt: Date.now(),
  });

  let lastProgressSavedAt = 0;
  // ── Watchdog state (local runtimes only) ──────────────────────────────
  // Budgets: whole run ≤ RUN_MAX_DURATION_MS, each inference turn ≤
  // TURN_MAX_DURATION_MS, each tool call ≤ TOOL_MAX_DURATION_MS, plus instant
  // cut of a PROVEN hang. A hang requires: the response stream is open and has
  // been silent > STALL_QUIET_MS, and two consecutive health probes found no
  // tokens AND no compute. Anything inconclusive counts as alive.
  const watchdogEnabled = WATCHDOG_ENABLED && !isolated;
  const runStartedAtMs = Date.now();
  let turnStartedAtMs = Date.now();
  let lastStreamActivityAtMs = Date.now();
  let lastProbeNPast: number | null = null;
  let consecutiveIdleProbes = 0;
  let quietStatusSentAt = 0;
  // Ollama model-eviction watch: last /api/ps poll time and how many
  // consecutive polls found zero models resident.
  let lastPsCheckAtMs = 0;
  let consecutiveModelGone = 0;
  const noteStreamActivity = () => {
    lastStreamActivityAtMs = Date.now();
    consecutiveIdleProbes = 0;
  };
  let watchdogTimer: ReturnType<typeof setInterval> | undefined;
  const stopWatchdog = () => {
    if (watchdogTimer) {
      clearInterval(watchdogTimer);
      watchdogTimer = undefined;
    }
  };
  let terminalSentWatchdog = false;
  // Main-loop gate, declared here so the watchdog interval can observe it.
  let iterating = true;
  const killRun = async (reason: string) => {
    stopWatchdog();
    terminalSentWatchdog = true;
    console.warn(`[watchdog] session=${sessionId} killing run: ${reason}`);
    send('killed', { reason });
    ac.abort();
  };

  if (watchdogEnabled) {
    watchdogTimer = setInterval(() => {
      void (async () => {
        try {
          if (ac.signal.aborted || !iterating) { stopWatchdog(); return; }
          const run = activeRuns.get(sessionId);
          const runStatus = run?.status ?? 'thinking';
          const now = Date.now();

          // Budget 1 — whole-run wall clock.
          if (now - runStartedAtMs >= RUN_MAX_DURATION_MS) {
            await killRun(`run exceeded its ${Math.round(RUN_MAX_DURATION_MS / 60_000)}-minute total budget`);
            return;
          }

          // Pauses waiting for user approval are heartbeated by pauseForUser
          // and have no runtime involvement; skip stall evaluation entirely.
          if (runStatus === 'paused') return;

          // While a tool runs there is no model stream to monitor; the tool's
          // own Promise.race cap enforces TOOL_MAX_DURATION_MS instead.
          if (runStatus === 'tool') return;

          // Fast model-eviction check (Ollama only) — independent of the quiet
          // window. If the runtime has unloaded every model mid-generation, the
          // open request will hang forever, so stop the run promptly. Throttled
          // to MODEL_PS_INTERVAL_MS; only fires after MODEL_GONE_CHECKS_TO_KILL
          // consecutive empty polls so a legitimate one-shot swap can't trip it.
          if (MODEL_EVICTION_WATCH && backend === 'ollama' && now - lastPsCheckAtMs >= MODEL_PS_INTERVAL_MS) {
            lastPsCheckAtMs = now;
            const loaded = await listLoadedOllamaModels();
            if (loaded !== null) { // null = inconclusive read; never act on it
              if (loaded.length === 0) {
                consecutiveModelGone += 1;
                console.warn(`[watchdog] session=${sessionId} ollama ps shows no model loaded (${consecutiveModelGone}/${MODEL_GONE_CHECKS_TO_KILL})`);
                send('status', { message: `Ollama shows no model loaded (${consecutiveModelGone}/${MODEL_GONE_CHECKS_TO_KILL}) — checking for eviction…` });
                if (consecutiveModelGone >= MODEL_GONE_CHECKS_TO_KILL) {
                  await killRun('model ejected — Ollama unloaded the model mid-run (likely memory pressure); stopping so the run does not hang');
                  return;
                }
              } else if (consecutiveModelGone > 0) {
                consecutiveModelGone = 0;
              }
            }
          }

          // Budget 2 — single inference turn wall clock.
          if (now - turnStartedAtMs >= TURN_MAX_DURATION_MS) {
            await killRun(`a single generation turn exceeded ${Math.round(TURN_MAX_DURATION_MS / 60_000)} minutes`);
            return;
          }

          // Stall evidence window.
          const quietMs = now - lastStreamActivityAtMs;
          if (quietMs < STALL_QUIET_MS) {
            if (consecutiveIdleProbes > 0) consecutiveIdleProbes = 0;
            return;
          }
          // Surface the quiet state so the UI shows why probing is happening.
          if (now - quietStatusSentAt >= 30_000) {
            quietStatusSentAt = now;
            send('status', { message: `Quiet ${Math.round(quietMs / 1000)}s — verifying runtime health…` });
          }

          const probe = await probeRuntimeComputing(backend, lastProbeNPast);
          lastProbeNPast = probe.nPast;

          // Inconclusive rounds (probe failed) are never evidence of a hang.
          if (!probe.conclusive) return;

          if (probe.computing) {
            if (consecutiveIdleProbes > 0) {
              console.warn(`[watchdog] session=${sessionId} runtime computing again — clearing idle streak`);
            }
            consecutiveIdleProbes = 0;
            return;
          }

          consecutiveIdleProbes += 1;
          console.warn(`[watchdog] session=${sessionId} idle probe ${consecutiveIdleProbes}/${WATCHDOG_IDLE_PROBES_TO_KILL} quiet=${Math.round(quietMs / 1000)}s`);
          // Make the count-down visible so a stall never looks silent again.
          send('status', { message: `Runtime idle ${consecutiveIdleProbes}/${WATCHDOG_IDLE_PROBES_TO_KILL} — will stop the run if it stays idle` });
          if (consecutiveIdleProbes >= WATCHDOG_IDLE_PROBES_TO_KILL) {
            await killRun(`runtime confirmed hung — no tokens and no compute for ~${Math.round((STALL_QUIET_MS + WATCHDOG_IDLE_PROBES_TO_KILL * WATCHDOG_PROBE_INTERVAL_MS) / 60_000)} min`);
          }
        } catch (error) {
          console.warn('[watchdog] tick failed:', error);
        }
      })();
    }, WATCHDOG_PROBE_INTERVAL_MS);
    if (typeof watchdogTimer === 'object' && watchdogTimer !== null && 'unref' in watchdogTimer) {
      (watchdogTimer as any).unref?.();
    }
  }
  let toolCallCount = 0;
  let toolFailureCount = 0;
  let searchFailureCount = 0;
  // Consecutive failures (reset on any success). 3 → pause for guidance,
  // 5 → kill the run for human review. Cumulative counts above are for logs.
  let consecutiveFailures = 0;
  // Consecutive thinking-only / empty turns. Bounded retries with an explicit
  // "emit a tool call or final answer" nudge before the run is ended.
  let emptyTurnCount = 0;
  // Token estimate at the last rolling-summary generation, so the summary is
  // only regenerated after meaningful further growth.
  let lastSummaryTokens = 0;
  // Files this run created/edited (for the end-of-run review pass) and how
  // many review rounds have already fired.
  const runChangedFiles = new Set<string>();
  let reviewRounds = 0;
  const recentFailures: { name: string; detail: string }[] = [];
  // Set when the run is killed (5 consecutive failures) to exit the loop.
  let killed = false;
  let iterationNumber = 0;
  let novelDraftContinuationCount = 0;
  let novelDraftText = '';
  let ollamaRequestStartedAt = 0;
  let currentOllamaPhase: 'ollama_request' | 'ollama_stream' = 'ollama_request';
  let currentSystemPromptType: string | undefined;
  let currentAllowTools = false;
  let lastToolWasError = false;
  let lastToolMetadata: { name: string; success: boolean; durationMs: number } | undefined;
  const recordOllamaFailure = async (
    phase: 'ollama_http' | 'ollama_stream' | 'ollama_request',
    error: unknown,
    response?: { status?: number; statusText?: string }
  ) => {
    const detail = error as { name?: string; message?: string; cause?: { message?: string } } | undefined;
    await appendOllamaFailureLog({
      timestamp: new Date().toISOString(),
      sessionId,
      model,
      ollamaUrl: backendUrl,
      backend,
      persona: selectedPersona,
      phase,
      error: {
        name: detail?.name || 'Error',
        message: detail?.message || 'Unknown Ollama error',
        cause: detail?.cause?.message,
      },
      ...(response?.status !== undefined ? { status: response.status } : {}),
      ...(response?.statusText ? { statusText: response.statusText } : {}),
      iteration: iterationNumber,
      toolCallCount,
      toolFailureCount,
      searchFailureCount,
      requestedContext,
      activeContext: activeContextSize,
      think: backend === 'ollama' ? Boolean(think) : false,
      allowTools: currentAllowTools,
      systemPromptType: currentSystemPromptType ?? null,
      elapsedMs: ollamaRequestStartedAt ? Date.now() - ollamaRequestStartedAt : null,
      messageSummary: summarizeOllamaMessages(requestCtx.getMessages()),
      lastTool: lastToolMetadata ?? null,
    });
  };
  const saveProgress = async (partialContent: string, force = false) => {
    const run = activeRuns.get(sessionId);
    if (!run) return;
    run.partialContent = partialContent;
    run.updatedAt = Date.now();
    if (force || run.updatedAt - lastProgressSavedAt >= 750) {
      lastProgressSavedAt = run.updatedAt;
      await saveSessionToFile(sessionId, ctx, selectedPersona, model);
    }
  };

  await saveProgress('', true);

  // When a stream fails mid-generation (Ollama/llama.cpp crash), this
  // holds the partial assistant text so it can be pushed to context before
  // the session is saved — the model can continue instead of restarting.
  let pendingPartialContent = '';

  try {
    // `iterating` is declared above (watchdog scope) and initialized true.
    // When research tools become unavailable, make one final model pass with
    // tools disabled so it can write a candid partial report from successful
    // results already in the conversation.
    let forceFinalResponse = false;
    let finalResponseRetryCount = 0;
    let contextOverflowRetryCount = 0;
    let contextBudgetPenalty = 0;
    // Ollama can return HTTP 200 while silently discarding a malformed native
    // Gemma tool call. Retry once without native tool parsing so the raw call
    // remains visible and can be repaired by parseFallbackToolCalls.
    let nativeToolRecoveryRetry = false;
    // Set when a turn's hidden reasoning crossed the think budget and the
    // stream was cut. Forces think:false for the rest of THIS run so the model
    // spends its output on a real tool call / answer instead of another
    // runaway thought. Bounded by budgetInterruptionCount below.
    let forceNoThinkForRun = false;
    let budgetInterruptionCount = 0;
    // Tracks whether a terminal SSE event (done/stopped/error) was emitted.
    // Some loop exits (e.g. forced-final-response retry exhaustion) leave the
    // loop without a 'done', which would strand the client in a running state.
    let terminalSent = false;
    while (iterating && !ac.signal.aborted) {
      iterationNumber += 1;
      turnStartedAtMs = Date.now();
      lastStreamActivityAtMs = Date.now();
      consecutiveIdleProbes = 0;
      pendingPartialContent = '';
      const capabilities = await getModelCapabilities(model, ac.signal, backend);
      const activeContext = activeContextSize;
      // Creative Mode is intentionally prose-only. Completion-only GGUFs also
      // reject tool-role messages and assistant tool calls from old sessions.
      const allowTools = backend === 'ollama'
        && !forceFinalResponse
        && !isolated
        && selectedPersona !== 'creative'
        && capabilities.supportsTools;
      currentSystemPromptType = capabilities.systemPromptType;
      currentAllowTools = allowTools;
      const outputReserve = Math.min(
        Math.max(2_048, outputNumPredict),
        Math.floor(activeContext * 0.25),
      );
      const nonHistoryReserve = (allowTools ? 2_048 : 1_024) + outputReserve + contextBudgetPenalty;
      const historyBudget = Math.max(1_024, activeContext - nonHistoryReserve);

      // Rolling LLM summary: once history passes ~50% of the runtime window,
      // fold the older turns into a durable brief with the summarizer sidecar
      // so continuity survives when raw turns are dropped. Regenerated only on
      // meaningful further growth; unavailable summarizer falls back to the
      // heuristic compaction notice below.
      if (backend === 'ollama' && !isolated && !forceFinalResponse) {
        const estTokens = requestCtx.getTokenEstimate();
        const growthSinceSummary = estTokens - lastSummaryTokens;
        if (estTokens > activeContext * 0.5 && growthSinceSummary > activeContext * 0.15) {
          const roles = await getModelRoles();
          const summarizer = roles.summarizer;
          if (summarizer) {
            const digest = buildSessionDigest(requestCtx.getMessages());
            if (digest) {
              send('status', { message: `Context past 50%; summarizing earlier work with ${summarizer}.` });
              const summary = await runOneShotModel(
                summarizer,
                [
                  { role: 'system', content: 'You compress an in-progress software engineering session into a durable brief for the coding model that continues the work. Capture: the user\'s goals, decisions made, files created or edited (with their purpose), current state, and the immediate next steps. Be specific and terse. Do not invent facts. Output only the brief.' },
                  { role: 'user', content: digest },
                ],
                ac.signal,
                { numCtx: 32_768, numPredict: 1_024 },
              );
              if (summary) {
                requestCtx.setDurableSummary(summary);
                lastSummaryTokens = estTokens;
                await saveSessionToFile(sessionId, ctx, selectedPersona, model);
                send('status', { message: 'Rolling summary updated.' });
              } else {
                // Avoid hammering an unavailable summarizer every turn.
                lastSummaryTokens = estTokens;
              }
            }
          }
        }
      }

      const contextWindow = requestCtx.getModelContext(historyBudget);
      if (contextWindow.compacted && iterationNumber === 1) {
        send('status', {
          message: `Context compacted for ${Math.round(activeContext / 1_024)}k runtime window; full chat remains saved.`,
        });
      }
      const contextMessages = contextWindow.messages;
      const modelMessages = forceFinalResponse
        ? [...contextMessages, {
          role: 'system' as const,
          content: 'Research tools are no longer available for this run. Write the final report now using only successful tool results already provided. Clearly disclose the search limitation, distinguish verified findings from unknowns, and do not call tools or invent sources.',
        }]
        : allowTools
          ? contextMessages
          : contextMessages
            .filter((message) => message.role !== 'tool')
            .map(({ tool_calls, ...message }) => message);

      // (1) swap in shrunk tool descriptions to save input tokens (caveman-shrink concept)
      // (2) append a system directive that compresses prose output, while
      //     still allowing one-line status lines between tool calls so the UI
      //     user can see "ran weblink check on www.example.com — 4 hits." etc.
      const CAVEMAN_TOOL_DEFINITIONS = [
        { type: 'function', function: { name: 'web_search',   description: 'Brave Search; DuckDuckGo only if Brave reports exhausted credits. Returns results.', parameters: { type: 'object', properties: { query:    { type: 'string' } }, required: ['query'] } } },
        { type: 'function', function: { name: 'browse_url',   description: 'Fetch URL. Returns page text.',               parameters: { type: 'object', properties: { url:      { type: 'string' } }, required: ['url'] } } },
        { type: 'function', function: { name: 'write_file',   description: 'Create NEW file only. Existing files never overwritten; change them with exact edit_file replacements. History placeholders are not content.', parameters: { type: 'object', properties: { filepath: { type: 'string' }, content: { type: 'string' } }, required: ['filepath', 'content'] } } },
        { type: 'function', function: { name: 'edit_file',    description: 'Replace one exact unique block. Functional rewrites: one function/component per call.', parameters: { type: 'object', properties: { filepath: { type: 'string' }, old_str: { type: 'string' }, new_str: { type: 'string' } }, required: ['filepath', 'old_str', 'new_str'] } } },
        { type: 'function', function: { name: 'read_file',    description: 'Read numbered file chunk. Default 100 lines.', parameters: { type: 'object', properties: { filepath: { type: 'string' }, offset: { type: 'number' }, limit: { type: 'number' } }, required: ['filepath'] } } },
        { type: 'function', function: { name: 'run_terminal', description: 'Run validation/build/server shell command. Shell file mutations blocked; use write_file/edit_file.', parameters: { type: 'object', properties: { command:  { type: 'string' } }, required: ['command'] } } },
      ];

      const CAVEMAN_DIRECTIVE = [
        'Respond terse like smart caveman. All technical substance stay. Only fluff die.',
        'Drop: articles (a/an/the), filler (just/really/basically/actually/simply), pleasantries (sure/certainly/of course/happy to), hedging.',
        'Fragments OK. Short synonyms preferred. No decorative tables/emoji.',
        'Never drop: not/never/no/only/except. Numbers exact. Technical terms exact. Code blocks unchanged. Errors quoted exact.',
        'Tool calls: fire direct. After every tool result, emit ONE short status line (single sentence, <120 chars) before the next call or final answer — e.g. "Ran weblink check on www.example.com — 4 hits." or "Read repo/notes.md, 312 lines." Then proceed immediately.',
        'No preamble, no plan recap, no apologies, no hedging. Text before a call only to: clarify ambiguity, warn security/irreversible risk.',
        'Pattern between calls: [short status line] [next tool call OR final answer].',
      ].join('\n');

      const isFirstTurn = iterationNumber === 1;
      const isErrorTurn = lastToolWasError;
      // Once a turn overran the think budget, thinking stays off for the rest
      // of the run so a weak model cannot burn its whole output window inside
      // <think> a second time.
      const effectiveThink = Boolean(think) && !forceNoThinkForRun;
      const thinkBudgetForTurn: number = computeThinkBudgetForTurn({
        initialBudget: THINK_BUDGET_INITIAL,
        followupRatio: THINK_BUDGET_FOLLOWUP_RATIO,
        thinkEnabled: effectiveThink,
        supportsThinking: backend !== 'ollama' || capabilities.supportsThinking,
        isFirstTurn,
        isErrorTurn,
      });

      if (thinkBudgetForTurn > 0 && !forceFinalResponse) {
        let directiveContent = '';
        if (isFirstTurn) {
          directiveContent = `<think_budget>${thinkBudgetForTurn}</think_budget>\n` +
            `[Reasoning Constraint - Initial Turn]\n` +
            `You have a maximum budget of ${thinkBudgetForTurn} tokens for internal reasoning inside <think>...</think> for initial task decomposition and architecture planning.\n` +
            `Do not meander or write whole files inside <think>. Keep planning concise. Once planned, immediately close </think> and emit your tool call.`;
        } else if (isErrorTurn) {
          directiveContent = `<think_budget>${thinkBudgetForTurn}</think_budget>\n` +
            `[Reasoning Constraint - Error Recovery]\n` +
            `A previous tool call encountered an error. A full budget of ${thinkBudgetForTurn} tokens is available in <think> to diagnose the failure and plan the targeted correction. Focus strictly on fixing the error. Once planned, close </think> and execute the fix.`;
        } else {
          directiveContent = `<think_budget>${thinkBudgetForTurn}</think_budget>\n` +
            `[Reasoning Constraint - Execution / Follow-up Turn]\n` +
            `You are in active execution mode. Your <think> budget is strictly limited to ${thinkBudgetForTurn} tokens (at most 30% of initial budget) for a single quick check.\n` +
            `Do NOT replan the project or pre-generate entire files in <think>. Make one quick decision, close </think>, and proceed with the next tool call or response immediately.`;
        }
        modelMessages.push({ role: 'system' as const, content: directiveContent });
      }

      // File-writing discipline for tool-enabled coder turns. The recurring
      // failure was source code drafted inside <think> until the output budget
      // ran out, leaving only a placeholder or a truncated write_file. Keep
      // source strictly in tool arguments and require substantive files.
      if (allowTools && selectedPersona === 'coder' && !forceFinalResponse) {
        modelMessages.push({
          role: 'system' as const,
          content: [
            'File-writing discipline:',
            '- Put all source code ONLY inside write_file/edit_file arguments. Never draft file bodies inside <think> or in visible prose.',
            '- In each write_file call, emit "filepath" first, then the complete "content" for that file. Do not write placeholder, stub, or comment-only files (e.g. a lone "/* init */").',
            '- Write one complete, substantive file per write_file call, then move to the next file. To change a file you already wrote, use edit_file with one exact old_str/new_str.',
            '- If a file is too large for one response, split it across multiple write_file calls to the SAME path is NOT allowed (write_file is create-only); instead create it once with a complete first section, then extend it with edit_file.',
          ].join('\n'),
        });
      }
      if (caveman) {
        modelMessages.push({ role: 'system' as const, content: CAVEMAN_DIRECTIVE });
      }
      if (backend === 'llamacpp') {
        modelMessages.push({
          role: 'system' as const,
          content: 'No tools are available in this runtime. Do not emit tool calls, XML tags, function-like commands, or <think> blocks. Reply directly to the user in plain markdown.',
        });
      }
      if (nativeToolRecoveryRetry) {
        modelMessages.push({
          role: 'system' as const,
          content: [
            'Your previous response contained no usable text or parsed tool call. Ollama may have rejected malformed native tool syntax.',
            'Native tool parsing is disabled for this one recovery response.',
            'If you intended a tool call, emit exactly one valid JSON call in this form and no other text:',
            '<tool_call>{"name":"read_file","arguments":{"filepath":"path/to/file","offset":1,"limit":100}}</tool_call>',
            'Replace the example name and arguments with the intended provided tool. Quote every JSON key and string value. Do not emit call:name{...} syntax.',
            'If no tool is needed, answer the user directly in plain text.',
          ].join('\n'),
        });
      }

      activePromptEstimatedTokens = estimateMessageTokens(modelMessages) + (allowTools ? 1_024 : 0);
      sendTokenUsage();

      const normalizedMessages = normalizeOllamaMessages(modelMessages, capabilities.systemPromptType);
      let assistantContent = '';
      let assistantThinking = '';
      let thinkBudgetExceededThisTurn = false;
      // Set when the model stops because it hit the output-token cap rather than
      // finishing. A truncated response is what drops a large write_file's
      // trailing filepath argument, so we react by giving the next turn room.
      let responseTruncated = false;
      const pendingToolCalls: any[] = [];

      if (backend === 'ollama') {
        const activeToolDefs = caveman ? CAVEMAN_TOOL_DEFINITIONS : TOOL_DEFINITIONS;
        // Re-read image attachments from disk and pass them as Ollama `images`
        // (base64, no data-URI prefix). Done every turn so earlier images stay
        // visible to a vision model across a multi-turn conversation.
        const hasImageAttachments = normalizedMessages.some((message) =>
          message.attachments?.some((attachment) => attachment.kind === 'image'),
        );
        if (hasImageAttachments && !capabilities.supportsVision) {
          send('status', { message: 'Image attachments skipped because selected model lacks vision.' });
        }
        const messagesWithImages = await Promise.all(
          normalizedMessages.map(async (m) => {
            const wire: Record<string, unknown> = { ...m };
            delete wire.attachments;
            if (capabilities.supportsVision) {
              const imgs = await readImageAttachments(SESSIONS_DIR, sessionId, m.attachments);
              if (imgs.length > 0) wire.images = imgs.map((i) => i.base64);
            }
            return wire;
          }),
        );
        let turnNumPredict = outputNumPredict;
        if (effectiveThink && !isNovelDraft) {
          turnNumPredict = OLLAMA_NUM_PREDICT + (thinkBudgetForTurn > 0 ? thinkBudgetForTurn : THINKING_NUM_PREDICT_OVERHEAD);
        }
        const body = {
          model,
          messages: messagesWithImages,
          stream: true,
          // Supported by current Ollama releases; prevents models with optional
          // reasoning from spending their visible response on a hidden plan.
          think: Boolean(effectiveThink && !nativeToolRecoveryRetry),
          options: {
            num_ctx: activeContext,
            num_predict: turnNumPredict,
            ...(requestedNumThread !== undefined ? { num_thread: requestedNumThread } : {}),
          },
          keep_alive: OLLAMA_KEEP_ALIVE,
          ...(allowTools && !nativeToolRecoveryRetry ? { tools: activeToolDefs } : {}),
        };

        ollamaRequestStartedAt = Date.now();
        currentOllamaPhase = 'ollama_request';
        const ollamaResp = await fetchOllamaChat(body, ac.signal);

        if (!ollamaResp.ok) {
          const detail = await ollamaResp.text();
          const contextOverflow = ollamaResp.status === 400
            && /exceed(?:s|ed)?(?:_| )context|context size|n_prompt_tokens/i.test(detail);
          if (contextOverflow && contextOverflowRetryCount < 1) {
            contextOverflowRetryCount += 1;
            contextBudgetPenalty = Math.max(4_096, Math.floor(activeContext * 0.15));
            send('status', {
              message: 'Runtime rejected prompt size; compacting more aggressively and retrying once.',
            });
            continue;
          }
          await recordOllamaFailure('ollama_http', new Error('Ollama returned a non-OK response'), ollamaResp);
          // A model may have been removed after this chat was saved. Clear its
          // persisted selection so reopening the chat cannot keep retrying it.
          if (ollamaResp.status === 404) {
            await saveSessionToFile(sessionId, ctx, selectedPersona, '');
          }
          send('error', { message: `Ollama error: ${ollamaResp.status} ${detail || ollamaResp.statusText}` });
          break;
        }

        currentOllamaPhase = 'ollama_stream';
        contextOverflowRetryCount = 0;
        const appendThinking = (content: string) => {
          if (!content) return;
          assistantThinking += content;
          if (thinkBudgetForTurn > 0) {
            const thinkBudgetChars = thinkBudgetForTurn * THINK_TOKENS_TO_CHARS;
            if (assistantThinking.length > thinkBudgetChars && !thinkBudgetExceededThisTurn) {
              thinkBudgetExceededThisTurn = true;
              // Enforce, don't just warn: disable thinking for the rest of the
              // run and signal the read loop to cut this stream so the model
              // cannot keep drafting a whole file inside <think> until it hits
              // the output-token cap (which is what silently loses source).
              forceNoThinkForRun = true;
              send('think_budget_exceeded', {
                budget: thinkBudgetForTurn,
                actual: Math.round(assistantThinking.length / THINK_TOKENS_TO_CHARS),
                iteration: iterationNumber,
              });
              console.warn(`[think-budget] session=${sessionId} turn=${iterationNumber} budget=${thinkBudgetForTurn} actual≈${Math.round(assistantThinking.length / THINK_TOKENS_TO_CHARS)} tokens exceeded — cutting stream and disabling thinking for the run`);
            }
          }
          if (!caveman) {
            send('thinking', { content });
          }
          sendTokenUsage([{
            role: 'assistant',
            content: assistantContent,
            thinking: assistantThinking,
            tool_calls: pendingToolCalls,
          }]);
        };
        const startInsideThink = Boolean(
          effectiveThink && capabilities.supportsThinking && !nativeToolRecoveryRetry,
        );
        const visibleContent = createVisibleContentFilter(appendThinking, startInsideThink);

        const reader = ollamaResp.body!.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        while (true) {
          // A think-budget breach cuts the stream: stop reading and cancel the
          // upstream response so the runtime stops generating the runaway
          // thought. The turn is then handled by the recovery branch below.
          if (thinkBudgetExceededThisTurn) {
            try { await reader.cancel(); } catch {}
            break;
          }
          const { done, value } = await reader.read();
          if (done || ac.signal.aborted) break;

          buf += decoder.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop() ?? '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            let parsed: any;
            try { parsed = JSON.parse(trimmed); } catch { continue; }

            if (parsed.done === true && parsed.done_reason === 'length') {
              responseTruncated = true;
            }

            const msg = parsed.message;
            if (!msg) continue;

            if (msg.thinking) {
              markGenerationStarted();
              noteStreamActivity();
              visibleContent.markNativeThinkingSeen();
              appendThinking(msg.thinking);
              // A pure-<think> stretch emits no visible content, so without a
              // heartbeat here saveProgress never runs and the on-disk snapshot
              // (updatedAt / hasStartedGenerating) stays frozen at request
              // start — a reload or second tab then can't tell live thinking
              // from a true hang. saveProgress is throttled to 750ms, so this is
              // cheap. Surface the growing thought length so a reload shows
              // movement, not just a fresher timestamp.
              const run = activeRuns.get(sessionId);
              if (run) run.statusText = `Thinking… (${assistantThinking.length} chars)`;
              void saveProgress(assistantContent);
            }

            if (msg.content) {
              markGenerationStarted();
              noteStreamActivity();
              const content = visibleContent(msg.content);
              if (content) {
                assistantContent += content;
                pendingPartialContent = assistantContent;
                send('token', { content });
                sendTokenUsage([{ role: 'assistant', content: assistantContent }]);
                void saveProgress(assistantContent);
              }
            }

            if (msg.tool_calls?.length) {
              markGenerationStarted();
              noteStreamActivity();
              for (const tc of msg.tool_calls) {
                pendingToolCalls.push(tc);
              }
              sendTokenUsage([{
                role: 'assistant',
                content: assistantContent,
                thinking: assistantThinking,
                tool_calls: pendingToolCalls,
              }]);
            }
          }
        }

        const finalVisibleContent = visibleContent.flush();
        if (finalVisibleContent) {
          assistantContent += finalVisibleContent;
          send('token', { content: finalVisibleContent });
        }
      } else {
        ollamaRequestStartedAt = Date.now();
        currentOllamaPhase = 'ollama_request';
        const llamaResp = await fetchLlamaCppChat({
          model: llamaCppModelIdForRequest ?? model,
          messages: normalizedMessages
            .filter((message) => message.role !== 'tool')
            .map(({ tool_calls, thinking, ...message }) => message),
          stream: true,
          max_tokens: outputNumPredict,
        }, ac.signal);

        if (!llamaResp.ok) {
          await recordOllamaFailure('ollama_http', new Error(`llama.cpp returned ${llamaResp.status}`), llamaResp);
          const detail = await llamaResp.text();
          send('error', { message: `llama.cpp error: ${llamaResp.status} ${detail || llamaResp.statusText}` });
          break;
        }

        currentOllamaPhase = 'ollama_stream';
        const reader = llamaResp.body?.getReader();
        if (!reader) {
          throw new Error('llama.cpp returned an empty response stream.');
        }
        const decoder = new TextDecoder();
        let buf = '';
        let rawAssistantContent = '';
        let emittedContentChars = 0;
        let llamaCppThinkingChars = 0;

        const consumeDataLine = (line: string) => {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith(':')) return;
          const dataPrefixMatch = trimmed.match(/^data:\s*(.*)$/i);
          const payload = (dataPrefixMatch ? dataPrefixMatch[1] : trimmed)?.trim();
          if (!payload || payload === '[DONE]') return;

          let parsed: any;
          try {
            parsed = JSON.parse(payload);
          } catch {
            return;
          }

          // Watchdog liveness: any byte of completion data counts as stream
          // activity, even before content is extractable.
          noteStreamActivity();

          const deltaContent = parsed?.choices?.[0]?.delta?.content;
          const messageContent = parsed?.choices?.[0]?.message?.content;
          const contentChunk = typeof deltaContent === 'string'
            ? deltaContent
            : typeof messageContent === 'string'
              ? messageContent
              : '';
          if (contentChunk) {
            rawAssistantContent += contentChunk;
            pendingPartialContent = rawAssistantContent;
            markGenerationStarted();
            // Forward the growing text live so a long single-shot generation
            // streams into the UI instead of appearing all at once at the end.
            if (rawAssistantContent.length > emittedContentChars) {
              send('token', { content: rawAssistantContent.slice(emittedContentChars) });
              emittedContentChars = rawAssistantContent.length;
            }
            return;
          }

          // Some builds stream reasoning in delta.reasoning_content /
          // delta.reasoning before any visible content. Surface it through the
          // thinking channel so the user sees progress and the watchdog sees
          // activity instead of a silent multi-hour "ghost" run.
          const reasoningChunk = parsed?.choices?.[0]?.delta?.reasoning_content
            ?? parsed?.choices?.[0]?.delta?.reasoning;
          if (typeof reasoningChunk === 'string' && reasoningChunk) {
            markGenerationStarted();
            // appendThinking lives in the ollama branch scope; mirror its
            // visible behavior here without the think-budget machinery.
            if (!caveman) send('thinking', { content: reasoningChunk });
            llamaCppThinkingChars += reasoningChunk.length;
            const run = activeRuns.get(sessionId);
            if (run) run.statusText = `Reasoning… (${llamaCppThinkingChars} chars)`;
          }
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done || ac.signal.aborted) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop() ?? '';
          for (const line of lines) {
            consumeDataLine(line);
          }
        }
        buf += decoder.decode();
        if (buf.trim()) {
          consumeDataLine(buf);
        }

        assistantContent = sanitizeNoToolsAssistantContent(rawAssistantContent);
        if (!assistantContent) {
          assistantContent = 'This llama.cpp runtime does not execute tools. Ask a direct question, or switch Runtime to Ollama for tool-enabled coding tasks.';
        }
        if (assistantContent && !generationStarted) {
          markGenerationStarted();
          send('token', { content: assistantContent });
        }
        sendTokenUsage([{ role: 'assistant', content: assistantContent }]);
        await saveProgress(assistantContent, true);
      }

      if (allowTools && pendingToolCalls.length === 0) {
        const fallbackToolCalls = parseFallbackToolCalls(assistantContent);
        if (fallbackToolCalls.length > 0) {
          pendingToolCalls.push(...fallbackToolCalls);
          assistantContent = removeFallbackToolCallMarkup(assistantContent);
        }
      }

      // Think-budget breach recovery. The stream was cut mid-thought, so the
      // interrupted reasoning is deliberately discarded (never pushed to
      // context) and the model is asked to redo the turn with thinking off and
      // tools still available, so it emits a real tool call / answer instead of
      // another runaway thought. Bounded so a persistently mute model still
      // falls through to the empty-turn ladder.
      if (
        shouldRecoverAfterThinkBreach({
          backendOllama: backend === 'ollama',
          budgetExceededThisTurn: thinkBudgetExceededThisTurn,
          hasVisibleContent: Boolean(assistantContent.trim()),
          hasPendingToolCalls: pendingToolCalls.length > 0,
          interruptionCount: budgetInterruptionCount,
          maxRetries: MAX_EMPTY_TURN_RETRIES,
          aborted: ac.signal.aborted,
        })
      ) {
        budgetInterruptionCount += 1;
        nativeToolRecoveryRetry = false;
        pendingPartialContent = '';
        requestCtx.push({
          role: 'system',
          content: 'Your internal reasoning exceeded its budget and was cut off. Thinking is now disabled for the rest of this task. Do NOT plan silently. Emit EITHER exactly one tool call now (put "filepath" before the complete "content" for write_file), OR your final answer in plain text. Write source code only inside tool arguments, never in your thoughts.',
          created_at: Date.now(),
        });
        send('status', { message: `Reasoning budget exceeded (${budgetInterruptionCount}/${MAX_EMPTY_TURN_RETRIES}); disabling thinking for this run and retrying with a direct tool call.` });
        await saveSessionToFile(sessionId, ctx, selectedPersona, model);
        continue;
      }

      if (backend === 'ollama' && !assistantContent.trim() && pendingToolCalls.length === 0) {
        // Step 1: a thinking-only / empty turn. Retry once with native tool
        // parsing disabled and thinking off (think:false is derived from
        // nativeToolRecoveryRetry in the request body), so a malformed native
        // tool call becomes repairable and the model spends the turn on output.
        if (allowTools && !nativeToolRecoveryRetry && !ac.signal.aborted) {
          nativeToolRecoveryRetry = true;
          send('status', {
            message: 'Runtime returned an empty tool response; retrying once with visible tool-call recovery.',
          });
          continue;
        }

        // Step 2: still empty. Rather than silently ending the run (the "model
        // thinks, emits nothing, session dies" pattern), push an explicit
        // directive and let the model try again a bounded number of times.
        // Each empty turn counts as a consecutive failure so a persistently
        // mute model is eventually killed for review instead of hanging.
        emptyTurnCount += 1;
        consecutiveFailures += 1;
        recentFailures.push({ name: 'empty_turn', detail: 'model produced only reasoning, no answer or tool call' });
        if (!ac.signal.aborted && emptyTurnCount < MAX_EMPTY_TURN_RETRIES && consecutiveFailures < MAX_CONSECUTIVE_FAILURES) {
          nativeToolRecoveryRetry = false;
          requestCtx.push({
            role: 'system',
            content: 'Your previous turn produced only internal reasoning and no visible output. Do NOT think silently. Emit EITHER exactly one tool call now, OR your final answer to the user in plain text. If the task is finished, state that it is done and summarize what you built.',
            created_at: Date.now(),
          });
          send('status', { message: `Empty turn ${emptyTurnCount}/${MAX_EMPTY_TURN_RETRIES}; nudging the model to emit a tool call or final answer.` });
          await saveSessionToFile(sessionId, ctx, selectedPersona, model);
          continue;
        }

        const message = 'The model produced only reasoning with no answer or tool call across multiple attempts. It may be stuck or the context may be too large; try again, reduce the context, or switch models.';
        await recordOllamaFailure('ollama_stream', new Error(message));
        send('error', { message });
        terminalSent = true;
        iterating = false;
        break;
      }
      nativeToolRecoveryRetry = false;
      // A turn with real output clears the empty-turn streak.
      emptyTurnCount = 0;

      // Push assistant response. For novel drafts strip [END_OF_CHAPTER] from the
      // stored context so continuation passes don't see the model's own "done"
      // signal and restart from scratch instead of continuing the prose.
      const assistantCtxContent = isNovelDraft
        ? assistantContent.replace(/\[END_OF_CHAPTER\]/gi, '').trimEnd()
        : assistantContent;
      const assistantMsg: Message = {
        role: 'assistant',
        content: assistantCtxContent,
        thinking: !caveman && assistantThinking ? assistantThinking : undefined,
        tool_calls: pendingToolCalls.length > 0 ? pendingToolCalls : undefined,
        created_at: Date.now(),
      };
      pendingPartialContent = '';
      requestCtx.push(assistantMsg);
      if (isolated) ctx.push(assistantMsg);
      
      let continueNovelDraft = false;

      // Novel drafts get bounded automatic continuation. Models often stop
      // after a scene even when target length has not been reached.
      if (novelDraftMatch && isNovelDraft) {
        const [, novelId, chapterNumStr] = novelDraftMatch;
        const chapterNum = parseInt(chapterNumStr, 10);

        const segmentText = assistantContent
          .replace(/\[END_OF_CHAPTER\]/gi, '')
          .trim();
        if (segmentText) {
          novelDraftText += `${novelDraftText ? '\n\n' : ''}${segmentText}`;
        }

        const draftWordCount = novelDraftText.split(/\s+/).filter(Boolean).length;
        if (
          draftWordCount < novelDraftTargetWords
          && novelDraftContinuationCount < NOVEL_DRAFT_MAX_CONTINUATIONS
          && !ac.signal.aborted
        ) {
          novelDraftContinuationCount += 1;
          continueNovelDraft = true;
          // Anchor the model at the exact end of what was written so it
          // continues rather than restarts. The last ~150 words of prose
          // serve as a verbatim "you are here" marker.
          const proseTail = novelDraftText.slice(-800);
          const remaining = novelDraftTargetWords - draftWordCount;
          // Give the continuation pass a budget sized for the remaining words
          // plus the same thinking overhead.
          const thinkingOverhead = think ? 20000 : 1500;
          outputNumPredict = Math.max(
            NOVEL_DRAFT_NUM_PREDICT,
            Math.ceil(remaining * 1.5) + thinkingOverhead,
          );
          const continuationPrompt: Message = {
            role: 'user',
            content: `The chapter draft is not yet complete. Continue writing from exactly where the prose ends below. Do not repeat, recap, or restart. Do not add a chapter heading or scene divider. Seamlessly continue the prose:\n\n...${proseTail}\n\n---\nContinue from the last word above. Write at least ${remaining} more words to finish the chapter. End with [END_OF_CHAPTER] only when the chapter's arc is complete.`,
            created_at: Date.now(),
          };
          requestCtx.push(continuationPrompt);
          if (isolated) ctx.push(continuationPrompt);
          send('status', {
            message: `Extending chapter draft (${draftWordCount}/${novelDraftTargetWords} words)...`,
          });
        } else {
          const chapterText = novelDraftText.trim();
          if (chapterText) {
            try {
              await novelStorage.saveChapterText(novelId, chapterNum, chapterText);
              await novelStorage.updateChapterOutline(novelId, chapterNum, { status: 'drafted' });
              await novelStorage.updateManifest(novelId, { current_chapter: chapterNum, status: 'drafting' });
            } catch (err) {
              console.error('[Failed to save novel chapter]', err);
            }
          }
        }
      }

      // Save state after assistant response
      await saveSessionToFile(sessionId, ctx, selectedPersona, model);

      if (continueNovelDraft) {
        continue;
      }

      if (responseTruncated && !isNovelDraft && !ac.signal.aborted) {
        // Give the next turn more room so a re-issued tool call (or answer) can
        // complete. Any incomplete tool calls from this turn are still rejected
        // below; the larger budget lets the model's retry actually fit.
        const bumped = Math.min(outputNumPredict * 2, 32_768);
        if (bumped > outputNumPredict) {
          outputNumPredict = bumped;
          send('status', { message: `Previous output hit the token limit; retrying with a larger budget (${outputNumPredict} tokens)…` });
        }
      }

      if (pendingToolCalls.length > 0 && !ac.signal.aborted) {
        if (forceFinalResponse) {
          // Reject any tool calls generated when tools are no longer available,
          // and ask the model to produce the final report.
          for (const tc of pendingToolCalls) {
            const toolName: ToolName = tc.function?.name;
            send('tool_start', { name: toolName, args: {} });
            send('tool_result', {
              name: toolName,
              success: false,
              output: 'Research tools are no longer available. Write the final report now using only successful tool results already provided.',
              durationMs: 0,
            });
            requestCtx.push({
              role: 'tool',
              content: 'Research tools are no longer available. Write the final report now using only successful tool results already provided. Do not call tools or invent sources.',
              tool_call_id: tc.id ?? toolName,
              tool_name: toolName,
              duration_ms: 0,
              created_at: Date.now(),
            });
          }
          await saveSessionToFile(sessionId, ctx, selectedPersona, model);
          finalResponseRetryCount += 1;
          if (finalResponseRetryCount >= 2) {
            iterating = false;
          }
          continue;
        }

        let toolBudgetExceeded = false;
        let limitReason = '';
        let anyToolFailedInTurn = false;
        for (let i = 0; i < pendingToolCalls.length; i++) {
          const tc = pendingToolCalls[i];

          // Near-exhaustion: 3 of 4 tool failures. Before the next attempt burns
          // the last failure and forces a final report, pause for human guidance
          // so the user can approve a fix command or steer the agent off a loop
          // (the gemma4 / `npx tailwindcss init -p` repeat-failure case).
          if (!forceFinalResponse && consecutiveFailures >= NEAR_FAILURE_THRESHOLD && consecutiveFailures < MAX_CONSECUTIVE_FAILURES) {
            const lastFailName = lastToolMetadata?.name ?? 'unknown';
            const lastFailOk = lastToolMetadata?.success ?? true;
            const lastFailDesc = lastFailOk
              ? 'multiple tool operations are failing in this run'
              : `the last failing operation was \`${lastFailName}\``;
            const decision = await pauseForUser({
              kind: 'guidance',
              prompt: `Tool failures are nearing the kill limit (${consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES} consecutive); ${lastFailDesc}. How should I proceed? You can approve a specific fix command, give different instructions, or stop.`,
              options: ['Approve a fix command', 'Give different instructions', 'Stop'],
            });
            if (!decision) break; // aborted
            consecutiveFailures = 0;
            recentFailures.length = 0;
            let guidanceText: string;
            if (decision.decision === 'timeout') {
              guidanceText = 'The user did not respond within the time limit. Proceed with your best alternative, or stop if no safe path remains.';
            } else if (decision.decision === 'approve' && decision.command) {
              approvedCommands.add(normalizeCommand(decision.command));
              guidanceText = `User approved running: \`${decision.command}\`. Run this exact command now with run_terminal.${decision.message ? ' Note: ' + decision.message : ''}`;
            } else {
              guidanceText = decision.message?.trim()
                ? `User guidance: ${decision.message.trim()}`
                : 'User did not provide specific guidance. Re-evaluate your approach and proceed, or stop if blocked.';
            }
            // Reject this and all remaining tool calls in this turn, then push
            // the user's guidance as a user message so the next model turn sees it.
            for (let j = i; j < pendingToolCalls.length; j++) {
              const rejectTc = pendingToolCalls[j];
              const rejectName = rejectTc.function?.name ?? 'unknown_tool';
              send('tool_start', { name: rejectName, args: {} });
              send('tool_result', { name: rejectName, success: false, output: 'Paused for user guidance. See the user message that follows.', durationMs: 0 });
              requestCtx.push({
                role: 'tool',
                content: 'Paused for user guidance before running. See the user message that follows.',
                tool_call_id: rejectTc.id ?? rejectName,
                tool_name: rejectName,
                duration_ms: 0,
                created_at: Date.now(),
              });
            }
            requestCtx.push({ role: 'user', content: guidanceText, created_at: Date.now() });
            if (isolated) ctx.push({ role: 'user', content: guidanceText, created_at: Date.now() });
            await saveSessionToFile(sessionId, ctx, selectedPersona, model);
            break;
          }

          // 5 consecutive failures — kill the run for human review. The agent
          // is stopped (not asked for a final report): the human reviews the
          // transcript and resumes by sending a guiding message.
          if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
            const failList = recentFailures.slice(-MAX_CONSECUTIVE_FAILURES)
              .map((f, idx) => `${idx + 1}. ${f.name}: ${f.detail}`)
              .join('\n');
            const reviewMsg = `[SESSION KILLED — ${MAX_CONSECUTIVE_FAILURES} consecutive tool failures]\nThe agent was stopped for human review. Recent failures:\n${failList}\n\nReview the transcript, then guide the AI with a new message (the correct command, a different approach, or permission to install a missing dependency).`;
            for (let j = i; j < pendingToolCalls.length; j++) {
              const rejectTc = pendingToolCalls[j];
              const rejectName = rejectTc.function?.name ?? 'unknown_tool';
              send('tool_start', { name: rejectName, args: {} });
              send('tool_result', { name: rejectName, success: false, output: 'Session killed: 5 consecutive failures. Stopped for human review.', durationMs: 0 });
              requestCtx.push({ role: 'tool', content: 'Session killed: 5 consecutive failures. Stopped for human review.', tool_call_id: rejectTc.id ?? rejectName, tool_name: rejectName, duration_ms: 0, created_at: Date.now() });
            }
            requestCtx.push({ role: 'user', content: reviewMsg, created_at: Date.now() });
            if (isolated) ctx.push({ role: 'user', content: reviewMsg, created_at: Date.now() });
            send('killed', { reason: `${MAX_CONSECUTIVE_FAILURES} consecutive tool failures`, failures: recentFailures.slice(-MAX_CONSECUTIVE_FAILURES) });
            await saveSessionToFile(sessionId, ctx, selectedPersona, model);
            killed = true;
            terminalSent = true;
            break;
          }

          const toolName: ToolName = tc.function?.name;
          let args: Record<string, string> = {};
          try {
            args = typeof tc.function?.arguments === 'string'
              ? JSON.parse(tc.function.arguments)
              : (tc.function?.arguments ?? {});
          } catch {}

          send('tool_start', { name: toolName, args });
          const run = activeRuns.get(sessionId);
          if (run) {
            run.status = 'tool';
            run.statusText = `Running ${toolName}…`;
          }
          await saveProgress(assistantContent, true);

          // Side-effecting install: pause for explicit user approval before the
          // agent is allowed to mutate the environment (npm/pip/winget/etc.).
          // On approval, allowlist the exact command and tell the agent to
          // re-issue it — the agent runs the install itself via run_terminal.
          if (toolName === 'run_terminal' && typeof args.command === 'string'
              && isSideEffectingInstall(args.command)
              && !autopilotInstalls
              && !approvedCommands.has(normalizeCommand(args.command))) {
            const decision = await pauseForUser({
              kind: 'install',
              prompt: `The agent wants to run an install command that changes your environment. Approve?`,
              command: args.command,
            });
            if (!decision) break; // aborted
            if (decision.decision === 'timeout') {
              toolCallCount += 1;
              toolFailureCount += 1;
              consecutiveFailures += 1;
              anyToolFailedInTurn = true;
              recentFailures.push({ name: toolName, detail: `${args.command} → install not approved (user did not respond in time)` });
              send('tool_result', { name: toolName, success: false, output: 'Install not approved: the user did not respond in time. Do not run this command. Find an alternative or stop.', durationMs: 0 });
              requestCtx.push({ role: 'tool', content: 'Install not approved: the user did not respond in time. Do not run this command. Find an alternative or stop.', tool_call_id: tc.id ?? toolName, tool_name: toolName, duration_ms: 0, created_at: Date.now() });
              continue;
            }
            if (decision.decision === 'deny') {
              toolCallCount += 1;
              toolFailureCount += 1;
              consecutiveFailures += 1;
              anyToolFailedInTurn = true;
              const reason = decision.message ? ` Reason: ${decision.message}` : '';
              recentFailures.push({ name: toolName, detail: `${args.command} → install denied by user${reason}` });
              send('tool_result', { name: toolName, success: false, output: `User DENIED the install: ${args.command}.${reason} Do not run it. Find another approach.`, durationMs: 0 });
              requestCtx.push({ role: 'tool', content: `User DENIED the install: ${args.command}.${reason} Do not run it. Find another approach.`, tool_call_id: tc.id ?? toolName, tool_name: toolName, duration_ms: 0, created_at: Date.now() });
              continue;
            }
            // approve_all: approve this install AND flip the in-run autopilot
            // on so all future side-effecting installs in this run bypass the
            // approval gate (no more notifications). CRITICAL: we EXECUTE the
            // command here rather than asking the agent to re-issue it. Weak
            // local models (e.g. gemma4-uncensored) frequently ignore the
            // "re-issue this exact command" instruction and try something else,
            // which is how mkdir && npm install chains end up with the folder
            // never created and the agent looping on the next failure.
            if (decision.decision === 'approve_all') {
              autopilotInstalls = true;
              const approvedCmd = decision.command?.trim() || args.command;
              approvedCommands.add(normalizeCommand(approvedCmd));
              send('tool_start', { name: toolName, args });
              const execStartedAt = Date.now();
              const installResult = await runTool(toolName, { command: approvedCmd }, sessionId);
              const execDurationMs = Date.now() - execStartedAt;
              send('tool_result', {
                name: toolName,
                success: installResult.success,
                output: (installResult.success
                  ? `[AUTOPILOT — server executed this for you. All future installs in this run will also run automatically.]\n${installResult.output}`
                  : `[AUTOPILOT — server executed this for you but it failed. Treat the failure below as a real failure (counts toward the kill limit).]\n${installResult.output}`
                ).slice(0, 2000),
                durationMs: execDurationMs,
              });
              requestCtx.push({
                role: 'tool',
                content: installResult.success
                  ? `[AUTOPILOT — server executed this. All future installs in this run will also run automatically.]\n${installResult.output}`
                  : `[AUTOPILOT — server executed this but it failed.]\n${installResult.output}`,
                tool_call_id: tc.id ?? toolName,
                tool_name: toolName,
                duration_ms: execDurationMs,
                created_at: Date.now(),
              });
              // Successful autopilot run resets the streak; failure counts as
              // one (the model that proposed it shares blame).
              toolCallCount += 1;
              if (installResult.success) {
                consecutiveFailures = 0;
                recentFailures.length = 0;
              } else {
                toolFailureCount += 1;
                consecutiveFailures += 1;
                anyToolFailedInTurn = true;
                recentFailures.push({ name: toolName, detail: `${approvedCmd} → ${installResult.output.slice(0, 200)}` });
              }
              continue;
            }
            // approve (or guidance carrying a command): allowlist + tell agent
            const approvedCmd = decision.command?.trim() || args.command;
            approvedCommands.add(normalizeCommand(approvedCmd));
            // User approval is a success — reset the consecutive-failure streak.
            consecutiveFailures = 0;
            recentFailures.length = 0;
            const note = decision.message ? ` Note: ${decision.message}` : '';
            send('tool_result', { name: toolName, success: true, output: `User APPROVED running: \`${approvedCmd}\`. Re-issue this exact command now with run_terminal to perform the install.${note}`, durationMs: 0 });
            requestCtx.push({ role: 'tool', content: `User APPROVED running: \`${approvedCmd}\`. Re-issue this exact command now with run_terminal to perform the install.${note}`, tool_call_id: tc.id ?? toolName, tool_name: toolName, duration_ms: 0, created_at: Date.now() });
            continue;
          }

          const toolStartedAt = Date.now();
          // Tool-call budget: never let one tool call exceed TOOL_MAX_DURATION_MS.
          // Promise.race only abandons the await — runTool's own timeouts
          // (10-min terminal cap, 12–22s web caps) still terminate the real
          // work, so this is a safety net against a wedged tool promise.
          const toolTimeoutMs = TOOL_MAX_DURATION_MS;
          const result = await Promise.race([
            toolName === 'browse_url' && !isBrowseUrlAllowed(args.url, requestCtx.getMessages())
              ? Promise.resolve({
                success: false,
                output: `Browse denied: ${args.url} was not returned by a successful search and was not supplied by the user. Do not guess URLs; use only search-result links already in the context.`,
              })
              : runTool(toolName, args, sessionId),
            new Promise<{ success: boolean; output: string }>((resolve) => setTimeout(
              () => resolve({
                success: false,
                output: `[watchdog] Tool call exceeded its ${Math.round(TOOL_MAX_DURATION_MS / 60_000)}-minute budget and was abandoned. Treat this as a failure and continue with a different approach.`,
              }),
              toolTimeoutMs,
            )),
          ]);
          const toolDurationMs = Date.now() - toolStartedAt;
          toolCallCount += 1;
          if (!result.success) {
            toolFailureCount += 1;
            consecutiveFailures += 1;
            anyToolFailedInTurn = true;
            if (toolName === 'web_search') searchFailureCount += 1;
            const failArg = (args.command || args.query || args.url || args.filepath || '').slice(0, 120);
            recentFailures.push({ name: toolName, detail: `${failArg} → ${result.output.slice(0, 200)}` });
          } else {
            // Any success resets the consecutive-failure streak.
            consecutiveFailures = 0;
            recentFailures.length = 0;
            // Live context compaction: once a write_file/edit_file succeeds, the
            // file on disk is the source of truth, so drop the (often large)
            // content/old_str/new_str from the stored assistant tool call
            // immediately. pendingToolCalls entries are the same object
            // references held on the persisted assistant message, so mutating
            // arguments here shrinks what the next turn sends without touching
            // the file itself. Prevents context ballooning within one turn set.
            if (toolName === 'write_file' || toolName === 'edit_file') {
              try {
                const compacted = compactToolCall(tc as ToolCall);
                if (tc.function) tc.function.arguments = compacted.function.arguments;
              } catch {}
              if (typeof args.filepath === 'string') runChangedFiles.add(args.filepath);
            }
          }
          lastToolMetadata = { name: toolName, success: result.success, durationMs: toolDurationMs };

          let toolOutput = result.output;
          if (toolName === 'browse_url' && result.success && selectedPersona === 'researcher') {
            try {
              const extracted = await extractEvidenceFromPage(model, args.url, result.output, ac.signal);
              await appendToResearchLedger(sessionId, args.url, extracted);
              toolOutput = formatCompactedSummary(extracted);
            } catch (err) {
              console.error('[browse_url ledger processing failed]', err);
            }
          }

          send('tool_result', {
            name: toolName,
            success: result.success,
            output: toolOutput.slice(0, 2000),
            durationMs: toolDurationMs,
          });

          requestCtx.push({
            role: 'tool',
            content: truncateToolOutputForContext(toolOutput),
            tool_call_id: tc.id ?? toolName,
            tool_name: toolName,
            duration_ms: toolDurationMs,
            created_at: Date.now(),
          });
          const resumedRun = activeRuns.get(sessionId);
          if (resumedRun) {
            resumedRun.status = 'thinking';
            resumedRun.statusText = 'Planning next step…';
            resumedRun.partialContent = '';
          }
        }

        lastToolWasError = anyToolFailedInTurn;
        sendTokenUsage();
        await saveSessionToFile(sessionId, ctx, selectedPersona, model);
        if (killed) {
          break; // 5 consecutive failures — exit the run for human review
        }
        continue;
      }

      // ── End-of-run code review (one bounded round) ────────────────────────
      // The run is about to finish and files were changed: give a small
      // reviewer sidecar ONE look at them. Findings are handed back to the
      // coder as fix instructions; a clean (or unavailable) reviewer lets the
      // run complete normally. Hard-capped so it can never ping-pong.
      if (
        REVIEW_AT_END
        && backend === 'ollama'
        && !isolated
        && selectedPersona === 'coder'
        && !forceFinalResponse
        && runChangedFiles.size > 0
        && reviewRounds < MAX_REVIEW_ROUNDS
        && !ac.signal.aborted
      ) {
        const rolesNow = await getModelRoles();
        const reviewer = rolesNow.reviewer;
        if (reviewer) {
          reviewRounds += 1;
          const digestParts: string[] = [];
          let digestBytes = 0;
          for (const relPath of Array.from(runChangedFiles)) {
            try {
              const abs = path.resolve(WORKSPACE_DIR, relPath);
              if (!abs.startsWith(WORKSPACE_DIR)) continue;
              const content = await fs.readFile(abs, 'utf-8');
              const head = content.split(/\r?\n/).slice(0, 120).join('\n').slice(0, 8_000);
              if (!head.trim()) continue;
              const part = `--- ${relPath} ---\n${head}`;
              if (digestBytes + part.length > 24_000) break;
              digestParts.push(part);
              digestBytes += part.length;
            } catch {}
          }
          const fileDigest = digestParts.join('\n\n');
          if (fileDigest) {
            send('status', { message: `Running final code review with ${reviewer} on ${runChangedFiles.size} changed file(s)…` });
            const findings = await runOneShotModel(
              reviewer,
              [
                { role: 'system', content: 'You are a strict but pragmatic code reviewer for an in-progress build on a small local setup. Report ONLY defects that would break or seriously degrade this code: syntax errors, undefined references/missing imports, logic errors, broken wiring between files, obvious framework misuse. Do NOT report style, naming, performance micro-issues, or hypothetical improvements. Output at most one issue per line in the exact format "file:line: issue" using the paths given. If nothing serious is wrong, output exactly: CLEAN' },
                { role: 'user', content: `Review these files written by another model.\n\n${fileDigest}` },
              ],
              ac.signal,
              { numCtx: 16_384, numPredict: 512 },
            );
            const cleaned = (findings ?? '').trim();
            const hasFindings = Boolean(cleaned)
              && !/^clean\b/i.test(cleaned)
              && /:\s*\d+\s*:/.test(cleaned);
            if (hasFindings) {
              const trimmedFindings = cleaned.split('\n').filter(Boolean).slice(0, 12).join('\n');
              requestCtx.push({
                role: 'user',
                content: [
                  'A separate reviewer model flagged these concrete issues in the files you just wrote. Fix each one with targeted edit_file calls now (read_file first if you need exact source), then confirm what you fixed.',
                  '',
                  trimmedFindings,
                ].join('\n'),
                created_at: Date.now(),
              });
              if (isolated) ctx.push({ role: 'user', content: 'Reviewer findings pending — see prior message.', created_at: Date.now() });
              await saveSessionToFile(sessionId, ctx, selectedPersona, model);
              send('status', { message: 'Reviewer found issues; handing them to the coder to fix.' });
              continue; // coder gets one repair turn before done
            }
            if (cleaned) send('status', { message: 'Final code review passed with no blocking issues.' });
          }
        }
      }

      iterating = false;
      sendTokenUsage();
      send('done', {});
      terminalSent = true;
    }

    if (ac.signal.aborted) {
      // The watchdog's killRun already emitted 'killed'; don't double-send a
      // terminal event (the client treats the first as final).
      if (!terminalSentWatchdog) {
        send('stopped', {});
        terminalSent = true;
      }
    } else if (!terminalSent) {
      // Any non-abort exit that did not already emit 'done' (e.g. forced-final
      // -response retry exhaustion) must still send one, or the client stream
      // closes with no terminal event and the UI stays stuck as if running.
      sendTokenUsage();
      send('done', {});
      terminalSent = true;
    }
  } catch (err: any) {
    if (terminalSentWatchdog) {
      // Watchdog already emitted the terminal 'killed' event; suppress the
      // AbortError fallthrough so the client sees exactly one ending.
    } else if (err.name !== 'AbortError') {
      // Preserve any partial assistant text so the model can resume where it
      // left off when the user sends the next message (e.g. after restarting
      // a crashed Ollama). Without this the model has no memory of its
      // interrupted response and will hallucinate or repeat itself.
      if (pendingPartialContent && pendingPartialContent.trim()) {
        const truncated = pendingPartialContent.trim();
        const partialMsg: Message = {
          role: 'assistant',
          content: truncated,
          created_at: Date.now(),
        };
        const interruptionNote: Message = {
          role: 'system',
          content: 'Your previous response was interrupted by a connection failure. Continue from where you left off, or restart if the partial text is broken.',
          created_at: Date.now(),
        };
        if (isolated) {
          ctx.push(partialMsg);
          ctx.push(interruptionNote);
        } else {
          requestCtx.push(partialMsg);
          requestCtx.push(interruptionNote);
        }
      }

      const detail = err.cause?.message || err.message || 'Unknown Ollama connection error';
      await recordOllamaFailure(currentOllamaPhase, err);
      console.error(`[Ollama chat] ${model} failed:`, detail);
      send('error', { message: `Runtime request failed: ${detail}` });
    } else {
      send('stopped', {});
    }
  } finally {
    stopWatchdog();
    abortControllers.delete(sessionId);
    activeRuns.delete(sessionId);
    // If the run was aborted while paused for approval, release the deferred
    // so a stale /resume cannot resolve a dead promise.
    resumeDeferred.delete(sessionId);
    await saveSessionToFile(sessionId, ctx, selectedPersona, model);
    res.end();
  }
});

// NOTE: llama.cpp server is started on-demand when user selects iGPU backend,
// not on server boot. This saves ~10+ GB RAM when using Ollama backend.

export default router;

import fs from 'fs/promises';
import path from 'path';
import { exec, spawn } from 'child_process';
import { promisify } from 'util';
import { search, SafeSearchType } from 'duck-duck-scrape';
import { browseUrl, SearchResult, searchBingWithBrowser, searchWithBrowser, searchYahooWithBrowser } from './playwright.js';
import { WORKSPACE_DIR } from './tools.js';

const execAsync = promisify(exec);
const TERMINAL_TIMEOUT_MS = 10 * 60 * 1000; // 10 min — generous for npm installs; shorter than 30 so stalls surface faster
// Absolute path to the real shell. `spawn('cmd.exe', …)` / `shell: 'cmd.exe'`
// resolve the bare name against `cwd` (WORKSPACE_DIR) FIRST on Windows, so a
// stray `agent-workspace/cmd.exe` gets executed instead of the shell and the
// call dies with `spawn UNKNOWN`. An absolute ComSpec path removes that
// ambiguity for good.
const WIN_SHELL = process.env.ComSpec ?? 'C:\\Windows\\System32\\cmd.exe';
// Files whose basename impersonates a shell/interpreter must never be created
// inside the workspace: they hijack relative-name process resolution.
const RESERVED_EXECUTABLE_RE = /^(?:cmd|powershell|pwsh|bash|sh|node|npm|npx|python|py)\.(?:exe|bat|cmd|ps1|com)$/i;
// Respect public search services. This is pacing/backoff, not an attempt to
// bypass a provider's access controls.
const SEARCH_MIN_INTERVAL_MS = 8_000;
const SEARCH_JITTER_MS = 2_000;
const SEARCH_PROVIDER_TIMEOUT_MS = 12_000;
const FALLBACK_SWITCH_DELAY_MS = 3_000;
const BRAVE_SEARCH_API_URL = 'https://api.search.brave.com/res/v1/web/search';
const BRAVE_SEARCH_API_KEY = process.env.BRAVE_SEARCH_API_KEY?.trim();
let nextSearchAt = 0;
const OMITTED_HISTORY_CONTENT = new Set([
  '[omitted from history; workspace file is source of truth]',
  '[omitted from history]',
]);

export type ToolName = 'web_search' | 'browse_url' | 'write_file' | 'edit_file' | 'read_file' | 'run_terminal';

export interface ToolResult {
  success: boolean;
  output: string;
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function withTimeout<T>(operation: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      operation,
      new Promise<T>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs / 1000}s.`)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

async function waitForSearchSlot() {
  const now = Date.now();
  const waitMs = Math.max(0, nextSearchAt - now);
  if (waitMs > 0) await sleep(waitMs);
  nextSearchAt = Date.now() + SEARCH_MIN_INTERVAL_MS + Math.floor(Math.random() * SEARCH_JITTER_MS);
}

function unwrapBingRedirect(url: string): string {
  try {
    const parsed = new URL(url);
    if (!parsed.hostname.endsWith('bing.com') || !parsed.pathname.startsWith('/ck/')) return url;
    const encoded = parsed.searchParams.get('u') ?? '';
    // Bing commonly prefixes a base64 target with "a1".
    const payload = encoded.startsWith('a1') ? encoded.slice(2) : encoded;
    const decoded = Buffer.from(payload.replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString('utf8');
    return /^https?:\/\//i.test(decoded) ? decoded : url;
  } catch {
    return url;
  }
}

function keepRelevantResults(results: SearchResult[], query: string): SearchResult[] {
  const ignored = new Set(['with', 'from', 'that', 'this', 'what', 'when', 'where', 'hardware', 'review', 'data']);
  const terms = (query.toLowerCase().match(/[a-z0-9]{3,}/g) ?? []).filter((term) => !ignored.has(term));
  if (!terms.length) return results;
  return results.filter((result) => {
    const haystack = `${result.title} ${result.description} ${result.url}`.toLowerCase();
    return terms.some((term) => haystack.includes(term));
  });
}

function formatSearchResults(results: SearchResult[]): string {
  return results.slice(0, 8).map((result) =>
    `**${result.title}**\n${unwrapBingRedirect(result.url)}\n${result.description ?? ''}`
  ).join('\n\n');
}

function isBraveCreditExhaustion(body: string): boolean {
  const message = body.toLowerCase();
  const namesCreditProblem = /credit|quota/.test(message);
  const saysUsedUp = /exhaust|deplet|insufficient|used\s*up|limit\s*(?:has\s*)?(?:been\s*)?(?:reached|exceeded)|exceed(?:ed)?|reach(?:ed)?/.test(message);
  return namesCreditProblem && saysUsedUp;
}

async function searchBrave(query: string): Promise<SearchResult[]> {
  if (!BRAVE_SEARCH_API_KEY) {
    throw new Error('Brave Search is not configured. Set BRAVE_SEARCH_API_KEY.');
  }

  const url = `${BRAVE_SEARCH_API_URL}?q=${encodeURIComponent(query)}&count=8`;
  const response = await fetch(url, {
    headers: {
      Accept: 'application/json',
      'X-Subscription-Token': BRAVE_SEARCH_API_KEY,
    },
    signal: AbortSignal.timeout(SEARCH_PROVIDER_TIMEOUT_MS),
  });
  const body = await response.text();

  if (!response.ok) {
    const creditExhausted = isBraveCreditExhaustion(body);
    const detail = body.slice(0, 500);
    const error = new Error(`Brave Search returned HTTP ${response.status}: ${detail || response.statusText}`);
    (error as Error & { creditExhausted?: boolean }).creditExhausted = creditExhausted;
    throw error;
  }

  let payload: any;
  try {
    payload = JSON.parse(body);
  } catch {
    throw new Error('Brave Search returned invalid JSON.');
  }

  return (payload.web?.results ?? []).map((result: any) => ({
    title: result.title ?? '',
    url: result.url ?? '',
    description: result.description ?? '',
  })).filter((result: SearchResult) => result.title && result.url);
}

async function runDuckDuckGoSearch(query: string): Promise<ToolResult> {
  await waitForSearchSlot();
  let libraryError = '';
  try {
    const response = await withTimeout(
      search(query, { safeSearch: SafeSearchType.OFF }),
      SEARCH_PROVIDER_TIMEOUT_MS,
      'DuckDuckGo API request',
    );
    const results = (response.results ?? []).map((result: any) => ({
      title: result.title ?? '', url: result.url ?? '', description: result.description ?? '',
    })).filter((result) => result.title && result.url);
    const relevantResults = keepRelevantResults(results, query);
    if (relevantResults.length) return { success: true, output: formatSearchResults(relevantResults) };
    libraryError = 'DuckDuckGo returned no usable results.';
  } catch (error: any) {
    libraryError = error?.message ?? String(error);
  }

  // One browser retry can handle transient package/API failures. Do not retry
  // repeatedly: the caller receives a clear failure and ends the research run.
  let duckDuckGoBrowserError = '';
  try {
    const results = keepRelevantResults(await withTimeout(searchWithBrowser(query), SEARCH_PROVIDER_TIMEOUT_MS, 'DuckDuckGo browser fallback'), query);
    if (results.length) return { success: true, output: formatSearchResults(results) };
    throw new Error('DuckDuckGo browser search returned no relevant results.');
  } catch (browserError: any) {
    duckDuckGoBrowserError = browserError?.message ?? String(browserError);
  }

  // Keep a small, jittered pause before trying a distinct public provider.
  await sleep(FALLBACK_SWITCH_DELAY_MS + Math.floor(Math.random() * SEARCH_JITTER_MS));
  let bingErrorDetail = '';
  try {
    const results = keepRelevantResults(await withTimeout(searchBingWithBrowser(query), SEARCH_PROVIDER_TIMEOUT_MS, 'Bing browser fallback'), query);
    if (results.length) return { success: true, output: formatSearchResults(results) };
    throw new Error('Bing browser search returned no relevant results.');
  } catch (bingError: any) {
    bingErrorDetail = bingError?.message ?? String(bingError);
  }

  // Try Yahoo Search as a third browser fallback (fourth search layer overall)
  await sleep(FALLBACK_SWITCH_DELAY_MS + Math.floor(Math.random() * SEARCH_JITTER_MS));
  try {
    const results = keepRelevantResults(await withTimeout(searchYahooWithBrowser(query), SEARCH_PROVIDER_TIMEOUT_MS, 'Yahoo browser fallback'), query);
    if (results.length) return { success: true, output: formatSearchResults(results) };
    throw new Error('Yahoo browser search returned no relevant results.');
  } catch (yahooError: any) {
    const detail = yahooError?.message ?? String(yahooError);
    return {
      success: false,
      output: `Search unavailable. DuckDuckGo API: ${libraryError}. DuckDuckGo browser fallback: ${duckDuckGoBrowserError}. Bing browser fallback: ${bingErrorDetail}. Yahoo browser fallback: ${detail}. Do not guess URLs or repeat this search; explain the limitation and provide only verifiable partial findings.`,
    };
  }
}

async function runWebSearch(query: string): Promise<ToolResult> {
  await waitForSearchSlot();

  try {
    const results = keepRelevantResults(
      await searchBrave(query),
      query,
    );
    if (results.length) return { success: true, output: formatSearchResults(results) };
    return {
      success: false,
      output: 'Brave Search returned no usable results. Do not guess URLs or switch search providers.',
    };
  } catch (error: any) {
    if (!(error?.creditExhausted === true)) {
      return {
        success: false,
        output: `Brave Search failed: ${error?.message ?? String(error)} Do not switch search providers unless Brave explicitly reports exhausted credits.`,
      };
    }
  }

  await sleep(FALLBACK_SWITCH_DELAY_MS);
  return runDuckDuckGoSearch(query);
}

// Commands that start a server and never return on their own. Run through
// execAsync they block until the timeout, stranding the agent. We launch these
// detached instead and return immediately.
//
// Also catches `start /b <server>` — Windows background-launch that still
// keeps execAsync waiting because the spawned node process keeps the console
// group alive until it exits.
const LONG_RUNNING_SERVER = /(^|&&|&|\||;)\s*(start\s+\/[bB]\s+(node|npm|npx|nodemon|python|flask|uvicorn|gunicorn|php)\b|node\s+[^&|;]*\b(server|app|index|main|start)\b|npm\s+(start|run\s+(dev|start|serve|preview))|pnpm\s+(dev|start|run\s+(dev|start|serve))|yarn\s+(dev|start)|npx\s+(vite|next|http-server|serve|nodemon)|vite(\s|$)|next\s+(dev|start)|nodemon\b|http-server\b|serve\b|python\s+-m\s+http\.server|flask\s+run|uvicorn\b|gunicorn\b|php\s+-S)\b/i;

// Translate the bash-only tokens weaker models emit so a cmd.exe foreground
// segment does not fail on `sleep`/`head`/`tail`. `&&`, `curl`, and pipes work
// natively on modern Windows, so they are left untouched.
function normalizeForegroundBashisms(command: string): string {
  return command
    // `sleep 2` → PowerShell Start-Sleep (cmd has no sleep)
    .replace(/(^|&&|&|;|\|)\s*sleep\s+(\d+)\b/gi, (_m, sep, secs) => `${sep} powershell -NoProfile -Command "Start-Sleep -Seconds ${secs}"`)
    // Drop `| head [-n N]` / `| tail [-n N]` — output is truncated downstream anyway
    .replace(/\s*\|\s*head(\s+-n?\s*\d+)?/gi, '')
    .replace(/\s*\|\s*tail(\s+-n?\s*\d+)?/gi, '');
}

// Split a command on a lone `&` (bash "background") but not on `&&` (cmd's
// run-if-success) and not on redirect operators like `>&`, `2>&1`, or `&>`.
//
// The original regex `/(?<!&)&(?!&)/` matched the `&` inside `2>&1` because
// the lookbehind only excluded `&`, not `>` or digits. That caused the shell
// to receive `1` as the foreground command → "1 is not recognized" error.
//
// Fixed regex excludes `&` when:
//   - preceded by `&`  (part of `&&`)
//   - preceded by `>`  (part of `>&`)
//   - preceded by digit (part of `2>&`)
//   - followed by `&`  (part of `&&`)
//   - followed by `>`  (part of `&>`)
function splitBackgroundOperator(command: string): { background: string | null; foreground: string } {
  const match = command.match(/(?<![&>0-9])&(?![&>])/);
  if (!match || match.index === undefined) return { background: null, foreground: command };
  return {
    background: command.slice(0, match.index).trim(),
    foreground: command.slice(match.index + 1).trim(),
  };
}

// Launch a detached child that keeps running after this tool call returns.
function startDetached(command: string, cwd: string = WORKSPACE_DIR): number | undefined {
  const child = spawn(WIN_SHELL, ['/c', command], {
    cwd,
    detached: true,
    stdio: 'ignore',
    windowsHide: true,
  });
  child.unref();
  return child.pid;
}

// Session-scoped registry of files this agent created with write_file. Overwrite
// is allowed only for paths in this set (the agent's own output), never for
// pre-existing or user-authored files.
const sessionCreatedFiles = new Map<string, Set<string>>();
function registerCreatedFile(sessionId: string | undefined, absPath: string) {
  const key = sessionId ?? 'nosession';
  let set = sessionCreatedFiles.get(key);
  if (!set) { set = new Set(); sessionCreatedFiles.set(key, set); }
  set.add(absPath);
}
function wasCreatedThisSession(sessionId: string | undefined, absPath: string): boolean {
  return sessionCreatedFiles.get(sessionId ?? 'nosession')?.has(absPath) ?? false;
}

// Session-scoped registry of failed tool calls. A model that re-issues an
// identical call that already failed is looping; we refuse to execute it again
// and hand back an escalation so the consecutive-failure ladder can act.
const sessionFailedCalls = new Map<string, Set<string>>();
function failedCallKey(name: string, args: Record<string, unknown>): string {
  return `${name}:${JSON.stringify(args)}`;
}
function recordFailedCall(sessionId: string | undefined, key: string) {
  const sk = sessionId ?? 'nosession';
  let set = sessionFailedCalls.get(sk);
  if (!set) { set = new Set(); sessionFailedCalls.set(sk, set); }
  set.add(key);
}
function hasFailedBefore(sessionId: string | undefined, key: string): boolean {
  return sessionFailedCalls.get(sessionId ?? 'nosession')?.has(key) ?? false;
}

// Called on session reset so a fresh run does not inherit stale registries.
export function clearSessionToolState(sessionId: string | undefined) {
  const key = sessionId ?? 'nosession';
  sessionCreatedFiles.delete(key);
  sessionFailedCalls.delete(key);
  for (const mapKey of Array.from(editRecoveryStates.keys())) {
    if (mapKey.startsWith(`${key}:`)) editRecoveryStates.delete(mapKey);
  }
}

const PROHIBITED_INSTALL_RE = /(^|&&|;|\|\|)\s*(npm\s+(?:install|i|ci|init|create)|npx\b|pnpm\s+(?:add|install|i|init|create)|yarn\s+(?:add|install|create|init)|pip3?\s+install|python\s+-m\s+pip\s+install|pipx\s+install|cargo\s+(?:add|init|new)|composer\s+(?:require|install)|dotnet\s+(?:add|new)|gem\s+install|winget\s+install|choco\s+install|scoop\s+install)\b/i;
const TERMINAL_FILE_MUTATION_PATTERNS = [
  /(^|(?:&&|\|\||[&;|])\s*)(?:del|erase|copy|xcopy|robocopy|move|ren|rename|replace|rm|mv|cp|truncate|touch)\b/i,
  /\b(?:Set-Content|Add-Content|Clear-Content|Out-File|Remove-Item|Move-Item|Copy-Item|Rename-Item)\b/i,
  /\b(?:sed\s+-i|perl\s+-pi)\b/i,
  /\bgit\s+(?:checkout|restore|reset|clean)\b/i,
  /\b(?:cmd(?:\.exe)?\s+\/[ck]|bash\s+-c|sh\s+-c)\s+["']?\s*(?:del|erase|copy|xcopy|robocopy|move|ren|rename|replace|rm|mv|cp|truncate|touch)\b/i,
  /\bnode(?:\.exe)?\s+(?:-e|--eval)\b[\s\S]*\b(?:writeFile(?:Sync)?|appendFile(?:Sync)?|copyFile(?:Sync)?|cp(?:Sync)?|createWriteStream|open(?:Sync)?|unlink(?:Sync)?|rename(?:Sync)?|truncate(?:Sync)?|rm(?:Sync)?)\b/i,
  /\bpython(?:\.exe)?\s+-c\b[\s\S]*(?:\bopen\s*\([^)]*,\s*['"][wax+]|\.\s*open\s*\(\s*['"][wax+]|\b(?:write_text|write_bytes|unlink|remove|rename|replace|rmtree|copyfile|move)\b)/i,
  /(^|[^0-9])\d*>(?![>&])/,
];

function terminalCommandMutatesFiles(command: string): boolean {
  return TERMINAL_FILE_MUTATION_PATTERNS.some((pattern) => pattern.test(command));
}

// ── edit_file miss ladder ───────────────────────────────────────────────────
// Weak models death-spiral on exact-match edit failures: retry blind, flood
// context with full-file reads, then emit malformed tool JSON. The ladder
// gives progressively stronger recovery evidence per file:
//   miss 1 → plain error + chunked-read guidance
//   miss 2 → fuzzy-located exact source window
//   miss 3+ → edit_file disabled until a bounded read
export const READ_FILE_CHUNK_LINES = 100;
interface EditRecoveryState {
  misses: number;
  locked: boolean;
  fullOverwriteUnsafe: boolean;
}
const editRecoveryStates = new Map<string, EditRecoveryState>();

function editMissKey(sessionId: string | undefined, target: string): string {
  return `${sessionId ?? 'nosession'}:${target}`;
}

async function locateSimilarSnippet(sourceLines: string[], oldStr: string): Promise<{ start: number; end: number } | null> {
  const oldLines = oldStr.replace(/\r\n/g, '\n').split('\n');
  const anchors = oldLines
    .map((line, index) => ({ text: line.trim(), index }))
    .filter(({ text }) => text.length >= 6 && !/^[{}()\[\];,<>/\s]+$/.test(text))
    .sort((a, b) => b.text.length - a.text.length);
  if (!anchors.length) return null;

  let bestIdx = -1;
  let bestAnchorIdx = -1;
  let bestScore = 0;
  for (const anchor of anchors) {
    const needle = anchor.text.toLowerCase();
    for (let i = 0; i < sourceLines.length; i++) {
      const hay = sourceLines[i].toLowerCase();
      let score = 0;
      if (hay.includes(needle)) score = needle.length;
      else {
        const overlapWindow = Math.min(needle.length, 24);
        for (let len = overlapWindow; len >= 8; len -= 4) {
          if (hay.includes(needle.slice(0, len))) { score = len; break; }
        }
      }
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
        bestAnchorIdx = anchor.index;
      }
    }
  }
  if (bestIdx < 0 || bestScore < 8) return null;
  const windowSize = oldLines.length;
  const maxStart = Math.max(0, sourceLines.length - windowSize);
  const start = Math.min(maxStart, Math.max(0, bestIdx - bestAnchorIdx));
  return {
    start,
    end: Math.min(sourceLines.length - 1, start + windowSize - 1),
  };
}

function findNewlineInsensitiveMatches(source: string, oldStr: string): Array<{ start: number; end: number }> {
  const normalizedOld = oldStr.replace(/\r\n/g, '\n');
  const pattern = normalizedOld
    .split('\n')
    .map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('\\r?\\n');
  const matches: Array<{ start: number; end: number }> = [];
  const regex = new RegExp(pattern, 'g');
  let match: RegExpExecArray | null;
  while ((match = regex.exec(source)) !== null) {
    matches.push({ start: match.index, end: match.index + match[0].length });
    if (match[0].length === 0) regex.lastIndex++;
  }
  return matches;
}

function replacementWithMatchedLineEndings(source: string, start: number, end: number, replacement: string): string {
  const matchedEol = source.slice(start, end).match(/\r\n|\n/)?.[0]
    ?? source.match(/\r\n|\n/)?.[0]
    ?? '\n';
  return replacement.replace(/\r\n|\n/g, matchedEol);
}

function renderNumberedLines(lines: string[], start: number, end: number): string {
  const width = String(end + 1).length;
  const out: string[] = [];
  for (let i = start; i <= end; i++) {
    out.push(`${String(i + 1).padStart(width)}| ${lines[i]}`);
  }
  return out.join('\n');
}

// Deterministic-failure tools: re-issuing the exact same failing call is a
// loop, not a transient retry. web_search / browse_url can fail transiently
// (network) and are excluded so a legitimate retry is not blocked.
const DEDUP_FAILURE_TOOLS = new Set<ToolName>(['write_file', 'edit_file', 'run_terminal']);

export async function runTool(name: ToolName, args: Record<string, string>, sessionId?: string): Promise<ToolResult> {
  const dedup = DEDUP_FAILURE_TOOLS.has(name);
  const callKey = dedup ? failedCallKey(name, args) : '';
  if (dedup && hasFailedBefore(sessionId, callKey)) {
    return {
      success: false,
      output: `This exact ${name} call already failed earlier in this run and was not run again. Repeating an identical failing call is a loop. Change the arguments or the approach (read the file in a small chunk first, fix the specific error, or move to the next step).`,
    };
  }
  const result = await runToolInner(name, args, sessionId);
  if (dedup && !result.success) recordFailedCall(sessionId, callKey);
  return result;
}

async function runToolInner(name: ToolName, args: Record<string, string>, sessionId?: string): Promise<ToolResult> {
  try {
    switch (name) {
      case 'web_search':
        return runWebSearch(args.query);

      case 'browse_url': {
        try {
          // browseUrl already internally handles safe timeout in 22s and ensures page closure
          const result = await browseUrl(args.url);
          if (result.status >= 400 || result.status === 0) {
            return { success: false, output: `Page request failed (HTTP ${result.status || 'unknown'}) for ${result.url}. Do not treat this page as a source or guess a replacement URL.` };
          }
          return { success: true, output: result.text || '(empty page)' };
        } catch (err: any) {
          return { success: false, output: `Page request timed out or failed: ${err.message || String(err)}. Do not treat this page as a source or guess a replacement URL.` };
        }
      }

      case 'write_file': {
        // A truncated/oversized tool-call arguments JSON can leave filepath or
        // content undefined. Fail with a clear message instead of letting
        // path.resolve throw a cryptic "paths[1] undefined" error.
        if (typeof args.filepath !== 'string' || !args.filepath.trim()) {
          return { success: false, output: 'write_file received "content" but no "filepath" — the call was cut off at the output-token limit before the filepath was emitted. Re-issue the call with "filepath" FIRST (before "content"). For very large files, split them into multiple smaller write_file calls.' };
        }
        if (typeof args.content !== 'string') {
          return { success: false, output: 'write_file requires a string "content". The previous call was likely truncated; put "filepath" first, then "content", or split the file into smaller write_file calls.' };
        }
        if (OMITTED_HISTORY_CONTENT.has(args.content.trim())) {
          return {
            success: false,
            output: 'write_file rejected a compacted-history placeholder. This is not file content. Re-read the current file and use edit_file for a targeted exact replacement.',
          };
        }
        if (RESERVED_EXECUTABLE_RE.test(path.basename(args.filepath))) {
          return { success: false, output: `write_file denied: "${path.basename(args.filepath)}" impersonates a shell/interpreter and would hijack command resolution. Choose a different filename.` };
        }
        const target = path.resolve(WORKSPACE_DIR, args.filepath);
        if (!target.startsWith(WORKSPACE_DIR)) return { success: false, output: 'Path traversal denied.' };
        // write_file is create-only. Pre-existing/user files AND files this run
        // already created must go through edit_file (which proves exact old
        // text). Allowing re-write of the agent's own file did NOT stop weak
        // models looping — they rewrote it every turn (the gemma4 loop) and
        // never advanced to the next file — so a full rewrite of a
        // session-created file is now blocked. Full rewrites happen only in the
        // final review pass, applied as targeted edit_file changes.
        try {
          const existing = await fs.readFile(target, 'utf-8');
          if (existing === args.content) {
            return { success: false, output: `write_file no-op: "${args.filepath}" already has this exact content. Do not rewrite it identically — you are repeating yourself. Move to the next file or the next step in your plan.` };
          }
          if (wasCreatedThisSession(sessionId, target)) {
            return {
              success: false,
              output: `write_file blocked: "${args.filepath}" was already created this run. Full-file rewrites mid-build are not allowed — they cause the rewrite loop and risk output-token truncation. To change it, use edit_file with ONE exact old_str/new_str. If it is already complete, STOP rewriting it and move to the NEXT file or step in your plan.`,
            };
          }
          return {
            success: false,
            output: `write_file blocked: "${args.filepath}" already exists and was preserved. write_file only creates new files. Read the relevant 1-${READ_FILE_CHUNK_LINES}-line chunk, then use edit_file with one exact function/component-sized old_str and new_str.`,
          };
        } catch (error: any) {
          if (error?.code !== 'ENOENT') {
            return { success: false, output: `Unable to inspect existing path "${args.filepath}": ${error?.message ?? String(error)}` };
          }
        }
        await fs.mkdir(path.dirname(target), { recursive: true });
        try {
          // `wx` closes the read/create race: another run cannot create the
          // path between the existence check and this write and then be erased.
          await fs.writeFile(target, args.content, { encoding: 'utf-8', flag: 'wx' });
        } catch (error: any) {
          if (error?.code === 'EEXIST') {
            if (wasCreatedThisSession(sessionId, target)) {
              return {
                success: false,
                output: `write_file blocked: "${args.filepath}" was already created this run. Full-file rewrites mid-build are not allowed — use edit_file for a targeted change, or move to the NEXT file or step in your plan.`,
              };
            }
            return {
              success: false,
              output: `write_file blocked: "${args.filepath}" was created by another operation and was preserved. Re-read it and use edit_file for any change.`,
            };
          }
          throw error;
        }
        registerCreatedFile(sessionId, target);
        // Report size so the model (and the next turn) can tell a real source
        // file from a tiny placeholder / truncated write.
        const byteCount = Buffer.byteLength(args.content, 'utf-8');
        const lineCount = args.content.length === 0 ? 0 : args.content.split(/\r?\n/).length;
        return { success: true, output: `Written: ${args.filepath} (${byteCount} bytes, ${lineCount} lines)` };
      }

      case 'edit_file': {
        // Targeted edit: replace one exact, unique occurrence. Avoids rewriting
        // whole files, which is slow and risks output-token truncation.
        if (typeof args.filepath !== 'string' || !args.filepath.trim()) {
          return { success: false, output: 'edit_file requires a non-empty string "filepath".' };
        }
        if (typeof args.old_str !== 'string' || args.old_str.length === 0) {
          return { success: false, output: 'edit_file requires a non-empty string "old_str" (the exact text to replace).' };
        }
        if (typeof args.new_str !== 'string') {
          return { success: false, output: 'edit_file requires a string "new_str" (use an empty string to delete the matched text).' };
        }
        const target = path.resolve(WORKSPACE_DIR, args.filepath);
        if (!target.startsWith(WORKSPACE_DIR)) return { success: false, output: 'Path traversal denied.' };

        const missKey = editMissKey(sessionId, target);
        const recovery = editRecoveryStates.get(missKey);
        if (recovery?.locked) {
          return {
            success: false,
            output: `edit_file recovery lock active for "${args.filepath}" after ${recovery.misses} failed match attempts. Read an explicit chunk from this same file with read_file limit=1-${READ_FILE_CHUNK_LINES}, then retry a smaller function/component-level exact edit. Full-file rewrite is not the fallback.`,
          };
        }

        let source: string;
        try {
          source = await fs.readFile(target, 'utf-8');
        } catch {
          return { success: false, output: `File not found: ${args.filepath}. Use write_file to create a new file.` };
        }
        const matches = findNewlineInsensitiveMatches(source, args.old_str);
        if (matches.length === 0) {
          const state = recovery ?? { misses: 0, locked: false, fullOverwriteUnsafe: false };
          state.misses++;
          editRecoveryStates.set(missKey, state);

          if (state.misses === 2) {
            const sourceLines = source.split(/\r?\n/);
            const region = await locateSimilarSnippet(sourceLines, args.old_str);
            const hint = region
              ? `Closest source window in "${args.filepath}", lines ${region.start + 1}-${region.end + 1}:\n--- BEGIN EXACT UNNUMBERED SOURCE ---\n${sourceLines.slice(region.start, region.end + 1).join('\n')}\n--- END EXACT UNNUMBERED SOURCE ---\nCopy exact source text directly. Next edit attempt remains allowed.`
              : `No plausible old_str anchor exists in "${args.filepath}". Use read_file in chunks of at most ${READ_FILE_CHUNK_LINES} lines to locate exact source. Next edit attempt remains allowed.`;
            return {
              success: false,
              output: `old_str not found (attempt ${state.misses}). ${hint}`,
            };
          }

          if (state.misses >= 3) {
            state.locked = true;
            state.fullOverwriteUnsafe = true;
            return {
              success: false,
              output: `old_str not found (attempt ${state.misses}). edit_file recovery lock now active for "${args.filepath}". Full-file rewrite is blocked for existing files over ${READ_FILE_CHUNK_LINES} lines. Read an explicit chunk from this same file with read_file limit=1-${READ_FILE_CHUNK_LINES}, then retry a smaller function/component-level exact edit.`,
            };
          }

          return {
            success: false,
            output: `old_str was not found in "${args.filepath}" (attempt 1). Do NOT re-read the whole file. Use read_file with limit=${READ_FILE_CHUNK_LINES} and offset to scan in chunks, then retry edit_file copying an exact snippet including whitespace.`,
          };
        }
        if (matches.length > 1) {
          return { success: false, output: `old_str matches ${matches.length} places. Add surrounding context so it matches exactly one location.` };
        }
        const match = matches[0];
        const replacement = replacementWithMatchedLineEndings(source, match.start, match.end, args.new_str);
        if (replacement === source.slice(match.start, match.end)) {
          return { success: false, output: `edit_file no-op: replacement matches existing text — the file would not change. Do not repeat the same edit. Move to the next step.` };
        }
        await fs.writeFile(target, source.slice(0, match.start) + replacement + source.slice(match.end), 'utf-8');
        editRecoveryStates.delete(missKey);
        return { success: true, output: `Edited: ${args.filepath}` };
      }

      case 'read_file': {
        if (typeof args.filepath !== 'string' || !args.filepath.trim()) {
          return { success: false, output: 'read_file requires a non-empty string "filepath".' };
        }
        const target = path.resolve(WORKSPACE_DIR, args.filepath);
        if (!target.startsWith(WORKSPACE_DIR)) return { success: false, output: 'Path traversal denied.' };

        // Chunked paging: full-file dumps (10KB+) bloat context and measurably
        // degrade weak models' tool-JSON quality on the next turn. Default
        // window is 100 lines; offset pages through larger files. Full-file
        // reads stay available via limit=0 for genuinely small files.
        const fileContent = await fs.readFile(target, 'utf-8');
        const totalLines = fileContent.split(/\r?\n/).length;
        const rawOffset = Number.parseInt(String(args.offset ?? '1'), 10);
        const offset = Number.isFinite(rawOffset) && rawOffset > 0 ? rawOffset : 1;
        if (offset > totalLines) {
          return {
            success: false,
            output: `read_file offset ${offset} is past end of "${args.filepath}" (${totalLines} lines). Choose an offset from 1-${totalLines}.`,
          };
        }
        let limit = READ_FILE_CHUNK_LINES;
        let explicitUnlockingLimit = false;
        if (args.limit !== undefined) {
          const parsedLimit = Number.parseInt(String(args.limit), 10);
          if (Number.isFinite(parsedLimit) && parsedLimit >= 0) {
            limit = parsedLimit;
            explicitUnlockingLimit = parsedLimit >= 1 && parsedLimit <= READ_FILE_CHUNK_LINES;
          }
        }
        if (explicitUnlockingLimit) {
          const recovery = editRecoveryStates.get(editMissKey(sessionId, target));
          if (recovery?.locked) {
            recovery.locked = false;
            recovery.misses = 0;
          }
        }

        if (limit === 0 || (offset === 1 && limit >= totalLines)) {
          // Explicit full read, or a single window that already covers the file.
          return {
            success: true,
            output: `${fileContent}\n\n[${totalLines} lines total${limit > 0 ? ' — full file' : ''}]`,
          };
        }
        const lines = fileContent.split(/\r?\n/);
        const startIdx = offset - 1;
        const endIdx = Math.min(lines.length - 1, startIdx + limit - 1);
        const numbered = renderNumberedLines(lines, startIdx, endIdx);
        const hasMore = endIdx < lines.length - 1;
        const footer = hasMore
          ? `\n\n[Showing lines ${offset}-${endIdx + 1} of ${lines.length}. Continue with offset=${endIdx + 2}, limit=${READ_FILE_CHUNK_LINES}, or jump to a region.]`
          : `[Showing lines ${offset}-${lines.length} of ${lines.length} — end of file]`;
        return { success: true, output: `${numbered}${footer}` };
      }

      case 'run_terminal': {
        // ── Windows command normalisation ────────────────────────────────────
        // Models trained on Linux/macOS often emit bash syntax that cmd.exe
        // rejects. Translate the most common patterns before execution so that
        // even weaker models can create files and directories reliably.
        if (typeof args.command !== 'string' || !args.command.trim()) {
          return { success: false, output: 'run_terminal requires a non-empty string "command".' };
        }
        let command: string = args.command.trim();

        // Leading `cd <dir> && rest` / `cd <dir>; rest`: cmd.exe only keeps the
        // changed directory within the same shell, and this executor spawns a
        // fresh shell per call. Instead of passing the `cd` (and the `&&` that
        // triggered `spawn UNKNOWN`), run the remainder with cwd = that subdir.
        let effectiveCwd = WORKSPACE_DIR;
        const cdMatch = command.match(/^cd\s+(?:\/d\s+)?(["']?)([^"'&;|]+)\1\s*(?:&&|;)\s*([\s\S]+)$/i);
        if (cdMatch) {
          const requested = path.resolve(WORKSPACE_DIR, cdMatch[2].trim());
          if (!requested.startsWith(WORKSPACE_DIR)) {
            return { success: false, output: `run_terminal denied: 'cd ${cdMatch[2].trim()}' leaves the workspace.` };
          }
          try { await fs.mkdir(requested, { recursive: true }); } catch {}
          effectiveCwd = requested;
          command = cdMatch[3].trim();
        }

        // File mutations must use write_file/edit_file so their preservation
        // guarantees cannot be bypassed with common shell redirection, copy,
        // move, delete, PowerShell content cmdlets, or inline interpreter code.
        if (terminalCommandMutatesFiles(command)) {
          return {
            success: false,
            output: 'run_terminal blocked a file-mutation command. Use write_file only to create a new file, or read_file plus edit_file to change one exact block in an existing file.',
          };
        }

        // Prohibit package installation / init commands: agent's role is code generation only
        if (PROHIBITED_INSTALL_RE.test(command)) {
          // If command includes mkdir, still perform directory creation safely
          const mkdirMatches = Array.from(command.matchAll(/mkdir\s+(?:-p\s+)?([^\s&;]+)/gi));
          for (const m of mkdirMatches) {
            const dirTarget = path.resolve(WORKSPACE_DIR, m[1].trim().replace(/['"]/g, ''));
            if (dirTarget.startsWith(WORKSPACE_DIR)) {
              try { await fs.mkdir(dirTarget, { recursive: true }); } catch {}
            }
          }
          return {
            success: true,
            output: `Package installs and init commands (${command}) are prohibited for the agent. Any requested folders have been created. Directly write all configuration files (package.json, tailwind.config.js, vite.config.js, etc.) and application code using write_file. Provide install and run instructions for the user in your completion pass.`,
          };
        }

        // 1. `mkdir -p <path>` → PowerShell New-Item which is idempotent
        command = command.replace(
          /^mkdir\s+-p\s+(.+)$/i,
          (_, p) => `powershell -Command "New-Item -ItemType Directory -Force -Path '${p.trim()}' | Out-Null"`,
        );

        // 2. `mkdir a && mkdir a/b` (or `mkdir a\b`) — convert each mkdir
        //    segment into an idempotent PowerShell call, then chain with `;`
        if (/^\s*mkdir\b/.test(command) && !/powershell/i.test(command)) {
          // Split on && or ; and translate every `mkdir foo` fragment
          const parts = command.split(/\s*(?:&&|;)\s*/);
          const translated = parts.map((part) => {
            const m = part.trim().match(/^mkdir\s+(.+)$/i);
            if (!m) return part;
            return `powershell -Command "New-Item -ItemType Directory -Force -Path '${m[1].trim()}' | Out-Null"`;
          });
          command = translated.join(' && ');
        }

        // 3. Background operator handling. A lone `&` (bash "run in background")
        //    is a sequential separator in cmd.exe, so `node server.js & curl…`
        //    runs the server in the FOREGROUND and blocks until the 30-minute
        //    timeout. Split it off and run the left side detached instead.
        const { background, foreground } = splitBackgroundOperator(command);
        const detachedPids: number[] = [];
        if (background) {
          const pid = startDetached(background, effectiveCwd);
          if (pid !== undefined) detachedPids.push(pid);
          command = foreground;
        }

        // 4. A foreground server start with no `&` still blocks forever. Detect
        //    known server starters and launch them detached so the agent can
        //    probe them with a separate command instead of hanging.
        if (command && !background && LONG_RUNNING_SERVER.test(command)) {
          const pid = startDetached(command, effectiveCwd);
          const pidNote = pid !== undefined ? ` (PID ${pid})` : '';
          return {
            success: true,
            output: `Started as a background process${pidNote} because this command runs a long-lived server that never returns. It is now running in agent-workspace. Do NOT start it in the foreground. Wait briefly, then use a SEPARATE run_terminal call to probe it (e.g. curl http://localhost:3000/...).`,
          };
        }

        // 5. Translate remaining bash-only tokens (sleep/head/tail) for cmd.exe.
        command = normalizeForegroundBashisms(command).trim();

        if (!command) {
          const pidList = detachedPids.length ? ` Background PID(s): ${detachedPids.join(', ')}.` : '';
          return { success: true, output: `Started background process.${pidList} Probe it with a separate run_terminal call.` };
        }

        // ── Execute ─────────────────────────────────────────────────────────
        // The 30-minute timeout is intentionally generous: large project scaffolds
        // and installs on a slow, GPU-less PC can legitimately take many minutes.
        try {
          const { stdout, stderr } = await execAsync(command, { cwd: effectiveCwd, timeout: TERMINAL_TIMEOUT_MS, shell: WIN_SHELL });
          const output = [stdout, stderr].filter(Boolean).join('\n').trim();
          const pidNote = detachedPids.length ? `\n(Background PID(s) started: ${detachedPids.join(', ')}.)` : '';
          return { success: true, output: (output || '(no output)') + pidNote };
        } catch (execErr: any) {
          const msg: string = execErr?.message ?? String(execErr);
          // "already exists" / EEXIST — directory is present, treat as success
          // so the model doesn't spiral into a retry loop.
          const alreadyExists = /already exists|subdirectory or file|eexist/i.test(msg);
          if (alreadyExists) {
            return { success: true, output: '(already existed — no action needed, continue with next step)' };
          }

          // Write failure details to session failure log in agent-workspace/.sessions/
          const cleanSessionId = (sessionId || 'terminal').replace(/[^a-zA-Z0-9_-]/g, '');
          const failureLogPath = path.join(WORKSPACE_DIR, '.sessions', `${cleanSessionId}-failures.log`);
          try {
            await fs.mkdir(path.join(WORKSPACE_DIR, '.sessions'), { recursive: true });
            const logEntry = `[${new Date().toISOString()}] Command: ${command}\nError:\n${msg}\n${'-'.repeat(50)}\n`;
            await fs.appendFile(failureLogPath, logEntry, 'utf-8');
          } catch {}

          const sanitizedOutput = msg.replace(/C:\\Users\\[^\\]+\\AppData\\Local\\npm-cache\\_logs\\[^\s]+/gi, `.sessions/${cleanSessionId}-failures.log`);
          return {
            success: false,
            output: `${sanitizedOutput}\n(Log written to session folder: .sessions/${cleanSessionId}-failures.log)`,
          };
        }
      }


      default:
        return { success: false, output: `Unknown tool: ${name}` };
    }
  } catch (err: any) {
    return { success: false, output: err?.message ?? String(err) };
  }
}

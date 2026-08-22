import fs from 'fs/promises';
import path from 'path';
import { exec, spawn } from 'child_process';
import { promisify } from 'util';
import { search, SafeSearchType } from 'duck-duck-scrape';
import { browseUrl, SearchResult, searchBingWithBrowser, searchWithBrowser, searchYahooWithBrowser } from './playwright.js';
import { WORKSPACE_DIR } from './tools.js';

const execAsync = promisify(exec);
const TERMINAL_TIMEOUT_MS = 10 * 60 * 1000; // 10 min — generous for npm installs; shorter than 30 so stalls surface faster
// Respect public search services. This is pacing/backoff, not an attempt to
// bypass a provider's access controls.
const SEARCH_MIN_INTERVAL_MS = 8_000;
const SEARCH_JITTER_MS = 2_000;
const SEARCH_PROVIDER_TIMEOUT_MS = 12_000;
const FALLBACK_SWITCH_DELAY_MS = 3_000;
const BRAVE_SEARCH_API_URL = 'https://api.search.brave.com/res/v1/web/search';
const BRAVE_SEARCH_API_KEY = process.env.BRAVE_SEARCH_API_KEY?.trim();
let nextSearchAt = 0;

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
function startDetached(command: string): number | undefined {
  const child = spawn('cmd.exe', ['/c', command], {
    cwd: WORKSPACE_DIR,
    detached: true,
    stdio: 'ignore',
    windowsHide: true,
  });
  child.unref();
  return child.pid;
}

const PROHIBITED_INSTALL_RE = /(^|&&|;|\|\|)\s*(npm\s+(?:install|i|ci|init|create)|npx\b|pnpm\s+(?:add|install|i|init|create)|yarn\s+(?:add|install|create|init)|pip3?\s+install|python\s+-m\s+pip\s+install|pipx\s+install|cargo\s+(?:add|init|new)|composer\s+(?:require|install)|dotnet\s+(?:add|new)|gem\s+install|winget\s+install|choco\s+install|scoop\s+install)\b/i;

export async function runTool(name: ToolName, args: Record<string, string>, sessionId?: string): Promise<ToolResult> {
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
        const target = path.resolve(WORKSPACE_DIR, args.filepath);
        if (!target.startsWith(WORKSPACE_DIR)) return { success: false, output: 'Path traversal denied.' };
        // No-op guard: if the file already exists with byte-identical content,
        // refuse the write. A weak model that lost its plan will otherwise
        // rewrite the same file hundreds of times (each returning success and
        // reinforcing the loop). Returning success:false engages the run's
        // consecutive-failure pause (3) and kill (5) so the human gets pulled in.
        try {
          const existing = await fs.readFile(target, 'utf-8');
          if (existing === args.content) {
            return { success: false, output: `write_file no-op: "${args.filepath}" already has this exact content. Do not rewrite it identically — you are repeating yourself. Move to the next file or the next step in your plan.` };
          }
        } catch {
          // File doesn't exist yet — proceed to create it.
        }
        await fs.mkdir(path.dirname(target), { recursive: true });
        await fs.writeFile(target, args.content, 'utf-8');
        return { success: true, output: `Written: ${args.filepath}` };
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
        let source: string;
        try {
          source = await fs.readFile(target, 'utf-8');
        } catch {
          return { success: false, output: `File not found: ${args.filepath}. Use write_file to create a new file.` };
        }
        const occurrences = source.split(args.old_str).length - 1;
        if (occurrences === 0) {
          return { success: false, output: 'old_str was not found in the file. Read the file and copy an exact snippet (including whitespace) to replace.' };
        }
        if (occurrences > 1) {
          return { success: false, output: `old_str matches ${occurrences} places. Add surrounding context so it matches exactly one location.` };
        }
        if (args.new_str === args.old_str) {
          return { success: false, output: `edit_file no-op: old_str and new_str are identical — the file would not change. Do not repeat the same edit. Move to the next step.` };
        }
        await fs.writeFile(target, source.replace(args.old_str, args.new_str), 'utf-8');
        return { success: true, output: `Edited: ${args.filepath}` };
      }

      case 'read_file': {
        if (typeof args.filepath !== 'string' || !args.filepath.trim()) {
          return { success: false, output: 'read_file requires a non-empty string "filepath".' };
        }
        const target = path.resolve(WORKSPACE_DIR, args.filepath);
        if (!target.startsWith(WORKSPACE_DIR)) return { success: false, output: 'Path traversal denied.' };
        return { success: true, output: await fs.readFile(target, 'utf-8') };
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
          const pid = startDetached(background);
          if (pid !== undefined) detachedPids.push(pid);
          command = foreground;
        }

        // 4. A foreground server start with no `&` still blocks forever. Detect
        //    known server starters and launch them detached so the agent can
        //    probe them with a separate command instead of hanging.
        if (command && !background && LONG_RUNNING_SERVER.test(command)) {
          const pid = startDetached(command);
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
          const { stdout, stderr } = await execAsync(command, { cwd: WORKSPACE_DIR, timeout: TERMINAL_TIMEOUT_MS, shell: 'cmd.exe' });
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

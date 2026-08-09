import fs from 'fs/promises';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';
import { search, SafeSearchType } from 'duck-duck-scrape';
import { browseUrl, SearchResult, searchBingWithBrowser, searchWithBrowser, searchYahooWithBrowser } from './playwright.js';
import { WORKSPACE_DIR } from './tools.js';

const execAsync = promisify(exec);
const TERMINAL_TIMEOUT_MS = 30 * 60 * 1000;
// Respect public search services. This is pacing/backoff, not an attempt to
// bypass a provider's access controls.
const SEARCH_MIN_INTERVAL_MS = 8_000;
const SEARCH_JITTER_MS = 2_000;
const SEARCH_PROVIDER_TIMEOUT_MS = 12_000;
const FALLBACK_SWITCH_DELAY_MS = 3_000;
let nextSearchAt = 0;

export type ToolName = 'web_search' | 'browse_url' | 'write_file' | 'read_file' | 'run_terminal';

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

async function runWebSearch(query: string): Promise<ToolResult> {
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

export async function runTool(name: ToolName, args: Record<string, string>): Promise<ToolResult> {
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
        const target = path.resolve(WORKSPACE_DIR, args.filepath);
        if (!target.startsWith(WORKSPACE_DIR)) return { success: false, output: 'Path traversal denied.' };
        await fs.mkdir(path.dirname(target), { recursive: true });
        await fs.writeFile(target, args.content, 'utf-8');
        return { success: true, output: `Written: ${args.filepath}` };
      }

      case 'read_file': {
        const target = path.resolve(WORKSPACE_DIR, args.filepath);
        if (!target.startsWith(WORKSPACE_DIR)) return { success: false, output: 'Path traversal denied.' };
        return { success: true, output: await fs.readFile(target, 'utf-8') };
      }

      case 'run_terminal': {
        const { stdout, stderr } = await execAsync(args.command, { cwd: WORKSPACE_DIR, timeout: TERMINAL_TIMEOUT_MS, shell: 'cmd.exe' });
        const output = [stdout, stderr].filter(Boolean).join('\n').trim();
        return { success: true, output: output || '(no output)' };
      }

      default:
        return { success: false, output: `Unknown tool: ${name}` };
    }
  } catch (err: any) {
    return { success: false, output: err?.message ?? String(err) };
  }
}

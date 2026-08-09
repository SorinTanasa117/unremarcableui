import { Router, Request, Response } from 'express';
import fs from 'fs/promises';
import path from 'path';
import { ContextManager, Message } from '../lib/contextManager.js';
import { runTool, ToolName } from '../lib/toolRunner.js';
import { TOOL_DEFINITIONS, WORKSPACE_DIR } from '../lib/tools.js';

const router = Router();
const OLLAMA_URL = process.env.OLLAMA_URL ?? 'http://127.0.0.1:11434';
// Keep the full working context by default. Deployments can override this
// without code changes through OLLAMA_NUM_CTX.
const OLLAMA_NUM_CTX = Number.parseInt(process.env.OLLAMA_NUM_CTX ?? '262144', 10);
// Isolated story packets are capped well below this. A smaller KV cache avoids
// reserving a 262k-token window for every prose continuation.
const STORY_NUM_CTX = Number.parseInt(process.env.STORY_NUM_CTX ?? '16384', 10);
const OLLAMA_KEEP_ALIVE = process.env.OLLAMA_KEEP_ALIVE ?? '15m';
const CONTEXT_SIZES = new Set([16_384, 32_768, 65_536, 131_072, 262_144]);
// A 4–7 source report normally needs a few searches and a handful of source
// reads. Keeping this small prevents CPU-only models from spending an hour
// reprocessing a growing tool transcript.
const MAX_TOOL_CALLS_PER_RUN = 10;
const MAX_TOOL_FAILURES_PER_RUN = 4;
const MAX_SEARCH_FAILURES_PER_RUN = 2;
const SESSIONS_DIR = path.resolve(WORKSPACE_DIR, '.sessions');
interface ModelCapabilities {
  supportsTools: boolean;
  supportsThinking: boolean;
  contextLength?: number;
  systemPromptType?: string;
}
const modelCapabilitiesCache = new Map<string, ModelCapabilities>();

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

// Ensure sessions directory exists
await fs.mkdir(SESSIONS_DIR, { recursive: true });

// Per-session context managers in-memory cache
const sessions = new Map<string, ContextManager>();

interface ActiveRun {
  model: string;
  status: 'thinking' | 'tool';
  statusText: string;
  partialContent: string;
  hasStartedGenerating: boolean;
  startedAt: number;
  updatedAt: number;
}

const activeRuns = new Map<string, ActiveRun>();

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


/**
 * Some reasoning models emit their chain of thought in `content` rather than
 * Ollama's separate `thinking` field. Remove those tags before content reaches
 * the browser or is retained in session history.
 */
function createVisibleContentFilter() {
  let insideThink = false;
  let pending = '';
  const openTag = '<think>';
  const closeTag = '</think>';

  const filter = (chunk: string): string => {
    let input = pending + chunk;
    pending = '';
    let visible = '';

    while (input) {
      const lower = input.toLowerCase();
      if (insideThink) {
        const closeAt = lower.indexOf(closeTag);
        if (closeAt < 0) {
          pending = input.slice(-(closeTag.length - 1));
          return visible;
        }
        input = input.slice(closeAt + closeTag.length);
        insideThink = false;
        continue;
      }

      const openAt = lower.indexOf(openTag);
      if (openAt < 0) {
        const safeLength = Math.max(0, input.length - (openTag.length - 1));
        visible += input.slice(0, safeLength);
        pending = input.slice(safeLength);
        return visible;
      }
      visible += input.slice(0, openAt);
      input = input.slice(openAt + openTag.length);
      insideThink = true;
    }
    return visible;
  };

  filter.flush = (): string => insideThink ? '' : pending;
  return filter;
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

async function getModelCapabilities(model: string, signal: AbortSignal): Promise<ModelCapabilities> {
  const cached = modelCapabilitiesCache.get(model);
  if (cached !== undefined) return cached;

  // Use model_map.json as a baseline / fallback for capability detection.
  // Ollama's /api/show only reports capabilities for models it explicitly tags;
  // many GGUF models have tool-call support but Ollama does not list it.
  const modelMap = await getModelMap();
  const mapDef = modelMap?.models?.find((m: any) => m.id === model);
  const mapSupportsTools = mapDef?.capabilities?.function_calling === true;
  const mapSupportsThinking = mapDef?.capabilities?.thinking_mode === true;
  const mapContextLength: number | undefined = mapDef?.max_context;
  const mapSystemPromptType: string | undefined = mapDef?.system_prompt_type;

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
        contextLength: mapContextLength,
        systemPromptType: mapSystemPromptType,
      };
      modelCapabilitiesCache.set(model, capabilities);
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
      // Prefer Ollama's value; fall back to model_map max_context
      contextLength: data.model_info?.['llama.context_length'] ?? mapContextLength,
      systemPromptType: mapSystemPromptType,
    };
    modelCapabilitiesCache.set(model, capabilities);
    return capabilities;
  } catch (error: any) {
    if (error.name === 'AbortError') throw error;
    const capabilities = {
      supportsTools: mapSupportsTools,
      supportsThinking: mapSupportsThinking,
      contextLength: mapContextLength,
      systemPromptType: mapSystemPromptType,
    };
    modelCapabilitiesCache.set(model, capabilities);
    return capabilities;
  }
}

function normalizeOllamaMessages(messages: Message[], systemPromptType?: string): Message[] {
  if (systemPromptType?.toLowerCase() !== 'mistral') return messages;

  const systemContent = messages
    .filter((message) => message.role === 'system')
    .map((message) => message.content)
    .join('\n\n');
  if (!systemContent) return messages;

  const nonSystemMessages = messages.filter((message) => message.role !== 'system');
  const firstUserIndex = nonSystemMessages.findIndex((message) => message.role === 'user');
  if (firstUserIndex < 0) {
    return [{ role: 'user', content: systemContent }, ...nonSystemMessages];
  }

  return nonSystemMessages.map((message, index) => index === firstUserIndex
      ? { ...message, content: `${systemContent}\n\n${message.content}` }
      : message);
}

// Default persona fallbacks in case file read fails
const FALLBACK_PERSONAS: Record<string, string> = {
  coder: `You are an elite software engineering assistant. Your goal is to write correct, idiomatic, and highly optimized code. Provide complete, production-ready solutions and explain your structural choices concisely. You have full access to workspace file utilities and terminal commands. Use them to write, test, and verify your implementations.`,
  researcher: `You are a systematic data intelligence researcher. Your goal is to search, verify, and compile accurate information using web search and URL browsing tools. Always cross-reference facts, outline sources, and present findings in clean, structured markdown tables and summaries. Prioritize objective data and analytical depth. For a 4–7 source request, make at most 3 broad searches, then inspect 4–7 valid result URLs and write the report. Only browse URLs returned by a successful search or explicitly supplied by the user. If a search or page request fails, do not guess URLs or repeat the failed request; clearly report the limitation and provide only verifiable partial findings.`,
  creative: `You are a creative writer specializing in atmospheric dark fantasy, gothic mystery, and dramatic literature. Your style is rich, poetic, and immersive, focusing on dark aesthetics, emotional complexity, suspense, and vivid sensory details. Create compelling narratives, dramatic dialogue, and atmospheric settings while adhering to standard creative writing limits.`
};

// Dynamic markdown parser for personas.md
async function getPersonaPrompt(personaKey: string): Promise<string> {
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
    } else {
      rule = '\n\nTool-use mandate: You MUST use the provided tools (write_file, run_terminal, read_file) to perform all file and command operations. NEVER describe, narrate, or simulate writing a file or running a command in plain text — always emit the actual tool call. If you support reasoning/thinking, place internal planning inside <think>...</think> tags; your visible response must contain only tool calls and brief status lines.';
    }
    return `${persona}${rule}`;
  } catch (err) {
    const persona = FALLBACK_PERSONAS[personaKey] || FALLBACK_PERSONAS.coder;
    let rule = '';
    if (personaKey === 'creative') {
      rule = '\n\nResponse rule: Begin directly with the requested final answer. Creative Mode is prose-only: write the requested text in the chat response. Never call tools, write files, read files, or describe file operations.';
    } else {
      rule = '\n\nTool-use mandate: You MUST use the provided tools (write_file, run_terminal, read_file) to perform all file and command operations. NEVER describe, narrate, or simulate writing a file or running a command in plain text — always emit the actual tool call. If you support reasoning/thinking, place internal planning inside <think>...</think> tags; your visible response must contain only tool calls and brief status lines.';
    }
    return `${persona}${rule}`;
  }
}

async function getCtx(sessionId: string, fallbackPersona: string = 'coder'): Promise<{ ctx: ContextManager; persona: string; model: string }> {
  if (sessions.has(sessionId)) {
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

async function saveSessionToFile(sessionId: string, ctx: ContextManager, persona: string, model: string, title?: string) {
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
function isBrowseUrlAllowed(url: string, messages: Message[]): boolean {
  const normalized = normalizeUrl(url);
  if (!normalized) return false;

  const urlPattern = /https?:\/\/[^\s)\]"']+/gi;
  return messages.some((message) => {
    // The user may explicitly supply a page to inspect.
    if (message.role !== 'user' && !(message.role === 'tool' && message.tool_name === 'web_search')) return false;
    return (message.content.match(urlPattern) ?? []).some((candidate) => normalizeUrl(candidate) === normalized);
  });
}

// GET /api/ollama/models
router.get('/models', async (_req: Request, res: Response) => {
  try {
    const resp = await fetch(`${OLLAMA_URL}/api/tags`);
    const data = await resp.json();
    res.json(data);
  } catch (err: any) {
    console.error(`❌ Failed to connect to Ollama at ${OLLAMA_URL}:`, err.message);
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

// GET /api/ollama/sessions — list persistent sessions
router.get('/sessions', async (_req: Request, res: Response) => {
  try {
    const files = await fs.readdir(SESSIONS_DIR);
    const list = [];

    for (const file of files) {
      if (!file.endsWith('.json')) continue;
      try {
        const content = await fs.readFile(path.join(SESSIONS_DIR, file), 'utf-8');
        const parsed = JSON.parse(content);
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
    const { ctx, persona, model } = await getCtx(sessionId);
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
const abortControllers = new Map<string, AbortController>();
router.post('/stop', (req: Request, res: Response) => {
  const { sessionId } = req.body as { sessionId: string };
  abortControllers.get(sessionId)?.abort();
  res.json({ ok: true });
});

// POST /api/ollama/unload — stop active work for a model and release it from memory.
router.post('/unload', async (req: Request, res: Response) => {
  const { model } = req.body as { model?: string };
  if (!model) { res.status(400).json({ error: 'Missing model' }); return; }

  for (const [sessionId, run] of activeRuns) {
    if (run.model === model) abortControllers.get(sessionId)?.abort();
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
  const filePath = path.join(SESSIONS_DIR, `${sessionId}.json`);
  await fs.rm(filePath, { force: true });
  res.json({ ok: true });
});

// GET /api/ollama/tokens
router.get('/tokens', async (req: Request, res: Response) => {
  const sessionId = req.query.sessionId as string;
  const { ctx } = await getCtx(sessionId);
  res.json(ctx.getTokenUsage());
});

// POST /api/ollama/chat
router.post('/chat', async (req: Request, res: Response) => {
  const { sessionId, model, message, pendingBrowse, persona, isolated, numCtx, think, caveman } = req.body as {
    sessionId: string;
    model: string;
    message?: string;
    pendingBrowse?: string;
    persona?: string;
    isolated?: boolean;
    numCtx?: number;
    think?: boolean;
    caveman?: boolean;
  };

  const selectedPersona = persona || 'coder';
  if (!model?.trim()) {
    res.status(400).json({ error: 'Select a model before sending a message.' });
    return;
  }
  const requestedContext = numCtx ?? (isolated ? STORY_NUM_CTX : OLLAMA_NUM_CTX);
  if (!CONTEXT_SIZES.has(requestedContext)) {
    res.status(400).json({ error: 'Context must be one of 16k, 32k, 64k, 128k, or 262k.' });
    return;
  }
  if (think) {
    const capabilities = await getModelCapabilities(model, new AbortController().signal);
    if (!capabilities.supportsThinking) {
      res.status(400).json({ error: `Thinking Mode is not supported by ${model}.` });
      return;
    }
  }
  // Research runs repeatedly feed source material back to the model. A 32k
  // context is ample for the bounded source budget and avoids huge CPU KV
  // caches (the UI can otherwise request 131k+).
  // Cap context to the model's actual max_context to avoid sending Ollama a
  // larger KV cache than the GGUF was quantised for, which degrades quality.
  const capCaps = await getModelCapabilities(model, new AbortController().signal);
  const modelMaxContext = capCaps.contextLength ?? OLLAMA_NUM_CTX;
  const activeContextSize = selectedPersona === 'researcher'
    ? Math.min(requestedContext, 32_768, modelMaxContext)
    : Math.min(requestedContext, modelMaxContext);
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
  if (pendingBrowse) {
    userMessage = { role: 'user', content: `[User approved browse: ${pendingBrowse}]`, created_at: Date.now() };
  } else if (message) {
    userMessage = { role: 'user', content: message, created_at: Date.now() };
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

  // Include the just-added user prompt in the selected session's meter before
  // the model starts responding.
  send('tokens', requestCtx.getTokenUsage());

  const ac = new AbortController();
  abortControllers.set(sessionId, ac);
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
  let toolCallCount = 0;
  let toolFailureCount = 0;
  let searchFailureCount = 0;
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

  try {
    let iterating = true;
    // When research tools become unavailable, make one final model pass with
    // tools disabled so it can write a candid partial report from successful
    // results already in the conversation.
    let forceFinalResponse = false;
    let finalResponseRetryCount = 0;
    while (iterating && !ac.signal.aborted) {
      const capabilities = await getModelCapabilities(model, ac.signal);
      const activeContext = activeContextSize;
      // Creative Mode is intentionally prose-only. Completion-only GGUFs also
      // reject tool-role messages and assistant tool calls from old sessions.
      const allowTools = !forceFinalResponse && !isolated && selectedPersona !== 'creative' && capabilities.supportsTools;
      const modelMessages = forceFinalResponse
        ? [...requestCtx.getMessages(), {
          role: 'system' as const,
          content: 'Research tools are no longer available for this run. Write the final report now using only successful tool results already provided. Clearly disclose the search limitation, distinguish verified findings from unknowns, and do not call tools or invent sources.',
        }]
        : allowTools
          ? requestCtx.getMessages()
          : requestCtx.getMessages()
            .filter((message) => message.role !== 'tool')
            .map(({ tool_calls, ...message }) => message);

      // (1) swap in shrunk tool descriptions to save input tokens (caveman-shrink concept)
      // (2) append a system directive that compresses prose output, while
      //     still allowing one-line status lines between tool calls so the UI
      //     user can see "ran weblink check on www.example.com — 4 hits." etc.
      const CAVEMAN_TOOL_DEFINITIONS = [
        { type: 'function', function: { name: 'web_search',   description: 'DuckDuckGo search. Returns results.',        parameters: { type: 'object', properties: { query:    { type: 'string' } }, required: ['query'] } } },
        { type: 'function', function: { name: 'browse_url',   description: 'Fetch URL. Returns page text.',               parameters: { type: 'object', properties: { url:      { type: 'string' } }, required: ['url'] } } },
        { type: 'function', function: { name: 'write_file',   description: 'Write file to agent-workspace.',              parameters: { type: 'object', properties: { filepath: { type: 'string' }, content: { type: 'string' } }, required: ['filepath', 'content'] } } },
        { type: 'function', function: { name: 'read_file',    description: 'Read file from agent-workspace.',             parameters: { type: 'object', properties: { filepath: { type: 'string' } }, required: ['filepath'] } } },
        { type: 'function', function: { name: 'run_terminal', description: 'Run shell command in agent-workspace.',        parameters: { type: 'object', properties: { command:  { type: 'string' } }, required: ['command'] } } },
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

      if (caveman) {
        modelMessages.push({ role: 'system' as const, content: CAVEMAN_DIRECTIVE });
      }

      const normalizedMessages = normalizeOllamaMessages(modelMessages, capabilities.systemPromptType);
      const activeToolDefs = caveman ? CAVEMAN_TOOL_DEFINITIONS : TOOL_DEFINITIONS;
      const body = {
        model,
        messages: normalizedMessages,
        stream: true,
        // Supported by current Ollama releases; prevents models with optional
        // reasoning from spending their visible response on a hidden plan.
        think: Boolean(think),
        options: { num_ctx: activeContext },
        keep_alive: OLLAMA_KEEP_ALIVE,
        ...(allowTools ? { tools: activeToolDefs } : {}),
      };

      const ollamaResp = await fetchOllamaChat(body, ac.signal);

      if (!ollamaResp.ok) {
        const detail = await ollamaResp.text();
        // A model may have been removed after this chat was saved. Clear its
        // persisted selection so reopening the chat cannot keep retrying it.
        if (ollamaResp.status === 404) {
          await saveSessionToFile(sessionId, ctx, selectedPersona, '');
        }
        send('error', { message: `Ollama error: ${ollamaResp.status} ${detail || ollamaResp.statusText}` });
        break;
      }

      let assistantContent = '';
      let assistantThinking = '';
      const pendingToolCalls: any[] = [];
      const visibleContent = createVisibleContentFilter();

      const reader = ollamaResp.body!.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      while (true) {
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

          const msg = parsed.message;
          if (!msg) continue;

          if (msg.content) {
            markGenerationStarted();
            const content = visibleContent(msg.content);
            if (content) {
              assistantContent += content;
              send('token', { content });
              send('tokens', requestCtx.getTokenUsage([{ role: 'assistant', content: assistantContent }]));
              void saveProgress(assistantContent);
            }
          }

          if (msg.thinking) {
            markGenerationStarted();
            assistantThinking += msg.thinking;
            send('tokens', requestCtx.getTokenUsage([{
              role: 'assistant',
              content: assistantContent,
              thinking: assistantThinking,
              tool_calls: pendingToolCalls,
            }]));
          }

          if (msg.tool_calls?.length) {
            markGenerationStarted();
            for (const tc of msg.tool_calls) {
              pendingToolCalls.push(tc);
            }
            send('tokens', requestCtx.getTokenUsage([{
              role: 'assistant',
              content: assistantContent,
              thinking: assistantThinking,
              tool_calls: pendingToolCalls,
            }]));
          }
        }
      }

      const finalVisibleContent = visibleContent.flush();
      if (finalVisibleContent) {
        assistantContent += finalVisibleContent;
        send('token', { content: finalVisibleContent });
      }

      // Push assistant response
      const assistantMsg: Message = {
        role: 'assistant',
        content: assistantContent,
        thinking: assistantThinking || undefined,
        tool_calls: pendingToolCalls.length > 0 ? pendingToolCalls : undefined,
        created_at: Date.now(),
      };
      requestCtx.push(assistantMsg);
      if (isolated) ctx.push(assistantMsg);

      // Save state after assistant response
      await saveSessionToFile(sessionId, ctx, selectedPersona, model);

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
        for (let i = 0; i < pendingToolCalls.length; i++) {
          const tc = pendingToolCalls[i];

          if (toolCallCount >= MAX_TOOL_CALLS_PER_RUN || toolFailureCount >= MAX_TOOL_FAILURES_PER_RUN || searchFailureCount >= MAX_SEARCH_FAILURES_PER_RUN) {
            toolBudgetExceeded = true;
            limitReason = toolCallCount >= MAX_TOOL_CALLS_PER_RUN
              ? `The research tool budget of ${MAX_TOOL_CALLS_PER_RUN} calls was reached.`
              : searchFailureCount >= MAX_SEARCH_FAILURES_PER_RUN
                ? 'Search providers repeatedly failed.'
                : 'Research tools repeatedly failed.';

            // Reject this and all subsequent tool calls in this turn to keep history valid
            for (let j = i; j < pendingToolCalls.length; j++) {
              const rejectTc = pendingToolCalls[j];
              const toolName = rejectTc.function?.name ?? 'unknown_tool';
              send('tool_start', { name: toolName, args: {} });
              send('tool_result', {
                name: toolName,
                success: false,
                output: `${limitReason} Tools are no longer available. Write the final report now.`,
                durationMs: 0,
              });
              requestCtx.push({
                role: 'tool',
                content: `${limitReason} Tools are no longer available. Write the final report now using only successful tool results already provided. Do not call tools or invent sources.`,
                tool_call_id: rejectTc.id ?? toolName,
                tool_name: toolName,
                duration_ms: 0,
                created_at: Date.now(),
              });
            }
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

          const toolStartedAt = Date.now();
          const result = toolName === 'browse_url' && !isBrowseUrlAllowed(args.url, requestCtx.getMessages())
            ? {
              success: false,
              output: `Browse denied: ${args.url} was not returned by a successful search and was not supplied by the user. Do not guess URLs; use only search-result links already in the context.`,
            }
            : await runTool(toolName, args);
          const toolDurationMs = Date.now() - toolStartedAt;
          toolCallCount += 1;
          if (!result.success) {
            toolFailureCount += 1;
            if (toolName === 'web_search') searchFailureCount += 1;
          }

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
            content: toolOutput,
            tool_call_id: tc.id ?? toolName,
            tool_name: toolName,
            duration_ms: toolDurationMs,
            created_at: Date.now(),
          });
          const resumedRun = activeRuns.get(sessionId);
          if (resumedRun) {
            resumedRun.status = 'thinking';
            resumedRun.statusText = 'Processing results…';
            resumedRun.partialContent = '';
          }
        }

        send('tokens', requestCtx.getTokenUsage());
        await saveSessionToFile(sessionId, ctx, selectedPersona, model);
        if (toolBudgetExceeded) {
          forceFinalResponse = true;
        }
        continue;
      }

      iterating = false;
      send('tokens', requestCtx.getTokenUsage());
      send('done', {});
    }

    if (ac.signal.aborted) {
      send('stopped', {});
    }
  } catch (err: any) {
    if (err.name !== 'AbortError') {
      const detail = err.cause?.message || err.message || 'Unknown Ollama connection error';
      console.error(`[Ollama chat] ${model} failed:`, detail);
      send('error', { message: `Ollama request failed: ${detail}` });
    } else {
      send('stopped', {});
    }
  } finally {
    abortControllers.delete(sessionId);
    activeRuns.delete(sessionId);
    await saveSessionToFile(sessionId, ctx, selectedPersona, model);
    res.end();
  }
});

export default router;

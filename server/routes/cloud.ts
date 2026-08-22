/**
 * Cloud provider routes.
 *
 * Wires OpenAI-compatible chat completions (OpenRouter, Factory.ai) into the
 * existing agent harness so cloud models get full feature parity with local
 * Ollama runs:
 *
 *   - Same ContextManager, persona prompt, and tool definitions
 *   - Same SSE event stream the React client already consumes
 *   - Same session persistence (sessions, novel drafts, token stats)
 *   - Same tool loop (web_search, browse_url, write_file, read_file,
 *     run_terminal) with bounded retries
 *
 * The API key is supplied per-request and never persisted server-side. It is
 * forwarded only to the selected provider and never logged.
 */

import { Router, Request, Response } from 'express';
import fs from 'fs/promises';
import path from 'path';
import { ContextManager, Message, ToolCall } from '../lib/contextManager.js';
import { runTool, ToolName } from '../lib/toolRunner.js';
import { TOOL_DEFINITIONS, WORKSPACE_DIR } from '../lib/tools.js';
import * as novelStorage from '../lib/novel/storage.js';
import { inferenceDispatcher } from '../lib/inferenceDispatcher.js';
import {
  CLOUD_PROVIDERS,
  CLOUD_PROVIDER_LIST,
  isCloudProviderId,
  listCloudModels,
  suggestContextSize,
  validateCloudKey,
  type CloudProviderId,
  type CloudModelInfo,
} from '../lib/cloudProviders.js';
import {
  abortControllers,
  activeRuns,
  isBrowseUrlAllowed,
  sessions,
} from './ollama.js';
import {
  ingestAttachments,
  readImageAttachments,
  type IncomingAttachment,
} from '../lib/attachments.js';

const router = Router();
const SESSIONS_DIR = path.resolve(WORKSPACE_DIR, '.sessions');
const CONTEXT_SIZES = [16_384, 32_768, 65_536, 131_072, 262_144];
const MAX_TOOL_CALLS_PER_RUN = 10;
const MAX_TOOL_FAILURES_PER_RUN = 4;
const MAX_SEARCH_FAILURES_PER_RUN = 2;
const STORY_NUM_CTX = Number.parseInt(process.env.STORY_NUM_CTX ?? '16384', 10);
const DEFAULT_NUM_CTX = Number.parseInt(process.env.OLLAMA_NUM_CTX ?? '262144', 10);
const NOVEL_DRAFT_NUM_PREDICT = Number.parseInt(process.env.NOVEL_DRAFT_NUM_PREDICT ?? '8192', 10);
const parsedNumPredict = Number.parseInt(process.env.OLLAMA_NUM_PREDICT ?? '8192', 10);
const OLLAMA_NUM_PREDICT = Number.isFinite(parsedNumPredict) && parsedNumPredict > 0 ? parsedNumPredict : 8192;
const TOOL_CONTEXT_CHAR_LIMIT = 8000;

// In-flight abort controllers and active-run snapshots are keyed by sessionId,
// matching the lifecycle in server/routes/ollama.ts. They let a refreshed
// browser tab rehydrate a still-running cloud chat.
interface ActiveRunSnapshot {
  status: 'thinking' | 'tool' | 'paused';
  statusText: string;
  partialContent: string;
  hasStartedGenerating?: boolean;
  startedAt: number;
  updatedAt: number;
  model?: string;
  provider?: CloudProviderId;
}

interface SessionRecord {
  sessionId: string;
  title?: string;
  persona: string;
  model: string;
  provider?: CloudProviderId;
  updatedAt: number;
  messages: Message[];
  run?: {
    status: 'thinking' | 'tool' | 'paused';
    statusText: string;
    partialContent: string;
    hasStartedGenerating?: boolean;
    startedAt: number;
    updatedAt: number;
    model?: string;
    provider?: string;
  };
}
// CAVEMAN_TOOL_DEFINITIONS is a tighter, more compact version of the tool
// list sent to remote providers. Same names, smaller surface, lower token
// cost on the first request of each turn.
const CAVEMAN_TOOL_DEFINITIONS = [
  { type: 'function', function: { name: 'web_search',   description: 'Brave Search; DuckDuckGo only if Brave reports exhausted credits. Returns results.', parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] } } },
  { type: 'function', function: { name: 'browse_url',   description: 'Fetch URL. Returns page text.',               parameters: { type: 'object', properties: { url: { type: 'string' } }, required: ['url'] } } },
  { type: 'function', function: { name: 'write_file',   description: 'Write new file to agent-workspace.',           parameters: { type: 'object', properties: { filepath: { type: 'string' }, content: { type: 'string' } }, required: ['filepath', 'content'] } } },
  { type: 'function', function: { name: 'edit_file',    description: 'Fix part of existing file. Replace one unique old_str with new_str. Use for edits, not full rewrite.', parameters: { type: 'object', properties: { filepath: { type: 'string' }, old_str: { type: 'string' }, new_str: { type: 'string' } }, required: ['filepath', 'old_str', 'new_str'] } } },
  { type: 'function', function: { name: 'read_file',    description: 'Read file from agent-workspace.',             parameters: { type: 'object', properties: { filepath: { type: 'string' } }, required: ['filepath'] } } },
  { type: 'function', function: { name: 'run_terminal', description: 'Run shell command in agent-workspace.',      parameters: { type: 'object', properties: { command: { type: 'string' } }, required: ['command'] } } },
];

const CAVEMAN_DIRECTIVE = [
  'Respond terse like smart caveman. All technical substance stay. Only fluff die.',
  'Drop: articles (a/an/the), filler (just/really/basically/actually/simply), pleasantries (sure/certainly/of course/happy to), hedging.',
  'Fragments OK. Short synonyms preferred. No decorative tables/emoji.',
  'Never drop: not/never/no/only/except. Numbers exact. Technical terms exact. Code blocks unchanged. Errors quoted exact.',
  'Tool calls: fire direct. After every tool result, emit ONE short status line (single sentence, <120 chars) before next call or final answer.',
  'No preamble, no plan recap, no apologies, no hedging.',
].join('\n');

const FALLBACK_PERSONAS: Record<string, string> = {
  coder: 'You are an elite software engineering assistant. Your goal is to write correct, idiomatic, and highly optimized code. Provide complete, production-ready solutions and explain your structural choices concisely. You have full access to workspace file utilities and terminal commands. Use them to write, test, and verify your implementations.',
  researcher: 'You are a systematic data intelligence researcher. Your goal is to search, verify, and compile accurate information using web search and URL browsing tools. Always cross-reference facts, outline sources, and present findings in clean, structured markdown tables and summaries. For a 4-7 source request, make at most 3 broad searches, then inspect 4-7 valid result URLs and write the report. If a search or page request fails, do not guess URLs or repeat the failed request; clearly report the limitation.',
  creative: 'You are a creative writer specializing in atmospheric dark fantasy, gothic mystery, and dramatic literature. Your style is rich, poetic, and immersive, focusing on dark aesthetics, emotional complexity, suspense, and vivid sensory details. Create compelling narratives, dramatic dialogue, and atmospheric settings.',
  system: 'You are a general-purpose AI assistant. Answer user questions directly and honestly without hallucinating. Research unknown topics using web search and browsing tools when your knowledge is insufficient. Admit uncertainty and verify facts rather than guessing.',
  novelist: 'You are a long-form novel-writing assistant who follows strict outline beats and character rosters.',
};

async function ensureSessionsDir(): Promise<void> {
  await fs.mkdir(SESSIONS_DIR, { recursive: true });
}

async function getPersonaPrompt(personaKey: string): Promise<string> {
  // Reuse the persona prompts from personas.md when available; identical to
  // the prompt used for local runs so behavior does not change based on
  // backend. The fallback table is intentionally narrower than the local
  // version because cloud models do not always ship the same tool-use
  // mandate prompt — the role rule below normalizes behavior.
  let basePrompt = FALLBACK_PERSONAS[personaKey] ?? FALLBACK_PERSONAS.coder;
  try {
    const filePath = path.resolve(WORKSPACE_DIR, '../personas.md');
    const content = await fs.readFile(filePath, 'utf-8');
    const sections = content.split(/^#\s+/m);
    for (const section of sections) {
      const lines = section.split('\n');
      const title = lines[0].trim().toLowerCase();
      if (title === personaKey.toLowerCase()) {
        const body = lines.slice(1).join('\n').trim();
        if (body) basePrompt = body;
        break;
      }
    }
  } catch {}
  let rule = '';
  if (personaKey === 'creative') {
    rule = '\n\nResponse rule: Begin directly with the requested final answer. Creative Mode is prose-only: write the requested text in the chat response. Never call tools, write files, read files, or describe file operations.';
  } else if (personaKey === 'system') {
    rule = '\n\nTool-use guidance: You have access to web_search, browse_url, and read_file tools. Use them proactively to research unknown topics, verify uncertain claims, and recommend models based on current knowledge.';
  } else {
    rule = '\n\nTool-use mandate: You MUST use the provided tools (write_file, edit_file, run_terminal, read_file) to perform all file and command operations. NEVER describe, narrate, or simulate writing a file or running a command in plain text - always emit the actual tool call. To FIX or CHANGE part of an existing file, use edit_file (replace a unique old_str with new_str) instead of rewriting the whole file with write_file. In each write_file call put "filepath" before "content". If you support reasoning/thinking, keep internal planning inside <think>...</think> strictly concise and bounded. Never draft whole file contents inside thoughts or meander; close </think> promptly and emit the tool call.';
  }
  return `${basePrompt}${rule}`;
}

async function getCtx(sessionId: string, fallbackPersona: string): Promise<{ ctx: ContextManager; persona: string; model: string; provider?: CloudProviderId }> {
  if (sessions.has(sessionId)) {
    let currentPersona = fallbackPersona;
    let currentModel = '';
    let currentProvider: CloudProviderId | undefined;
    try {
      const filePath = path.join(SESSIONS_DIR, `${sessionId}.json`);
      const fileContent = await fs.readFile(filePath, 'utf-8');
      const data = JSON.parse(fileContent) as Partial<SessionRecord>;
      if (data && data.persona) currentPersona = data.persona;
      if (data && typeof data.model === 'string') currentModel = data.model;
      if (data && isCloudProviderId(data.provider)) currentProvider = data.provider;
    } catch {}
    return { ctx: sessions.get(sessionId)!, persona: currentPersona, model: currentModel, provider: currentProvider };
  }
  await ensureSessionsDir();
  const filePath = path.join(SESSIONS_DIR, `${sessionId}.json`);
  let loadedPersona = fallbackPersona;
  let loadedModel = '';
  let loadedProvider: CloudProviderId | undefined;
  let messagesToLoad: Message[] = [];
  try {
    const fileContent = await fs.readFile(filePath, 'utf-8');
    const data = JSON.parse(fileContent) as Partial<SessionRecord>;
    if (data) {
      if (data.persona) loadedPersona = data.persona;
      if (typeof data.model === 'string') loadedModel = data.model;
      if (isCloudProviderId(data.provider)) loadedProvider = data.provider;
      if (Array.isArray(data.messages)) messagesToLoad = data.messages;
    }
  } catch {}
  const systemPrompt = await getPersonaPrompt(loadedPersona);
  const ctx = new ContextManager(systemPrompt);
  if (messagesToLoad.length > 0) {
    ctx.reset();
    for (const msg of messagesToLoad) {
      if (msg.role !== 'system') ctx.push(msg);
    }
  }
  sessions.set(sessionId, ctx);
  return { ctx, persona: loadedPersona, model: loadedModel, provider: loadedProvider };
}

async function saveSessionToFile(
  sessionId: string,
  ctx: ContextManager,
  persona: string,
  model: string,
  provider: CloudProviderId,
  title?: string,
): Promise<void> {
  await ensureSessionsDir();
  const filePath = path.join(SESSIONS_DIR, `${sessionId}.json`);
  const messages = ctx.getMessages();
  let sessionTitle = title;
  if (!sessionTitle) {
    const firstUser = messages.find((m) => m.role === 'user');
    sessionTitle = firstUser
      ? firstUser.content.slice(0, 30) + (firstUser.content.length > 30 ? '…' : '')
      : 'New Chat';
  }
  const payload: SessionRecord = {
    sessionId,
    title: sessionTitle,
    persona,
    model,
    provider,
    updatedAt: Date.now(),
    messages,
    run: activeRuns.get(sessionId),
  };
  await fs.writeFile(filePath, JSON.stringify(payload, null, 2), 'utf-8');
}

function summarizeCloudMessages(messages: Message[]) {
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
    lastRoles: messages.slice(-5).map((message) => message.role),
    assistantToolCallNames: assistantToolCallNames.slice(-5),
    assistantToolCallCount,
  };
}

interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

interface OpenAIDelta {
  role?: string;
  content?: string | null;
  reasoning_content?: string | null;
  tool_calls?: Array<{
    index?: number;
    id?: string;
    type?: 'function';
    function?: { name?: string; arguments?: string };
  }>;
}

interface OpenAIChoice {
  index?: number;
  delta?: OpenAIDelta;
  finish_reason?: string | null;
  message?: OpenAIDelta;
}

interface OpenAIChunk {
  id?: string;
  model?: string;
  choices?: OpenAIChoice[];
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
}

interface OpenAIRequestBody {
  model: string;
  stream: true;
  messages: Array<Record<string, unknown>>;
  tools?: unknown[];
  tool_choice?: 'auto' | 'none' | 'required';
  temperature?: number;
  max_tokens?: number;
}

function buildOpenAIMessages(messages: Message[]): Array<Record<string, unknown>> {
  // Strip internal-only fields (tool_name, duration_ms, created_at) that
  // OpenAI-compatible APIs do not understand.
  return messages.map((message) => {
    if (message.role === 'assistant' && message.tool_calls?.length) {
      const toolCalls: OpenAIToolCall[] = message.tool_calls.map((tc) => ({
        id: tc.id,
        type: 'function',
        function: {
          name: tc.function.name,
          arguments: typeof tc.function.arguments === 'string'
            ? tc.function.arguments
            : JSON.stringify(tc.function.arguments ?? {}),
        },
      }));
      return {
        role: 'assistant',
        content: message.content ?? '',
        ...(message.thinking ? { reasoning_content: message.thinking } : {}),
        tool_calls: toolCalls,
      };
    }
    if (message.role === 'tool') {
      return {
        role: 'tool',
        tool_call_id: message.tool_call_id ?? '',
        content: typeof message.content === 'string' ? message.content : '',
      };
    }
    return {
      role: message.role,
      content: message.content ?? '',
    };
  });
}

// Like buildOpenAIMessages but, for user turns that carry image attachments,
// rewrites `content` into the OpenAI multimodal array form
// ([{type:'text'}, {type:'image_url', image_url:{url:'data:…'}}]). Images are
// re-read from disk each request so earlier turns keep their images too.
async function buildOpenAIMessagesWithImages(
  messages: Message[],
  sessionId: string,
): Promise<Array<Record<string, unknown>>> {
  const base = buildOpenAIMessages(messages);
  return Promise.all(base.map(async (record, index) => {
    const original = messages[index];
    if (!original || original.role !== 'user') return record;
    const imgs = await readImageAttachments(SESSIONS_DIR, sessionId, original.attachments);
    if (imgs.length === 0) return record;
    const text = typeof record.content === 'string' ? record.content : '';
    const parts: Array<Record<string, unknown>> = [];
    if (text) parts.push({ type: 'text', text });
    for (const img of imgs) {
      parts.push({ type: 'image_url', image_url: { url: `data:${img.mimeType};base64,${img.base64}` } });
    }
    return { ...record, content: parts };
  }));
}

function summarizeToolStart(name: string, args?: Record<string, string>): string {
  const file = args?.filepath;
  const query = args?.query;
  const url = args?.url;
  const cmd = args?.command;
  switch (name) {
    case 'write_file': return file ? `Writing file \`${file}\`…` : 'Writing a file…';
    case 'edit_file': return file ? `Editing file \`${file}\`…` : 'Editing a file…';
    case 'read_file': return file ? `Reading file \`${file}\`…` : 'Reading a file…';
    case 'web_search': return query ? `Searching the web for "${query}"…` : 'Searching the web…';
    case 'browse_url': return url ? `Browsing webpage: ${url}…` : 'Browsing webpage…';
    case 'run_terminal': return cmd ? `Running terminal command \`${cmd}\`…` : 'Running terminal command…';
    default: return `Executing ${name}…`;
  }
}

function summarizeToolResult(name: string, args?: Record<string, string>, success = true): string {
  const file = args?.filepath;
  const outcome = success ? '' : ' but it reported an error';
  switch (name) {
    case 'write_file': return file ? `Wrote \`${file}\`${outcome}.` : `Wrote a file${outcome}.`;
    case 'edit_file': return file ? `Edited \`${file}\`${outcome}.` : `Edited a file${outcome}.`;
    case 'read_file': return file ? `Read \`${file}\`${outcome}.` : `Read a file${outcome}.`;
    case 'web_search': return `Completed a web search${outcome}.`;
    case 'browse_url': return `Inspected the requested page${outcome}.`;
    case 'run_terminal': return `Ran a terminal command${outcome}.`;
    default: return `Completed the requested action${outcome}.`;
  }
}

// Provider-aware request URL. Most OpenAI-compatible gateways accept
// /chat/completions directly; both OpenRouter and Factory.ai follow that
// convention.
function chatCompletionsUrl(provider: CloudProviderId): string {
  return `${CLOUD_PROVIDERS[provider].baseUrl}/chat/completions`;
}

function resolveApiKey(provider: CloudProviderId, supplied: unknown): string {
  if (typeof supplied === 'string' && supplied.trim()) return supplied.trim();
  return process.env[CLOUD_PROVIDERS[provider].apiKeyEnv]?.trim() ?? '';
}

function buildAuthHeaders(descriptor: typeof CLOUD_PROVIDERS[CloudProviderId], apiKey: string): Record<string, string> {
  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${apiKey}`,
    ...(descriptor.defaultHeaders ?? {}),
  };
}

async function runCloudChat(
  descriptor: typeof CLOUD_PROVIDERS[CloudProviderId],
  body: OpenAIRequestBody,
  apiKey: string,
  signal: AbortSignal,
): Promise<globalThis.Response> {
  // No retry loop here; cloud APIs return user-fixable errors (401, 402,
  // 429) and we want to surface them immediately rather than burning budget
  // on automatic retries.
  return fetch(chatCompletionsUrl(descriptor.id), {
    method: 'POST',
    headers: buildAuthHeaders(descriptor, apiKey),
    body: JSON.stringify(body),
    signal,
    dispatcher: inferenceDispatcher,
  });
}

// /api/cloud/providers — static metadata the client uses to render the picker.
router.get('/providers', (_req: Request, res: Response) => {
  res.json({
    providers: CLOUD_PROVIDER_LIST.map((provider) => ({
      id: provider.id,
      label: provider.label,
      description: provider.description,
      keyHint: provider.keyHint,
      hasEnvKey: Boolean(process.env[provider.apiKeyEnv]?.trim()),
    })),
  });
});

// /api/cloud/models — fresh or cached model list, used by the modal picker.
// Use POST so API keys never appear in browser history, proxy logs, or URLs.
router.post('/models', async (req: Request, res: Response) => {
  const provider = req.body?.provider;
  const apiKey = isCloudProviderId(provider) ? resolveApiKey(provider, req.body?.apiKey) : '';
  if (!isCloudProviderId(provider)) {
    res.status(400).json({ error: 'Unknown provider.' });
    return;
  }
  if (!apiKey) {
    res.status(400).json({ error: 'API key is required.' });
    return;
  }
  try {
    const { models, cached } = await listCloudModels(provider, apiKey);
    res.json({
      cached,
      provider,
      models: models.map((m: CloudModelInfo) => ({
        id: m.id,
        name: m.name,
        contextLength: m.contextLength,
        supportsTools: m.supportsTools,
      })),
    });
  } catch (err: any) {
    const status = err?.message?.match(/\((\d{3})\)/)?.[1];
    res.status(status ? Number(status) : 502).json({ error: err?.message ?? 'Failed to fetch models.' });
  }
});

// /api/cloud/validate — quick key check used by the modal "Test connection" button.
router.post('/validate', async (req: Request, res: Response) => {
  const { provider, apiKey } = req.body ?? {};
  if (!isCloudProviderId(provider)) {
    res.status(400).json({ error: 'Unknown provider.' });
    return;
  }
  const resolvedApiKey = resolveApiKey(provider, apiKey);
  if (!resolvedApiKey) {
    res.status(400).json({ error: 'API key is required.' });
    return;
  }
  const result = await validateCloudKey(provider, resolvedApiKey);
  if (!result.ok) {
    res.status(result.status === 401 || result.status === 403 ? 401 : 502).json(result);
    return;
  }
  res.json({ ok: true, modelCount: result.modelCount });
});

// /api/cloud/chat — main SSE streaming endpoint. Mirrors the Ollama chat
// surface but rewrites the wire format to OpenAI-compatible chat completions.
router.post('/chat', async (req: Request, res: Response) => {
  const {
    sessionId,
    model,
    message,
    pendingBrowse,
    persona,
    isolated,
    numCtx,
    caveman,
    provider,
    apiKey,
    attachments,
  } = req.body as {
    sessionId: string;
    model: string;
    message?: string;
    pendingBrowse?: string;
    persona?: string;
    isolated?: boolean;
    numCtx?: number;
    caveman?: boolean;
    provider: CloudProviderId;
    apiKey: string;
    attachments?: IncomingAttachment[];
  };

  if (!isCloudProviderId(provider)) {
    res.status(400).json({ error: 'Unknown provider.' });
    return;
  }
  const resolvedApiKey = resolveApiKey(provider, apiKey);
  if (!resolvedApiKey) {
    res.status(400).json({ error: 'API key is required for cloud providers.' });
    return;
  }
  if (!model?.trim()) {
    res.status(400).json({ error: 'Select a model before sending a message.' });
    return;
  }

  const descriptor = CLOUD_PROVIDERS[provider];
  const cleanApiKey = resolvedApiKey;
  const ac = new AbortController();

  const selectedPersona = persona || 'coder';
  let requestedContext = numCtx ?? (isolated ? STORY_NUM_CTX : DEFAULT_NUM_CTX);
  if (!CONTEXT_SIZES.includes(requestedContext)) {
    // Caller sent an unsupported context size; clamp to the largest one we
    // accept rather than rejecting the request outright.
    requestedContext = CONTEXT_SIZES[CONTEXT_SIZES.length - 1];
  }
  const activeContextSize = selectedPersona === 'researcher'
    ? Math.min(requestedContext, 32_768)
    : requestedContext;

  let modelSupportsTools: boolean | undefined;
  try {
    const listed = await listCloudModels(provider, cleanApiKey);
    modelSupportsTools = listed.models.find((item) => item.id === model)?.supportsTools;
  } catch {
    // Some compatible gateways do not expose /models. Keep tool calling
    // enabled unless the provider explicitly reports that model lacks it.
  }

  const { ctx } = await getCtx(sessionId, selectedPersona);
  const requestCtx = isolated
    ? new ContextManager(await getPersonaPrompt(selectedPersona))
    : ctx;

  // Novel draft handling — same flow as the local Ollama path so novels
  // work identically whether the user picks a local model or a cloud one.
  let userMessage: Message | undefined;
  let novelDraftContext: string | null = null;
  let novelDraftTargetWords = 1500;
  const novelDraftMatch = message?.match(/^\[NOVEL_DRAFT:([^:]+):(\d+)\]$/);
  const isNovelDraft = Boolean(novelDraftMatch && selectedPersona === 'novelist');
  let outputNumPredict = isNovelDraft ? NOVEL_DRAFT_NUM_PREDICT : OLLAMA_NUM_PREDICT;
  if (isNovelDraft && novelDraftMatch) {
    const [, novelId, chapterNumStr] = novelDraftMatch;
    const chapterNum = parseInt(chapterNumStr, 10);
    try {
      const novel = await novelStorage.getNovel(novelId);
      if (novel) {
        const outlineEntry = novel.outline.find((o) => o.number === chapterNum);
        const rollingContext = await novelStorage.getRollingContext(novelId, chapterNum);
        const bible = novel.bible;
        let contextContent = '';
        const rosterNames = [
          outlineEntry?.pov_character,
          ...(outlineEntry?.characters_involved ?? []),
          ...(outlineEntry?.characters ?? []),
        ].filter(Boolean).map((c) => c!.toLowerCase().trim());
        const presentChars = bible.characters.filter((c) => {
          const charNameLower = c.name.toLowerCase();
          return rosterNames.some((r) => charNameLower.includes(r) || r.includes(charNameLower));
        });
        const futureChars = bible.characters.filter((c) => {
          const charNameLower = c.name.toLowerCase();
          return !rosterNames.some((r) => charNameLower.includes(r) || r.includes(charNameLower));
        });
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
            for (const loc of bible.locations) contextContent += `- **${loc.name}**: ${loc.description}\n`;
          }
          if (bible.facts?.length > 0) contextContent += `World Facts: ${bible.facts.join('; ')}\n`;
          if (bible.style_notes) contextContent += `Style & Tone: ${bible.style_notes}\n`;
          contextContent += '\n';
        }
        if (bible.steering_notes && bible.steering_notes.length > 0) {
          contextContent += '## AUTHOR MANDATORY DIRECTIVES & COURSE CORRECTIONS\n';
          for (const note of bible.steering_notes) contextContent += `- ${note}\n`;
          contextContent += '\n';
        }
        if (rollingContext.arcSummaries.length > 0) {
          contextContent += '## Story So Far\n';
          for (const arc of rollingContext.arcSummaries) contextContent += `${arc.summary}\n`;
          contextContent += '\n';
        }
        if (rollingContext.recentSummaries.length > 0) {
          contextContent += '## Recent Chapters\n';
          for (const sum of rollingContext.recentSummaries) contextContent += `Ch ${sum.chapter}: ${sum.summary}\n`;
          contextContent += '\n';
        }
        if (outlineEntry) {
          contextContent += `## STRICT CHAPTER OUTLINE & SCOPE\n`;
          contextContent += `Chapter ${outlineEntry.number}: ${outlineEntry.title}\n`;
          contextContent += `POV Character: ${outlineEntry.pov_character}\n`;
          contextContent += `Description / Beats: ${outlineEntry.description || outlineEntry.beat_summary}\n`;
          contextContent += `Approved Roster: ${presentChars.map((c) => c.name).join(', ') || (outlineEntry.characters_involved ?? []).join(', ') || 'POV character only'}\n\n`;
          if (futureChars.length > 0) {
            contextContent += `WARNING: ABSOLUTE CHARACTER RESTRICTION:\n`;
            contextContent += `DO NOT include future-chapter characters in Chapter ${chapterNum}: ${futureChars.map((c) => c.name).join(', ')}.\n\n`;
          }
          const requestedWords = Math.max(
            outlineEntry.target_words ?? 0,
            (outlineEntry.target_pages ?? 8) * (novel.manifest.words_per_page || 250),
          ) || 1500;
          novelDraftTargetWords = requestedWords;
          contextContent += `Target: approximately ${requestedWords} words.\n`;
          contextContent += `Write this chapter as immersive narrative prose. End with [END_OF_CHAPTER] on its own line.\n`;
          contextContent += '\n';
        }
        if (rollingContext.rawTail) contextContent += `## Previous Chapter Ending\n...${rollingContext.rawTail}\n\n`;
        contextContent += `Now write Chapter ${chapterNum}. When complete, end with [END_OF_CHAPTER] on its own line.`;
        novelDraftContext = contextContent;
        userMessage = {
          role: 'user',
          content: `Draft Chapter ${chapterNum}: ${outlineEntry?.title ?? 'Untitled'}`,
          created_at: Date.now(),
        };
      }
    } catch (err) {
      console.error('[Cloud novel draft context assembly failed]', err);
    }
  }
  if (isNovelDraft) {
    outputNumPredict = Math.max(NOVEL_DRAFT_NUM_PREDICT, Math.ceil(novelDraftTargetWords * 1.5) + 1500);
  }
  // Persist attachments to the session folder and fold text files into the
  // prompt; image metadata rides on the user message and the bytes are re-read
  // as data URLs when the request payload is built.
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
  if (novelDraftContext) requestCtx.push({ role: 'system', content: novelDraftContext, created_at: Date.now() });
  if (userMessage) {
    requestCtx.push(userMessage);
    if (isolated) ctx.push(userMessage);
  }
  await saveSessionToFile(sessionId, ctx, selectedPersona, model, provider);

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

  send('tokens', requestCtx.getTokenUsage());
  abortControllers.set(sessionId, ac);
  activeRuns.set(sessionId, {
    model,
    provider,
    status: 'thinking',
    statusText: 'Thinking…',
    partialContent: '',
    hasStartedGenerating: false,
    startedAt: Date.now(),
    updatedAt: Date.now(),
  });

  let toolCallCount = 0;
  let toolFailureCount = 0;
  let searchFailureCount = 0;
  let iterationNumber = 0;
  let lastProgressSavedAt = 0;
  let pendingPartialContent = '';
  let forceFinalResponse = false;
  const saveProgress = async (partialContent: string, force = false) => {
    const run = activeRuns.get(sessionId);
    if (!run) return;
    run.partialContent = partialContent;
    run.updatedAt = Date.now();
    if (force || run.updatedAt - lastProgressSavedAt >= 750) {
      lastProgressSavedAt = run.updatedAt;
      await saveSessionToFile(sessionId, ctx, selectedPersona, model, provider);
    }
  };
  await saveProgress('', true);

  try {
    let iterating = true;
    while (iterating && !ac.signal.aborted) {
      iterationNumber += 1;
      pendingPartialContent = '';
      const allowTools = !forceFinalResponse
        && !isolated
        && selectedPersona !== 'creative'
        && modelSupportsTools !== false;
      let modelMessages = requestCtx.getMessages();
      if (!allowTools) {
        modelMessages = forceFinalResponse
          ? [
            ...modelMessages,
            {
              role: 'system',
              content: 'Research tools are no longer available for this run. Write the final report now using only successful tool results already provided. Clearly disclose limitations, distinguish verified findings from unknowns, and do not call tools or invent sources.',
            },
          ]
          : modelMessages
            .filter((message) => message.role !== 'tool')
            .map(({ tool_calls, tool_name, duration_ms, ...rest }) => rest as Message);
      }
      if (caveman) {
        modelMessages = [
          ...modelMessages,
          { role: 'system', content: CAVEMAN_DIRECTIVE, created_at: Date.now() },
        ];
      }
      const activeToolDefs = caveman ? CAVEMAN_TOOL_DEFINITIONS : TOOL_DEFINITIONS;
      const openAiMessages = await buildOpenAIMessagesWithImages(modelMessages, sessionId);
      const body: OpenAIRequestBody = {
        model,
        stream: true,
        messages: openAiMessages,
        ...(allowTools ? { tools: activeToolDefs, tool_choice: 'auto' as const } : {}),
        max_tokens: outputNumPredict,
      };

      const upstream = await runCloudChat(descriptor, body, cleanApiKey, ac.signal);
      if (!upstream.ok) {
        const detail = await upstream.text().catch(() => '');
        const safeDetail = detail.replace(/sk-[A-Za-z0-9_-]+/g, 'sk-***').slice(0, 500);
        const message = `${descriptor.label} error: ${upstream.status} ${upstream.statusText}${safeDetail ? ` — ${safeDetail}` : ''}`;
        console.error('[Cloud chat upstream error]', upstream.status, detail.slice(0, 200));
        send('error', { message });
        break;
      }
      const reader = upstream.body?.getReader();
      if (!reader) {
        send('error', { message: 'Provider returned an empty response stream.' });
        break;
      }
      const decoder = new TextDecoder();
      let buf = '';
      let assistantContent = '';
      let assistantThinking = '';
      const pendingToolCalls = new Map<number, OpenAIToolCall>();
      let sawToolCall = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done || ac.signal.aborted) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line) continue;
          if (line.startsWith(':')) continue; // SSE comments
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === '[DONE]') continue;
          let parsed: OpenAIChunk;
          try {
            parsed = JSON.parse(payload);
          } catch {
            continue;
          }
          const choice = parsed.choices?.[0];
          if (!choice) continue;
          const delta = choice.delta ?? choice.message;
          if (!delta) continue;
          if (delta.content) {
            markGenerationStarted();
            assistantContent += delta.content;
            pendingPartialContent = assistantContent;
            send('token', { content: delta.content });
            send('tokens', requestCtx.getTokenUsage([{ role: 'assistant', content: assistantContent, thinking: assistantThinking, tool_calls: Array.from(pendingToolCalls.values()) }]));
            void saveProgress(assistantContent);
          }
          if (delta.reasoning_content) {
            markGenerationStarted();
            assistantThinking += delta.reasoning_content;
            if (!caveman) send('thinking', { content: delta.reasoning_content });
            send('tokens', requestCtx.getTokenUsage([{ role: 'assistant', content: assistantContent, thinking: assistantThinking, tool_calls: Array.from(pendingToolCalls.values()) }]));
          }
          if (delta.tool_calls?.length) {
            markGenerationStarted();
            for (const partial of delta.tool_calls) {
              const idx = partial.index ?? 0;
              let existing = pendingToolCalls.get(idx);
              if (!existing) {
                existing = {
                  id: partial.id ?? `call_${Math.random().toString(36).slice(2, 10)}`,
                  type: 'function',
                  function: { name: partial.function?.name ?? '', arguments: partial.function?.arguments ?? '' },
                };
                pendingToolCalls.set(idx, existing);
              } else {
                if (partial.id) existing.id = partial.id;
                if (partial.function?.name) existing.function.name += partial.function.name;
                if (partial.function?.arguments) existing.function.arguments += partial.function.arguments;
              }
            }
          }
          if (choice.finish_reason === 'tool_calls' || (choice.finish_reason === 'stop' && pendingToolCalls.size > 0)) {
            sawToolCall = pendingToolCalls.size > 0;
          }
        }
      }

      if (!sawToolCall && pendingToolCalls.size === 0) {
        // Plain text completion — push the assistant turn into context and
        // exit the iteration loop.
        const assistantMessage: Message = {
          role: 'assistant',
          content: assistantContent,
          // Caveman hides the thought, not the thinking: reasoning still runs,
          // but its text is never streamed (above) nor persisted, so it stays
          // hidden on session reload too. Matches ollama.ts.
          ...(!caveman && assistantThinking ? { thinking: assistantThinking } : {}),
          created_at: Date.now(),
        };
        requestCtx.push(assistantMessage);
        if (isolated) ctx.push(assistantMessage);
        await saveSessionToFile(sessionId, ctx, selectedPersona, model, provider);
        break;
      }

      // Tool-call completion — finalize tool call objects, push assistant
      // turn into context, execute each tool, push results, loop.
      const finalizedToolCalls: ToolCall[] = Array.from(pendingToolCalls.values()).map((tc, idx) => ({
        id: tc.id || `call_${idx}_${Math.random().toString(36).slice(2, 8)}`,
        type: 'function',
        function: {
          name: tc.function.name,
          arguments: tc.function.arguments,
        },
      }));
      const assistantMessage: Message = {
        role: 'assistant',
        content: assistantContent,
        // See note above: caveman never persists the thought, only suppresses
        // its display; the model still reasons. Matches ollama.ts.
        ...(!caveman && assistantThinking ? { thinking: assistantThinking } : {}),
        tool_calls: finalizedToolCalls,
        created_at: Date.now(),
      };
      requestCtx.push(assistantMessage);
      if (isolated) ctx.push(assistantMessage);

      if (toolCallCount + finalizedToolCalls.length > MAX_TOOL_CALLS_PER_RUN) {
        send('error', { message: 'Reached the maximum number of tool calls per run. Stopping to protect budget.' });
        break;
      }
      toolCallCount += finalizedToolCalls.length;
      let allToolsOk = true;
      for (const toolCall of finalizedToolCalls) {
        const name = toolCall.function.name as ToolName;
        let parsedArgs: Record<string, string> = {};
        try {
          const raw = JSON.parse(toolCall.function.arguments || '{}');
          if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
            parsedArgs = Object.fromEntries(
              Object.entries(raw as Record<string, unknown>).map(([k, v]) => [k, typeof v === 'string' ? v : JSON.stringify(v)]),
            );
          }
        } catch {}
        send('tool_start', { name, args: parsedArgs });
        const running = activeRuns.get(sessionId);
        if (running) {
          running.status = 'tool';
          running.statusText = `Running ${name}…`;
        }
        const startMs = Date.now();
        let result: { success: boolean; output: string };
        try {
          result = name === 'browse_url' && !isBrowseUrlAllowed(parsedArgs.url, requestCtx.getMessages())
            ? {
              success: false,
              output: `Browse denied: ${parsedArgs.url} was not returned by a successful search and was not supplied by the user. Do not guess URLs; use only search-result links already in the context.`,
            }
            : await runTool(name, parsedArgs, sessionId);
        } catch (err: any) {
          result = { success: false, output: err?.message ?? String(err) };
        }
        const durationMs = Date.now() - startMs;
        if (!result.success) {
          allToolsOk = false;
          toolFailureCount += 1;
          if (name === 'web_search') searchFailureCount += 1;
        }
        const truncatedOutput = result.output.length > TOOL_CONTEXT_CHAR_LIMIT
          ? `${result.output.slice(0, TOOL_CONTEXT_CHAR_LIMIT)}\n[output truncated at ${TOOL_CONTEXT_CHAR_LIMIT} chars]`
          : result.output;
        send('tool_result', {
          name,
          args: parsedArgs,
          success: result.success,
          output: truncatedOutput,
          durationMs,
        });
        const toolMessage: Message = {
          role: 'tool',
          tool_call_id: toolCall.id,
          tool_name: name,
          content: truncatedOutput,
          duration_ms: durationMs,
          created_at: Date.now(),
        };
        requestCtx.push(toolMessage);
        if (isolated) ctx.push(toolMessage);
        const resumed = activeRuns.get(sessionId);
        if (resumed) {
          resumed.status = 'thinking';
          resumed.statusText = 'Processing results…';
        }
        await saveSessionToFile(sessionId, ctx, selectedPersona, model, provider);
      }
      if (!allToolsOk && (toolFailureCount >= MAX_TOOL_FAILURES_PER_RUN || searchFailureCount >= MAX_SEARCH_FAILURES_PER_RUN)) {
        forceFinalResponse = true;
      }
    }
    send('done', {});
  } catch (err: any) {
    if (err?.name === 'AbortError') {
      send('stopped', {});
    } else {
      console.error('[Cloud chat stream error]', err);
      send('error', { message: err?.message ?? 'Cloud chat failed.' });
    }
  } finally {
    abortControllers.delete(sessionId);
    activeRuns.delete(sessionId);
    await saveSessionToFile(sessionId, ctx, selectedPersona, model, provider);
    res.end();
  }
});

// /api/cloud/stop — cancel the active run for a session.
router.post('/stop', (req: Request, res: Response) => {
  const { sessionId } = req.body as { sessionId?: string };
  if (!sessionId) {
    res.status(400).json({ error: 'sessionId is required.' });
    return;
  }
  const ac = abortControllers.get(sessionId);
  if (ac) ac.abort();
  const run = activeRuns.get(sessionId);
  if (run) {
    run.status = 'thinking';
    run.statusText = 'Stopped';
  }
  res.json({ ok: true });
});

// /api/cloud/suggest-context — best-fit context size for a given model id.
router.post('/suggest-context', (req: Request, res: Response) => {
  const { modelContextLength } = req.body as { modelContextLength?: number };
  const pick = suggestContextSize(
    typeof modelContextLength === 'number' ? modelContextLength : undefined,
    CONTEXT_SIZES,
  );
  res.json({ suggested: pick ?? null });
});

export default router;

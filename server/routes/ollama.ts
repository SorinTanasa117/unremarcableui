import { Router, Request, Response } from 'express';
import { spawn } from 'child_process';
import fsNative from 'fs';
import fs from 'fs/promises';
import path from 'path';
import { ContextManager, Message, ToolCall } from '../lib/contextManager.js';
import { runTool, ToolName } from '../lib/toolRunner.js';
import { TOOL_DEFINITIONS, WORKSPACE_DIR } from '../lib/tools.js';
import * as novelStorage from '../lib/novel/storage.js';
import { inferenceDispatcher } from '../lib/inferenceDispatcher.js';

const router = Router();
const OLLAMA_URL = process.env.OLLAMA_URL ?? 'http://127.0.0.1:11434';
const LLAMACPP_URL = process.env.LLAMACPP_URL ?? 'http://127.0.0.1:8080';
// Keep the full working context by default. Deployments can override this
// without code changes through OLLAMA_NUM_CTX.
const OLLAMA_NUM_CTX = Number.parseInt(process.env.OLLAMA_NUM_CTX ?? '262144', 10);
const LLAMACPP_NUM_CTX = Number.parseInt(process.env.LLAMACPP_NUM_CTX ?? '32768', 10);
// Isolated story packets are capped well below this. A smaller KV cache avoids
// reserving a 262k-token window for every prose continuation.
const STORY_NUM_CTX = Number.parseInt(process.env.STORY_NUM_CTX ?? '16384', 10);
const OLLAMA_KEEP_ALIVE = process.env.OLLAMA_KEEP_ALIVE ?? '15m';
const parsedNumPredict = Number.parseInt(process.env.OLLAMA_NUM_PREDICT ?? '1536', 10);
const OLLAMA_NUM_PREDICT = Number.isFinite(parsedNumPredict) && parsedNumPredict > 0 ? parsedNumPredict : 1536;
const parsedNovelDraftNumPredict = Number.parseInt(process.env.NOVEL_DRAFT_NUM_PREDICT ?? '8192', 10);
const NOVEL_DRAFT_NUM_PREDICT = Number.isFinite(parsedNovelDraftNumPredict) && parsedNovelDraftNumPredict > 0
  ? parsedNovelDraftNumPredict
  : 8192;
const NOVEL_DRAFT_MAX_CONTINUATIONS = 3;
const parsedToolContextCharLimit = Number.parseInt(process.env.TOOL_CONTEXT_CHAR_LIMIT ?? '8000', 10);
const TOOL_CONTEXT_CHAR_LIMIT = Number.isFinite(parsedToolContextCharLimit) && parsedToolContextCharLimit > 0 ? parsedToolContextCharLimit : 8000;
const CONTEXT_SIZES = new Set([16_384, 32_768, 65_536, 131_072, 262_144]);
// A 4–7 source report normally needs a few searches and a handful of source
// reads. Keeping this small prevents CPU-only models from spending an hour
// reprocessing a growing tool transcript.
const MAX_TOOL_CALLS_PER_RUN = 10;
const MAX_TOOL_FAILURES_PER_RUN = 4;
const MAX_SEARCH_FAILURES_PER_RUN = 2;
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
 * Ollama's separate `thinking` field. Remove those tags from answer content
 * while forwarding their inner text through the thinking callback.
 */
function createVisibleContentFilter(onThinking?: (content: string) => void) {
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
          const safeLength = Math.max(0, input.length - (closeTag.length - 1));
          if (safeLength > 0) onThinking?.(input.slice(0, safeLength));
          pending = input.slice(safeLength);
          return visible;
        }
        if (closeAt > 0) onThinking?.(input.slice(0, closeAt));
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

// Parse tool-call XML emitted as visible model content.
function parseFallbackToolCalls(content: string): ToolCall[] {
  const calls: ToolCall[] = [];
  let nextId = 0;
  const addCall = (name: string, args: Record<string, unknown>) => {
    if (!FALLBACK_TOOL_NAMES.has(name as ToolName)) return;
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
  return calls;
}

function removeFallbackToolCallMarkup(content: string): string {
  return content
    .replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, '')
    .replace(/<function=[A-Za-z_][\w-]*(?:\s*>)?[\s\S]*?<\/function>/gi, '')
    .replace(/<\/tool_call>/gi, '');
}

function sanitizeNoToolsAssistantContent(content: string): string {
  if (!content) return '';
  const withoutThink = content
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<\/?think>/gi, '');
  const withoutMarkup = removeFallbackToolCallMarkup(withoutThink);
  const toolCallLine = /^\s*(web_search|browse_url|write_file|read_file|run_terminal)\s*\(.+\)\s*$/i;
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
  const mapContextLength: number | undefined = mapDef?.max_context;
  const mapSystemPromptType: string | undefined = mapDef?.system_prompt_type;

  if (backend === 'llamacpp') {
    const capabilities = {
      supportsTools: false,
      supportsThinking: false,
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

  if (systemPromptType?.toLowerCase() !== 'mistral') return outboundMessages;

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

// Default persona fallbacks in case file read fails
const FALLBACK_PERSONAS: Record<string, string> = {
  coder: `You are an elite software engineering assistant. Your goal is to write correct, idiomatic, and highly optimized code. Provide complete, production-ready solutions and explain your structural choices concisely. You have full access to workspace file utilities and terminal commands. Use them to write, test, and verify your implementations.`,
  researcher: `You are a systematic data intelligence researcher. Your goal is to search, verify, and compile accurate information using web search and URL browsing tools. Always cross-reference facts, outline sources, and present findings in clean, structured markdown tables and summaries. Prioritize objective data and analytical depth. For a 4–7 source request, make at most 3 broad searches, then inspect 4–7 valid result URLs and write the report. Only browse URLs returned by a successful search or explicitly supplied by the user. If a search or page request fails, do not guess URLs or repeat the failed request; clearly report the limitation and provide only verifiable partial findings.`,
  creative: `You are a creative writer specializing in atmospheric dark fantasy, gothic mystery, and dramatic literature. Your style is rich, poetic, and immersive, focusing on dark aesthetics, emotional complexity, suspense, and vivid sensory details. Create compelling narratives, dramatic dialogue, and atmospheric settings while adhering to standard creative writing limits.`,
  system: `You are a general-purpose AI assistant and local model advisor. Answer user questions directly and honestly without hallucinating. Recommend the best installed model (Ollama or GPU-accelerated) for their task. Research unknown topics using web search and browsing tools when your knowledge is insufficient. Admit uncertainty and verify facts rather than guessing.`
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
    } else if (personaKey === 'system') {
      rule = '\n\nTool-use guidance: You have access to web_search, browse_url, and read_file tools. Use them proactively to research unknown topics, verify uncertain claims, and recommend models based on current knowledge. For model recommendations, consider task complexity, speed requirements, and available VRAM. Admit when you don\'t know rather than guessing.';
    } else {
      rule = '\n\nTool-use mandate: You MUST use the provided tools (write_file, run_terminal, read_file) to perform all file and command operations. NEVER describe, narrate, or simulate writing a file or running a command in plain text - always emit the actual tool call. If you support reasoning/thinking, place internal planning inside <think>...</think> tags; your visible response must contain only tool calls and brief status lines.';
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
      rule = '\n\nTool-use mandate: You MUST use the provided tools (write_file, run_terminal, read_file) to perform all file and command operations. NEVER describe, narrate, or simulate writing a file or running a command in plain text - always emit the actual tool call. If you support reasoning/thinking, place internal planning inside <think>...</think> tags; your visible response must contain only tool calls and brief status lines.';
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
router.get('/models', async (req: Request, res: Response) => {
  const backend = resolveInferenceBackend(req.query.backend);
  const backendUrl = getBackendUrl(backend);
  try {
    if (backend === 'llamacpp') {
      const runtimeLoadedModel = await getLlamaCppLoadedModelId();
      const runtimeFriendly = runtimeLoadedModel ? await resolveFriendlyLlamaModel(runtimeLoadedModel) : null;
      const localModels = await listLocalOllamaManifestModels();
      const models = await Promise.all(localModels.map(async (model) => {
        const isLoaded = runtimeFriendly?.mapId === model.name;
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
        });
      }

      res.json({ models });
      return;
    }

    const resp = await fetch(`${OLLAMA_URL}/api/tags`);
    if (!resp.ok) {
      const detail = await resp.text();
      res.status(resp.status).json({ error: detail || resp.statusText });
      return;
    }
    res.json(await resp.json());
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
  const { sessionId, model, message, pendingBrowse, persona, isolated, numCtx, think, numThread, caveman, inferenceBackend } = req.body as {
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
  };
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
  if (!CONTEXT_SIZES.has(requestedContext)) {
    res.status(400).json({ error: 'Context must be one of 16k, 32k, 64k, 128k, or 262k.' });
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
  }

  if (!userMessage) {
    if (pendingBrowse) {
      userMessage = { role: 'user', content: `[User approved browse: ${pendingBrowse}]`, created_at: Date.now() };
    } else if (message) {
      userMessage = { role: 'user', content: message, created_at: Date.now() };
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
  let iterationNumber = 0;
  let novelDraftContinuationCount = 0;
  let novelDraftText = '';
  let ollamaRequestStartedAt = 0;
  let currentOllamaPhase: 'ollama_request' | 'ollama_stream' = 'ollama_request';
  let currentSystemPromptType: string | undefined;
  let currentAllowTools = false;
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
    let iterating = true;
    // When research tools become unavailable, make one final model pass with
    // tools disabled so it can write a candid partial report from successful
    // results already in the conversation.
    let forceFinalResponse = false;
    let finalResponseRetryCount = 0;
    while (iterating && !ac.signal.aborted) {
      iterationNumber += 1;
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
        { type: 'function', function: { name: 'web_search',   description: 'Brave Search; DuckDuckGo only if Brave reports exhausted credits. Returns results.', parameters: { type: 'object', properties: { query:    { type: 'string' } }, required: ['query'] } } },
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
      if (backend === 'llamacpp') {
        modelMessages.push({
          role: 'system' as const,
          content: 'No tools are available in this runtime. Do not emit tool calls, XML tags, function-like commands, or <think> blocks. Reply directly to the user in plain markdown.',
        });
      }

      const normalizedMessages = normalizeOllamaMessages(modelMessages, capabilities.systemPromptType);
      let assistantContent = '';
      let assistantThinking = '';
      const pendingToolCalls: any[] = [];

      if (backend === 'ollama') {
        const activeToolDefs = caveman ? CAVEMAN_TOOL_DEFINITIONS : TOOL_DEFINITIONS;
        const body = {
          model,
          messages: normalizedMessages,
          stream: true,
          // Supported by current Ollama releases; prevents models with optional
          // reasoning from spending their visible response on a hidden plan.
          think: Boolean(think),
          options: {
            num_ctx: activeContext,
            num_predict: outputNumPredict,
            ...(requestedNumThread !== undefined ? { num_thread: requestedNumThread } : {}),
          },
          keep_alive: OLLAMA_KEEP_ALIVE,
          ...(allowTools ? { tools: activeToolDefs } : {}),
        };

        ollamaRequestStartedAt = Date.now();
        currentOllamaPhase = 'ollama_request';
        const ollamaResp = await fetchOllamaChat(body, ac.signal);

        if (!ollamaResp.ok) {
          await recordOllamaFailure('ollama_http', new Error('Ollama returned a non-OK response'), ollamaResp);
          const detail = await ollamaResp.text();
          // A model may have been removed after this chat was saved. Clear its
          // persisted selection so reopening the chat cannot keep retrying it.
          if (ollamaResp.status === 404) {
            await saveSessionToFile(sessionId, ctx, selectedPersona, '');
          }
          send('error', { message: `Ollama error: ${ollamaResp.status} ${detail || ollamaResp.statusText}` });
          break;
        }

        currentOllamaPhase = 'ollama_stream';
        const appendThinking = (content: string) => {
          if (!content) return;
          assistantThinking += content;
          if (!caveman) {
            send('thinking', { content });
          }
          send('tokens', requestCtx.getTokenUsage([{
            role: 'assistant',
            content: assistantContent,
            thinking: assistantThinking,
            tool_calls: pendingToolCalls,
          }]));
        };
        const visibleContent = createVisibleContentFilter(appendThinking);

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
                pendingPartialContent = assistantContent;
                send('token', { content });
                send('tokens', requestCtx.getTokenUsage([{ role: 'assistant', content: assistantContent }]));
                void saveProgress(assistantContent);
              }
            }

            if (msg.thinking) {
              markGenerationStarted();
              appendThinking(msg.thinking);
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

          const deltaContent = parsed?.choices?.[0]?.delta?.content;
          const messageContent = parsed?.choices?.[0]?.message?.content;
          const contentChunk = typeof deltaContent === 'string'
            ? deltaContent
            : typeof messageContent === 'string'
              ? messageContent
              : '';
          if (!contentChunk) return;
          rawAssistantContent += contentChunk;
          pendingPartialContent = rawAssistantContent;
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
        send('tokens', requestCtx.getTokenUsage([{ role: 'assistant', content: assistantContent }]));
        await saveProgress(assistantContent, true);
      }

      if (allowTools && pendingToolCalls.length === 0) {
        const fallbackToolCalls = parseFallbackToolCalls(assistantContent);
        if (fallbackToolCalls.length > 0) {
          pendingToolCalls.push(...fallbackToolCalls);
          assistantContent = removeFallbackToolCallMarkup(assistantContent);
        }
      }

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
    abortControllers.delete(sessionId);
    activeRuns.delete(sessionId);
    await saveSessionToFile(sessionId, ctx, selectedPersona, model);
    res.end();
  }
});

// NOTE: llama.cpp server is started on-demand when user selects iGPU backend,
// not on server boot. This saves ~10+ GB RAM when using Ollama backend.

export default router;

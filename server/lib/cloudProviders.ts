/**
 * Cloud LLM provider registry.
 *
 * Both OpenRouter and Factory.ai expose an OpenAI-compatible
 * `/chat/completions` and `/models` endpoint, so we treat them as
 * variations of the same wire format. Each provider just contributes:
 *
 *   - baseUrl     → origin of the OpenAI-compatible API
 *   - apiKeyEnv   → optional environment variable that holds a default key
 *   - defaultHeaders → extra headers attached to every request
 *                     (OpenRouter requires HTTP-Referer and X-Title;
 *                      Factory.ai requires Authorization: Bearer)
 *
 * Model lists are fetched with the user-supplied API key (which is never
 * persisted server-side), cached for one hour, and exposed through the
 * `/api/cloud/models` endpoint. The list is also used by the UI to populate
 * the model dropdown next to the API-key input.
 */

import { createHash } from 'crypto';
import { inferenceDispatcher } from './inferenceDispatcher.js';

export type CloudProviderId = 'openrouter' | 'factory';

export interface CloudProviderDescriptor {
  id: CloudProviderId;
  label: string;
  /** Short user-facing description shown next to the provider picker. */
  description: string;
  baseUrl: string;
  /** Optional environment variable that can pre-fill the API key field. */
  apiKeyEnv: string;
  /** Default headers added to every request, on top of `Authorization`. */
  defaultHeaders?: Record<string, string>;
  /** Hint displayed under the API key input. */
  keyHint: string;
}

export interface CloudModelInfo {
  id: string;
  name: string;
  contextLength?: number;
  /** Whether the model advertises tool/function support. */
  supportsTools?: boolean;
  /** Provider-internal raw record, useful for advanced UI. */
  raw?: unknown;
}

export const CLOUD_PROVIDERS: Record<CloudProviderId, CloudProviderDescriptor> = {
  openrouter: {
    id: 'openrouter',
    label: 'OpenRouter',
    description:
      'Unified gateway for OpenAI, Anthropic, Google, Meta, and dozens of open models. Single API key, pay-as-you-go pricing.',
    baseUrl: process.env.OPENROUTER_BASE_URL?.trim() || 'https://openrouter.ai/api/v1',
    apiKeyEnv: 'OPENROUTER_API_KEY',
    defaultHeaders: {
      // OpenRouter ranks free apps and routes traffic accordingly. Safe static
      // values are used here; users running their own deployment can override
      // these in the registry below if needed.
      'HTTP-Referer': 'https://localhost/unremarcable-ui',
      'X-Title': 'Unremarkable UI',
    },
    keyHint: 'Get an OpenRouter key at https://openrouter.ai/keys',
  },
  factory: {
    id: 'factory',
    label: 'Factory.ai',
    description:
      'Factory BYOK-compatible OpenAI chat endpoint. Use a Factory deployment or compatible gateway that exposes /v1/models and /v1/chat/completions.',
    baseUrl: process.env.FACTORY_OPENAI_BASE_URL?.trim() || 'https://api.factory.ai/v1',
    apiKeyEnv: 'FACTORY_API_KEY',
    keyHint: 'Get a Factory.ai key at https://app.factory.ai/settings/api-keys',
  },
};

export const CLOUD_PROVIDER_LIST: CloudProviderDescriptor[] = [
  CLOUD_PROVIDERS.openrouter,
  CLOUD_PROVIDERS.factory,
];

interface ProviderCacheEntry {
  expiresAt: number;
  models: CloudModelInfo[];
}

const MODEL_CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour, per product decision.
const providerCache = new Map<string, ProviderCacheEntry>();

function cacheKey(provider: CloudProviderId, apiKey: string): string {
  // Keep credentials out of cache keys and diagnostics. Full-key hashing also
  // prevents two keys sharing the same prefix from receiving stale models.
  const digest = createHash('sha256').update(apiKey).digest('hex');
  return `${provider}:${digest}`;
}

export function isCloudProviderId(value: unknown): value is CloudProviderId {
  return value === 'openrouter' || value === 'factory';
}

/** Throw a sanitized error so we never echo the API key back to the caller. */
function providerError(provider: CloudProviderId, status: number, detail: string, apiKey: string): Error {
  const safeDetail = detail
    .replaceAll(apiKey, '[redacted]')
    .replace(/(?:sk|key|token|fct)[-_][A-Za-z0-9_-]+/gi, '[redacted]')
    .slice(0, 500);
  return new Error(`${CLOUD_PROVIDERS[provider].label} error (${status}): ${safeDetail}`);
}

async function readOpenAiListModels(
  provider: CloudProviderId,
  descriptor: CloudProviderDescriptor,
  apiKey: string,
  signal: AbortSignal,
): Promise<CloudModelInfo[]> {
  const response = await fetch(`${descriptor.baseUrl}/models`, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      ...(descriptor.defaultHeaders ?? {}),
    },
    signal,
    dispatcher: inferenceDispatcher,
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw providerError(provider, response.status, detail || response.statusText, apiKey);
  }

  const json = (await response.json()) as { data?: Array<Record<string, unknown>> };
  const items = Array.isArray(json.data) ? json.data : [];

  return items
    .map((item): CloudModelInfo | null => {
      const id = typeof item.id === 'string' ? item.id : null;
      if (!id) return null;
      const name = typeof item.name === 'string' && item.name.trim().length > 0 ? item.name : id;
      const contextLength = typeof item.context_length === 'number'
        ? item.context_length
        : (typeof item.max_context_length === 'number' ? item.max_context_length : undefined);
      const supportsTools = typeof item.supports_tools === 'boolean'
        ? item.supports_tools
        : Array.isArray(item.supported_parameters)
          ? item.supported_parameters.includes('tools') || item.supported_parameters.includes('tool_choice')
          : undefined;
      return {
        id,
        name,
        contextLength,
        supportsTools,
        raw: item,
      };
    })
    .filter((item): item is CloudModelInfo => item !== null)
    .sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * Factory.ai's `/models` endpoint follows the same OpenAI shape but each
 * model can include a `tools` capability flag. We surface that explicitly so
 * the UI can warn when a user picks a non-tool model for the Coder persona.
 */
async function readFactoryListModels(
  descriptor: CloudProviderDescriptor,
  apiKey: string,
  signal: AbortSignal,
): Promise<CloudModelInfo[]> {
  const response = await fetch(`${descriptor.baseUrl}/models`, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      ...(descriptor.defaultHeaders ?? {}),
    },
    signal,
    dispatcher: inferenceDispatcher,
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw providerError('factory', response.status, detail || response.statusText, apiKey);
  }

  const json = (await response.json()) as { data?: Array<Record<string, unknown>> };
  const items = Array.isArray(json.data) ? json.data : [];

  return items
    .map((item): CloudModelInfo | null => {
      const id = typeof item.id === 'string' ? item.id : null;
      if (!id) return null;
      const name = typeof item.display_name === 'string' && item.display_name.trim().length > 0
        ? item.display_name
        : (typeof item.name === 'string' && item.name.trim().length > 0 ? item.name : id);
      const contextLength = typeof item.context_length === 'number'
        ? item.context_length
        : (typeof item.max_context_length === 'number' ? item.max_context_length : undefined);
      const supportsTools = Array.isArray(item.tools)
        ? (item.tools as unknown[]).length > 0
        : typeof item.supports_tools === 'boolean'
          ? item.supports_tools
          : undefined;
      return {
        id,
        name,
        contextLength,
        supportsTools,
        raw: item,
      };
    })
    .filter((item): item is CloudModelInfo => item !== null)
    .sort((a, b) => a.name.localeCompare(b.name));
}

export async function listCloudModels(
  provider: CloudProviderId,
  apiKey: string,
  options: { signal?: AbortSignal; forceRefresh?: boolean } = {},
): Promise<{ models: CloudModelInfo[]; cached: boolean }> {
  if (!apiKey || apiKey.trim().length === 0) {
    throw new Error('API key is required to list models.');
  }

  const key = cacheKey(provider, apiKey);
  const now = Date.now();
  if (!options.forceRefresh) {
    const cached = providerCache.get(key);
    if (cached && cached.expiresAt > now) {
      return { models: cached.models, cached: true };
    }
  }

  const descriptor = CLOUD_PROVIDERS[provider];
  const ac = new AbortController();
  options.signal?.addEventListener('abort', () => ac.abort());
  const models = provider === 'factory'
    ? await readFactoryListModels(descriptor, apiKey, ac.signal)
    : await readOpenAiListModels(provider, descriptor, apiKey, ac.signal);

  providerCache.set(key, { models, expiresAt: now + MODEL_CACHE_TTL_MS });
  return { models, cached: false };
}

export async function validateCloudKey(
  provider: CloudProviderId,
  apiKey: string,
  signal?: AbortSignal,
): Promise<{ ok: true; modelCount: number } | { ok: false; status?: number; message: string }> {
  try {
    const { models } = await listCloudModels(provider, apiKey, { signal, forceRefresh: true });
    return { ok: true, modelCount: models.length };
  } catch (err: any) {
    if (err?.name === 'AbortError') {
      return { ok: false, message: 'Validation was cancelled.' };
    }
    const message = err instanceof Error ? err.message : String(err);
    const statusMatch = message.match(/\((\d{3})\)/);
    const status = statusMatch ? Number(statusMatch[1]) : undefined;
    return { ok: false, status, message };
  }
}

/**
 * Pick the most appropriate context size for a given model. The chat UI
 * offers a fixed set of context sizes; we clamp to the largest one that fits
 * inside the model's reported context window.
 */
export function suggestContextSize(
  modelContextLength: number | undefined,
  candidates: number[],
): number | undefined {
  if (!modelContextLength || modelContextLength <= 0) return undefined;
  const sorted = [...candidates].sort((a, b) => a - b);
  let pick: number | undefined;
  for (const size of sorted) {
    if (size <= modelContextLength) pick = size;
  }
  return pick;
}

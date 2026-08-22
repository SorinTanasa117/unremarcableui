import { useEffect, useState } from 'react';

export type CloudProviderId = 'openrouter' | 'factory';

export interface CloudModel {
  id: string;
  name: string;
  contextLength?: number;
  supportsTools?: boolean;
}

export interface CloudProviderSettings {
  provider: CloudProviderId;
  apiKey: string;
  model: string;
}

export interface CloudProviderMetadata {
  id: CloudProviderId;
  label: string;
  description: string;
  keyHint: string;
  hasEnvKey: boolean;
}

const STORAGE_KEY = 'ollama_cloud_provider_settings';

const DEFAULT_SETTINGS: CloudProviderSettings = {
  provider: 'openrouter',
  apiKey: '',
  model: '',
};

function readSettings(): CloudProviderSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw) as Partial<CloudProviderSettings>;
    return {
      provider: parsed.provider === 'factory' ? 'factory' : 'openrouter',
      apiKey: typeof parsed.apiKey === 'string' ? parsed.apiKey : '',
      model: typeof parsed.model === 'string' ? parsed.model : '',
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function useCloudProviderSettings() {
  const [settings, setSettings] = useState<CloudProviderSettings>(readSettings);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      // Keep settings usable in memory when browser storage is unavailable.
    }
  }, [settings]);

  return { settings, setSettings };
}

export function isCloudProviderId(value: unknown): value is CloudProviderId {
  return value === 'openrouter' || value === 'factory';
}

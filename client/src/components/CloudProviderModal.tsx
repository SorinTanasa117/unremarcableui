import React, { useEffect, useMemo, useState } from 'react';
import type {
  CloudModel,
  CloudProviderId,
  CloudProviderMetadata,
  CloudProviderSettings,
} from '../lib/providerConfig';

interface Props {
  open: boolean;
  disabled?: boolean;
  settings: CloudProviderSettings;
  onSave: (settings: CloudProviderSettings) => void;
  onClose: () => void;
}

const FALLBACK_PROVIDERS: CloudProviderMetadata[] = [
  {
    id: 'openrouter',
    label: 'OpenRouter',
    description: 'OpenAI-compatible gateway for many model suppliers.',
    keyHint: 'https://openrouter.ai/keys',
    hasEnvKey: false,
  },
  {
    id: 'factory',
    label: 'Factory.ai',
    description: 'Factory BYOK-compatible OpenAI chat endpoint.',
    keyHint: 'https://app.factory.ai/settings/api-keys',
    hasEnvKey: false,
  },
];

export function CloudProviderModal({ open, disabled, settings, onSave, onClose }: Props) {
  const [draft, setDraft] = useState(settings);
  const [providers, setProviders] = useState<CloudProviderMetadata[]>(FALLBACK_PROVIDERS);
  const [models, setModels] = useState<CloudModel[]>([]);
  const [showKey, setShowKey] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState<{ type: 'info' | 'error' | 'success'; text: string } | null>(null);

  const selectedProvider = useMemo(
    () => providers.find((provider) => provider.id === draft.provider) ?? FALLBACK_PROVIDERS[0],
    [draft.provider, providers],
  );

  useEffect(() => {
    if (!open) return;
    setDraft(settings);
    setModels([]);
    setMessage(null);
    setShowKey(false);
    void fetch('/api/cloud/providers')
      .then((response) => response.ok ? response.json() : Promise.reject(new Error('Unable to load providers')))
      .then((data: { providers?: CloudProviderMetadata[] }) => {
        if (Array.isArray(data.providers) && data.providers.length > 0) setProviders(data.providers);
      })
      .catch(() => {
        // Static fallback keeps the settings panel usable when server starts late.
      });
  }, [open, settings]);

  if (!open) return null;

  const loadModels = async () => {
    setLoadingModels(true);
    setMessage(null);
    try {
      const response = await fetch('/api/cloud/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: draft.provider, apiKey: draft.apiKey }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `Model request failed (${response.status})`);
      const nextModels = Array.isArray(data.models) ? data.models as CloudModel[] : [];
      setModels(nextModels);
      setDraft((current) => ({
        ...current,
        model: nextModels.some((model) => model.id === current.model)
          ? current.model
          : (nextModels[0]?.id ?? current.model),
      }));
      setMessage({
        type: 'success',
        text: `${nextModels.length} model${nextModels.length === 1 ? '' : 's'} loaded${data.cached ? ' from cache' : ''}.`,
      });
    } catch (error: any) {
      setMessage({ type: 'error', text: error?.message || 'Unable to load models.' });
    } finally {
      setLoadingModels(false);
    }
  };

  const testConnection = async () => {
    setTesting(true);
    setMessage(null);
    try {
      const response = await fetch('/api/cloud/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: draft.provider, apiKey: draft.apiKey }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `Connection failed (${response.status})`);
      setMessage({ type: 'success', text: `Connection valid. ${data.modelCount ?? 0} models available.` });
      await loadModels();
    } catch (error: any) {
      setMessage({ type: 'error', text: error?.message || 'Connection test failed.' });
    } finally {
      setTesting(false);
    }
  };

  const save = () => {
    const model = draft.model.trim();
    if (!model) {
      setMessage({ type: 'error', text: 'Choose a model or enter a model ID.' });
      return;
    }
    if (!draft.apiKey.trim() && !selectedProvider.hasEnvKey) {
      setMessage({ type: 'error', text: 'Enter an API key.' });
      return;
    }
    onSave({ ...draft, apiKey: draft.apiKey.trim(), model });
    onClose();
  };

  const updateProvider = (provider: CloudProviderId) => {
    setDraft((current) => ({ ...current, provider, model: '' }));
    setModels([]);
    setMessage(null);
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="cloud-provider-title"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !disabled) onClose();
      }}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 50,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
        background: 'rgba(0, 0, 0, 0.65)',
      }}
    >
      <div
        style={{
          width: 'min(620px, 100%)',
          maxHeight: '90vh',
          overflow: 'auto',
          padding: 22,
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-lg)',
          background: 'var(--bg-surface)',
          boxShadow: '0 20px 70px rgba(0,0,0,0.45)',
        }}
      >
        <div className="flex items-center justify-between" style={{ marginBottom: 16 }}>
          <div>
            <h2 id="cloud-provider-title" style={{ margin: 0, fontSize: 18, color: 'var(--text-primary)' }}>
              Cloud provider
            </h2>
            <p style={{ margin: '5px 0 0', color: 'var(--text-muted)', fontSize: 12 }}>
              API key stays in this browser and is sent only when you use cloud chat.
            </p>
          </div>
          <button className="btn btn-icon" onClick={onClose} disabled={disabled} aria-label="Close cloud settings">✕</button>
        </div>

        <label className="flex flex-col" style={{ gap: 6, marginBottom: 14 }}>
          <span style={{ color: 'var(--text-secondary)', fontSize: 12 }}>Provider</span>
          <select
            value={draft.provider}
            onChange={(event) => updateProvider(event.target.value as CloudProviderId)}
            disabled={disabled}
            style={{
              padding: '9px 10px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-primary)',
            }}
          >
            {providers.map((provider) => <option key={provider.id} value={provider.id}>{provider.label}</option>)}
          </select>
          <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>{selectedProvider.description}</span>
        </label>

        <label className="flex flex-col" style={{ gap: 6, marginBottom: 14 }}>
          <span style={{ color: 'var(--text-secondary)', fontSize: 12 }}>API key</span>
          <div className="flex items-center" style={{ gap: 6 }}>
            <input
              type={showKey ? 'text' : 'password'}
              value={draft.apiKey}
              onChange={(event) => setDraft((current) => ({ ...current, apiKey: event.target.value }))}
              placeholder={selectedProvider.hasEnvKey ? 'Using server environment key if blank' : 'Paste provider API key'}
              disabled={disabled}
              autoComplete="off"
              spellCheck={false}
              style={{
                flex: 1,
                minWidth: 0,
                padding: '9px 10px',
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-mono)',
              }}
            />
            <button className="btn" onClick={() => setShowKey((value) => !value)} disabled={disabled} type="button">
              {showKey ? 'Hide' : 'Show'}
            </button>
          </div>
          <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            {selectedProvider.keyHint}
          </span>
        </label>

        <div className="flex items-end" style={{ gap: 8, marginBottom: 8 }}>
          <label className="flex flex-col" style={{ flex: 1, minWidth: 0, gap: 6 }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: 12 }}>Model</span>
            <select
              value={models.some((model) => model.id === draft.model) ? draft.model : ''}
              onChange={(event) => setDraft((current) => ({ ...current, model: event.target.value }))}
              disabled={disabled || loadingModels || models.length === 0}
              style={{
                padding: '9px 10px',
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)',
                color: 'var(--text-primary)',
              }}
            >
              <option value="">{loadingModels ? 'Loading models…' : models.length ? 'Select a model' : 'Load models first'}</option>
              {models.map((model) => (
                <option key={model.id} value={model.id}>{model.name === model.id ? model.id : `${model.name} (${model.id})`}</option>
              ))}
            </select>
          </label>
          <button className="btn" onClick={() => void loadModels()} disabled={disabled || loadingModels} type="button">
            {loadingModels ? 'Loading…' : 'Refresh'}
          </button>
        </div>

        <label className="flex flex-col" style={{ gap: 6, marginBottom: 16 }}>
          <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>Or enter model ID manually</span>
          <input
            value={draft.model}
            onChange={(event) => setDraft((current) => ({ ...current, model: event.target.value }))}
            placeholder="provider/model-name"
            disabled={disabled}
            spellCheck={false}
            style={{
              padding: '9px 10px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-mono)',
            }}
          />
        </label>

        {message && (
          <div style={{
            marginBottom: 14,
            padding: '9px 10px',
            borderRadius: 'var(--radius-md)',
            background: message.type === 'error' ? 'rgba(239,68,68,0.12)' : message.type === 'success' ? 'rgba(34,197,94,0.12)' : 'var(--bg-card)',
            color: message.type === 'error' ? 'var(--red, #ef4444)' : message.type === 'success' ? 'var(--green, #22c55e)' : 'var(--text-secondary)',
            fontSize: 12,
          }}>
            {message.text}
          </div>
        )}

        <div style={{
          marginBottom: 16,
          padding: '9px 10px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(245,166,35,0.10)',
          color: 'var(--text-secondary)',
          fontSize: 11,
          lineHeight: 1.5,
        }}>
          Browser localStorage is convenient, not a secure vault. Clear cloud settings after use on shared computers.
        </div>

        <div className="flex items-center justify-between" style={{ gap: 8 }}>
          <button className="btn" onClick={() => void testConnection()} disabled={disabled || testing} type="button">
            {testing ? 'Testing…' : 'Test connection'}
          </button>
          <div className="flex items-center" style={{ gap: 8 }}>
            <button className="btn" onClick={onClose} disabled={disabled} type="button">Cancel</button>
            <button className="btn btn-primary" onClick={save} disabled={disabled} type="button">Save cloud settings</button>
          </div>
        </div>
      </div>
    </div>
  );
}

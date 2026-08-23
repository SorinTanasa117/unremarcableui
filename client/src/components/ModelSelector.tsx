import React, { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { InferenceBackend } from '../hooks/useOllamaStream';

interface Props {
  value: string;
  onChange: (model: string) => void;
  disabled?: boolean;
  persona: 'coder' | 'researcher' | 'creative' | 'system' | 'novelist';
  backend: InferenceBackend;
}

interface OllamaModel {
  name: string;
  map_id?: string;
  display_name?: string;
  sycl_available?: boolean;
  sycl_reason?: string;
  size: number;
  modified_at: string;
  is_loaded?: boolean;
}

interface LoadedResponse {
  loaded: boolean;
  models: Array<{
    name: string;
    model?: string;
    size?: number;
    sizeVram?: number;
    expiresAt?: string;
  }>;
  backend: string;
}

interface ModelMapEntry {
  id: string;
  name: string;
  size_gb?: number;
  quantization?: string;
  recommended_use?: string;
  capabilities?: {
    function_calling?: boolean;
  };
}

function modelAllowedForPersona(
  mapDef: ModelMapEntry | undefined,
  persona: 'coder' | 'researcher' | 'creative' | 'system' | 'novelist',
): boolean {
  // Creative persona is prose-only, any model is allowed.
  if (persona === 'creative') return true;
  // Coder/Researcher/System personas can use tools, require explicit tool support.
  return Boolean(mapDef?.capabilities?.function_calling);
}

export function ModelSelector({ value, onChange, disabled, persona, backend }: Props) {
  const { data, isLoading, isError } = useQuery<{ models: OllamaModel[] }>({
    queryKey: ['ollama-models', backend],
    queryFn: async () => {
      const res = await fetch(`/api/ollama/models?backend=${backend}`);
      if (!res.ok) throw new Error('Failed to fetch models');
      return res.json();
    },
    staleTime: 60_000,
  });

  const { data: mapData } = useQuery<{ default_model?: string; models?: ModelMapEntry[] }>({
    queryKey: ['model-map'],
    queryFn: async () => {
      const res = await fetch('/api/ollama/model-map');
      if (!res.ok) throw new Error('Failed to fetch model map');
      return res.json();
    },
    staleTime: 60_000,
  });

  const { data: loadedData } = useQuery<LoadedResponse>({
    queryKey: ['ollama-loaded', backend],
    queryFn: async () => {
      const res = await fetch(`/api/ollama/loaded?backend=${backend}`);
      if (!res.ok) throw new Error('Failed to fetch loaded model');
      return res.json();
    },
    refetchInterval: 5_000,
  });

  const models = data?.models ?? [];
  const modelMap = mapData?.models ?? [];
  const loadedModels = loadedData?.models ?? [];

  // Check if currently selected model is loaded
  const isSelectedLoaded = loadedData?.loaded && loadedModels.some(
    (lm) => lm.name === value || (lm.model && lm.model.startsWith(value)) || (value && lm.name.startsWith(value)),
  );

  // Filter models based on persona
  const filteredModels = models.filter((m) => {
    const mapDef = modelMap.find((item) => item.id === (m.map_id ?? m.name));
    return modelAllowedForPersona(mapDef, persona);
  });

  const displayModels = persona === 'creative'
    ? models
    : filteredModels;
  // Alphabetical by displayed label so the dropdown reads predictably.
  const sortedDisplayModels = [...displayModels].sort((a, b) => {
    const nameA = a.display_name ?? a.name;
    const nameB = b.display_name ?? b.name;
    return nameA.localeCompare(nameB, undefined, { sensitivity: 'base', numeric: true });
  });
  const selectableModels = backend === 'llamacpp'
    ? sortedDisplayModels.filter((m) => m.sycl_available !== false)
    : sortedDisplayModels;

  // Sync selected model: if active model is not in the list of displayModels, 
  // auto-select the first available one to avoid invalid selection.
  useEffect(() => {
    if (isLoading || models.length === 0) return;
    
    // Check if the current value is valid for this persona
    const candidateModels = selectableModels.length > 0 ? selectableModels : displayModels;
    const isValid = candidateModels.some((m) => m.name === value);
    if (!isValid && candidateModels.length > 0) {
      // Find default model from modelMap if defined
      const defaultId = mapData?.default_model;
      const defaultModel = candidateModels.find((m) => (m.map_id ?? m.name) === defaultId);
      if (defaultModel && defaultId) {
        onChange(defaultModel.name);
      } else {
        onChange(candidateModels[0].name);
      }
    } else if (!isValid && candidateModels.length === 0 && value) {
      onChange('');
    }
  }, [displayModels, selectableModels, value, onChange, isLoading, models, mapData]);

  // Session metadata can outlive an Ollama model that the user has removed.
  // Do not keep submitting that stale name just because the chat is reopened.
  useEffect(() => {
    if (!data || !value) return;
    if (!models.some((model) => model.name === value)) onChange('');
  }, [data, models, onChange, value]);

  return (
    <div className="flex items-center gap-2">
      <span style={{ fontSize: 12, color: 'var(--text-muted)', flexShrink: 0 }}>Model</span>
      <select
        id="model-selector"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled || isLoading}
        style={{
          minWidth: 180,
          maxWidth: 280,
          padding: '5px 10px',
          fontSize: 12,
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          color: 'var(--text-primary)',
          cursor: 'pointer',
        }}
      >
        {!isLoading && !isError && displayModels.length > 0 && !value && (
          <option value="">Select a model</option>
        )}
        {isLoading && <option>Loading…</option>}
        {isError && <option>⚠ Runtime unreachable</option>}
        {displayModels.length === 0 && !isLoading && !isError && (
          <option value="">No models found</option>
        )}
        {sortedDisplayModels.map((m) => {
          const mapDef = modelMap.find((item) => item.id === (m.map_id ?? m.name));
          const unavailableInSycl = backend === 'llamacpp' && m.sycl_available === false;
          const isModelLoaded = m.is_loaded || loadedModels.some((lm) => lm.name === m.name || (lm.model && lm.model.startsWith(m.name)));
          const prefix = isModelLoaded ? '● ' : '';
          const displayName = mapDef
            ? `${prefix}${mapDef.name} (${mapDef.size_gb}GB)`
            : unavailableInSycl
              ? `${prefix}${m.display_name ?? m.name} (no local GGUF)`
              : `${prefix}${m.display_name ?? m.name}`;
          const tooltip = mapDef
            ? `${mapDef.recommended_use} | Quant: ${mapDef.quantization}${isModelLoaded ? ' (Resident in VRAM)' : ''}`
            : unavailableInSycl
              ? (m.sycl_reason ?? 'Not available for local llama.cpp runtime (no local GGUF blob found).')
              : (m.display_name ? `Runtime model id: ${m.name}` : 'Runtime model');
          return (
            <option key={m.name} value={m.name} title={tooltip} disabled={unavailableInSycl}>
              {displayName}
            </option>
          );
        })}
      </select>
      {/* Live resident/offloaded status pill */}
      {value && !isLoading && !isError && (
        <span
          title={
            isSelectedLoaded
              ? `Model is currently resident in memory / VRAM (${backend})`
              : loadedData?.loaded
                ? `Another model is resident: ${loadedModels.map((m) => m.name).join(', ')}`
                : `Model is offloaded from memory (${backend})`
          }
          style={{
            fontSize: 11,
            padding: '2px 7px',
            borderRadius: '10px',
            background: isSelectedLoaded
              ? 'rgba(34, 197, 94, 0.12)'
              : loadedData?.loaded
                ? 'rgba(234, 179, 8, 0.12)'
                : 'rgba(148, 163, 184, 0.10)',
            color: isSelectedLoaded
              ? 'var(--green, #22c55e)'
              : loadedData?.loaded
                ? 'var(--amber, #eab308)'
                : 'var(--text-muted, #94a3b8)',
            border: `1px solid ${
              isSelectedLoaded
                ? 'rgba(34, 197, 94, 0.3)'
                : loadedData?.loaded
                  ? 'rgba(234, 179, 8, 0.3)'
                  : 'rgba(148, 163, 184, 0.2)'
            }`,
            display: 'inline-flex',
            alignItems: 'center',
            gap: 4,
            whiteSpace: 'nowrap',
            userSelect: 'none',
            flexShrink: 0,
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: isSelectedLoaded
                ? '#22c55e'
                : loadedData?.loaded
                  ? '#eab308'
                  : '#94a3b8',
            }}
          />
          {isSelectedLoaded ? 'In VRAM' : loadedData?.loaded ? 'Other Loaded' : 'Offloaded'}
        </span>
      )}
    </div>
  );
}

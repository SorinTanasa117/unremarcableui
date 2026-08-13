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

  const models = data?.models ?? [];
  const modelMap = mapData?.models ?? [];

  // Filter models based on persona
  const filteredModels = models.filter((m) => {
    const mapDef = modelMap.find((item) => item.id === (m.map_id ?? m.name));
    return modelAllowedForPersona(mapDef, persona);
  });

  const displayModels = persona === 'creative'
    ? models
    : filteredModels;
  const selectableModels = backend === 'llamacpp'
    ? displayModels.filter((m) => m.sycl_available !== false)
    : displayModels;

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
        {displayModels.map((m) => {
          const mapDef = modelMap.find((item) => item.id === (m.map_id ?? m.name));
          const unavailableInSycl = backend === 'llamacpp' && m.sycl_available === false;
          const displayName = mapDef
            ? `${mapDef.name} (${mapDef.size_gb}GB)`
            : unavailableInSycl
              ? `${m.display_name ?? m.name} (no local GGUF)`
              : (m.display_name ?? m.name);
          const tooltip = mapDef
            ? `${mapDef.recommended_use} | Quant: ${mapDef.quantization}`
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
    </div>
  );
}

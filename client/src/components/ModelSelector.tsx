import React, { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';

interface Props {
  value: string;
  onChange: (model: string) => void;
  disabled?: boolean;
  persona: 'coder' | 'researcher' | 'creative';
}

interface OllamaModel {
  name: string;
  size: number;
  modified_at: string;
}

function getModelCategories(model: any): string[] {
  const categories: string[] = [];
  const recUse = (model.recommended_use || '').toLowerCase();
  const name = (model.name || '').toLowerCase();
  const id = (model.id || '').toLowerCase();

  // Coding Category
  if (
    id.includes('code') || 
    id.includes('devstral') || 
    id.includes('laguna') || 
    id.includes('gpt') ||
    name.includes('code') || 
    name.includes('devstral') || 
    name.includes('laguna') || 
    name.includes('gpt') ||
    recUse.includes('code') || 
    recUse.includes('script') || 
    recUse.includes('refactoring') || 
    recUse.includes('programming') || 
    recUse.includes('engineering')
  ) {
    categories.push('coder');
  }

  // Research Category
  if (
    id.includes('research') || 
    id.includes('r1') || 
    id.includes('orchestrator') || 
    id.includes('gpt') ||
    name.includes('research') || 
    name.includes('orchestrator') || 
    name.includes('gpt') ||
    recUse.includes('research') || 
    recUse.includes('agentic') || 
    recUse.includes('reasoning') || 
    recUse.includes('extraction') || 
    recUse.includes('document') || 
    recUse.includes('stem') || 
    recUse.includes('math') || 
    recUse.includes('logic') || 
    recUse.includes('planning') || 
    recUse.includes('tool')
  ) {
    categories.push('researcher');
  }

  // Writing / Creative Category
  if (
    id.includes('laguna') || 
    id.includes('virtuoso') || 
    name.includes('laguna') || 
    name.includes('virtuoso') || 
    recUse.includes('writing') || 
    recUse.includes('story') || 
    recUse.includes('roleplay') || 
    recUse.includes('creative') || 
    recUse.includes('prose') || 
    recUse.includes('text') ||
    recUse.includes('instruction') ||
    recUse.includes('chat')
  ) {
    categories.push('creative');
  }

  // Fallbacks:
  if (categories.length === 0) {
    if (model.capabilities?.function_calling) {
      categories.push('coder', 'researcher');
    } else {
      categories.push('creative');
    }
  }

  return categories;
}

export function ModelSelector({ value, onChange, disabled, persona }: Props) {
  const { data, isLoading, isError } = useQuery<{ models: OllamaModel[] }>({
    queryKey: ['ollama-models'],
    queryFn: async () => {
      const res = await fetch('/api/ollama/models');
      if (!res.ok) throw new Error('Failed to fetch models');
      return res.json();
    },
    staleTime: 60_000,
  });

  const { data: mapData } = useQuery<any>({
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
    const mapDef = modelMap.find((item: any) => item.id === m.name);
    if (!mapDef) {
      // Show unknown/unlisted models in all categories
      return true;
    }
    const categories = getModelCategories(mapDef);
    return categories.includes(persona);
  });

  // Fallback to all models if none match the current persona
  const displayModels = filteredModels.length > 0 ? filteredModels : models;

  // Sync selected model: if active model is not in the list of displayModels, 
  // auto-select the first available one to avoid invalid selection.
  useEffect(() => {
    if (isLoading || models.length === 0) return;
    
    // Check if the current value is valid for this persona
    const isValid = displayModels.some((m) => m.name === value);
    if (!isValid && displayModels.length > 0) {
      // Find default model from modelMap if defined
      const defaultId = mapData?.default_model;
      const hasDefault = displayModels.some((m) => m.name === defaultId);
      if (hasDefault && defaultId) {
        onChange(defaultId);
      } else {
        onChange(displayModels[0].name);
      }
    }
  }, [displayModels, value, onChange, isLoading, models, mapData]);

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
        {isError && <option>⚠ Ollama unreachable</option>}
        {displayModels.length === 0 && !isLoading && !isError && (
          <option value="">No models found</option>
        )}
        {displayModels.map((m) => {
          const mapDef = modelMap.find((item: any) => item.id === m.name);
          const displayName = mapDef ? `${mapDef.name} (${mapDef.size_gb}GB)` : m.name;
          const tooltip = mapDef ? `${mapDef.recommended_use} | Quant: ${mapDef.quantization}` : 'Ollama Model';
          return (
            <option key={m.name} value={m.name} title={tooltip}>
              {displayName}
            </option>
          );
        })}
      </select>
    </div>
  );
}

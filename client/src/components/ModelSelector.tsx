import React, { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';

interface Props {
  value: string;
  onChange: (model: string) => void;
  disabled?: boolean;
}

interface OllamaModel {
  name: string;
  size: number;
  modified_at: string;
}

export function ModelSelector({ value, onChange, disabled }: Props) {
  const { data, isLoading, isError } = useQuery<{ models: OllamaModel[] }>({
    queryKey: ['ollama-models'],
    queryFn: async () => {
      const res = await fetch('/api/ollama/models');
      if (!res.ok) throw new Error('Failed to fetch models');
      return res.json();
    },
    staleTime: 60_000,
  });

  const models = data?.models ?? [];

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
          maxWidth: 260,
          padding: '5px 10px',
          fontSize: 12,
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          color: 'var(--text-primary)',
          cursor: 'pointer',
        }}
      >
        {!isLoading && !isError && models.length > 0 && !value && (
          <option value="">Select a model</option>
        )}
        {isLoading && <option>Loading…</option>}
        {isError && <option>⚠ Ollama unreachable</option>}
        {models.length === 0 && !isLoading && !isError && (
          <option value="">No models found</option>
        )}
        {models.map((m) => (
          <option key={m.name} value={m.name}>
            {m.name}
          </option>
        ))}
      </select>
    </div>
  );
}

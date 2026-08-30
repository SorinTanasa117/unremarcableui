import React from 'react';
import { getTokenPercentage, formatTokenCount, MAX_TOKENS } from '../lib/tokenCounter';

interface Props {
  input: number;
  output: number;
  contextUsed?: number;
  contextLimit?: number;
}

export function ProgressBar({ input, output, contextUsed, contextLimit }: Props) {
  const total = input + output;
  const activeLimit = contextLimit && contextLimit > 0 ? contextLimit : MAX_TOKENS;
  const hasActiveWindow = contextUsed !== undefined;
  const activeUsed = contextUsed ?? 0;
  const pct = hasActiveWindow ? getTokenPercentage(activeUsed, activeLimit) : 0;
  const color = pct > 90 ? '#f06070' : pct > 70 ? '#f5a623' : '#7c6af7';

  return (
    <div style={{ minWidth: 240 }} aria-label={hasActiveWindow
      ? `Active context used: ${activeUsed} of ${activeLimit} tokens`
      : `Saved session contains approximately ${total} tokens`}>
      <div className="flex items-baseline justify-between gap-3" style={{ marginBottom: 5, fontSize: 11, fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>
        <span style={{ color: 'var(--text-secondary)' }}>
          Session <span style={{ color: 'var(--cyan)' }}>{formatTokenCount(input)}</span>
          {' + '}
          <span style={{ color: 'var(--accent-light)' }}>{formatTokenCount(output)}</span>
        </span>
        <span style={{ color: 'var(--text-muted)' }}>
          {hasActiveWindow
            ? `${formatTokenCount(activeUsed)} / ${formatTokenCount(activeLimit)}`
            : `${formatTokenCount(total)} saved`}
        </span>
      </div>
      <div style={{
        flex: 1,
        height: 6,
        background: 'var(--bg-card)',
        borderRadius: 99,
        overflow: 'hidden',
        border: '1px solid var(--border)',
      }}>
        <div style={{
          height: '100%',
          width: `${pct}%`,
          background: color,
          borderRadius: 99,
          transition: 'width 0.25s ease, background 0.4s ease',
          boxShadow: `0 0 8px ${color}55`,
          position: 'relative',
        }} />
      </div>
    </div>
  );
}

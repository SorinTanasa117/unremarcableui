import React from 'react';
import { getTokenPercentage, formatTokenCount, MAX_TOKENS } from '../lib/tokenCounter';

interface Props {
  input: number;
  output: number;
}

export function ProgressBar({ input, output }: Props) {
  const total = input + output;
  const pct = getTokenPercentage(total);
  const inputPct = getTokenPercentage(input);
  const color = pct > 90 ? '#f06070' : pct > 70 ? '#f5a623' : '#7c6af7';

  return (
    <div style={{ minWidth: 240 }} aria-label={`Context used: ${input} input tokens and ${output} output tokens`}>
      <div className="flex items-baseline justify-between gap-3" style={{ marginBottom: 5, fontSize: 11, fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>
        <span style={{ color: 'var(--text-secondary)' }}>
          In <span style={{ color: 'var(--cyan)' }}>{formatTokenCount(input)}</span> · Out <span style={{ color: 'var(--accent-light)' }}>{formatTokenCount(output)}</span>
        </span>
        <span style={{ color: 'var(--text-muted)' }}>{formatTokenCount(total)} / {formatTokenCount(MAX_TOKENS)}</span>
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
        }}>
          <div style={{
            height: '100%',
            width: `${pct ? (inputPct / pct) * 100 : 0}%`,
            background: 'var(--cyan)',
            opacity: 0.65,
            transition: 'width 0.25s ease',
          }} />
        </div>
      </div>
    </div>
  );
}

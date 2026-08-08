import React, { useEffect, useState } from 'react';
import type { AgentStatus } from '../hooks/useOllamaStream';

interface Props {
  status: AgentStatus;
  statusText: string;
}

const icons: Record<AgentStatus, string> = {
  idle: '●',
  thinking: '◌',
  tool: '⚙',
  stopped: '■',
  error: '✕',
};

const colors: Record<AgentStatus, string> = {
  idle: 'var(--green)',
  thinking: 'var(--accent)',
  tool: 'var(--amber)',
  stopped: 'var(--text-muted)',
  error: 'var(--red)',
};

export function StatusIndicator({ status, statusText }: Props) {
  const color = colors[status];
  const animate = status === 'thinking' || status === 'tool';
  const [elapsed, setElapsed] = useState(0);

  // Live timer for active generations/tool execution
  useEffect(() => {
    if (!animate) {
      setElapsed(0);
      return;
    }

    const start = Date.now();
    setElapsed(0);

    const timer = setInterval(() => {
      setElapsed((Date.now() - start) / 1000);
    }, 100);

    return () => clearInterval(timer);
  }, [animate, status]);

  return (
    <div className="flex items-center gap-2" style={{ fontSize: 12 }}>
      <span
        style={{
          color,
          fontSize: 10,
          animation: animate ? 'pulse-dot 1s ease infinite' : 'none',
          display: 'inline-block',
        }}
      >
        {icons[status]}
      </span>
      {statusText && (
        <span style={{ color: 'var(--text-secondary)', maxWidth: 300 }} className="truncate">
          {statusText}
        </span>
      )}
      {animate && (
        <span style={{
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          marginLeft: 4,
          background: 'rgba(255,255,255,0.03)',
          padding: '1px 5px',
          borderRadius: 4,
          border: '1px solid var(--border)',
        }}>
          {elapsed.toFixed(1)}s
        </span>
      )}
    </div>
  );
}

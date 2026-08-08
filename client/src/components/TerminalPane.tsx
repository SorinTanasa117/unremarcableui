import React, { useRef } from 'react';
import { useTerminal } from '../hooks/useTerminal';
import '@xterm/xterm/css/xterm.css';

interface Props {
  label?: string;
}

export function TerminalPane({ label = 'Terminal' }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { focus } = useTerminal(containerRef);

  return (
    <div className="flex-col" style={{ height: '100%', display: 'flex', background: 'var(--bg-base)' }}>
      <div style={{
        padding: '8px 14px',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
      }}>
        <span className="dot dot-green" />
        <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>
          {label}
        </span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          agent-workspace/
        </span>
      </div>
      <div
        ref={containerRef}
        className="flex-1"
        style={{ overflow: 'hidden', padding: 8 }}
        onClick={focus}
      />
    </div>
  );
}

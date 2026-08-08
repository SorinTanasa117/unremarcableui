import { useEffect, useRef, useCallback } from 'react';
import type { RefObject } from 'react';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import { WebLinksAddon } from '@xterm/addon-web-links';

export function useTerminal(containerRef: RefObject<HTMLDivElement>) {
  const termRef = useRef<Terminal | null>(null);
  const fitRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const term = new Terminal({
      theme: {
        background: '#0b0d12',
        foreground: '#e8ecf4',
        cursor: '#7c6af7',
        selectionBackground: 'rgba(124,106,247,0.3)',
        black: '#1c2236',
        red: '#f06070',
        green: '#39d98a',
        yellow: '#f5a623',
        blue: '#4dabf7',
        magenta: '#9d8fff',
        cyan: '#22d3ee',
        white: '#e8ecf4',
        brightBlack: '#374060',
        brightRed: '#f06070',
        brightGreen: '#39d98a',
        brightYellow: '#f5a623',
        brightBlue: '#4dabf7',
        brightMagenta: '#9d8fff',
        brightCyan: '#22d3ee',
        brightWhite: '#ffffff',
      },
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      fontSize: 13,
      lineHeight: 1.4,
      cursorBlink: true,
      cursorStyle: 'bar',
      allowTransparency: true,
      scrollback: 5000,
    });

    const fit = new FitAddon();
    const links = new WebLinksAddon();
    term.loadAddon(fit);
    term.loadAddon(links);
    term.open(containerRef.current);
    fit.fit();

    termRef.current = term;
    fitRef.current = fit;

    // Connect WebSocket
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/terminal`);
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'output') term.write(msg.data);
        if (msg.type === 'exit') term.write('\r\n\x1b[31m[Session ended]\x1b[0m\r\n');
      } catch {}
    };

    ws.onclose = () => {
      term.write('\r\n\x1b[31m[Disconnected]\x1b[0m\r\n');
    };

    term.onData((data) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'input', data }));
      }
    });

    // Resize observer
    const ro = new ResizeObserver(() => {
      if (!termRef.current) return;
      try {
        fit.fit();
        if (ws.readyState === WebSocket.OPEN) {
          const dims = fit.proposeDimensions();
          if (dims) ws.send(JSON.stringify({ type: 'resize', cols: dims.cols, rows: dims.rows }));
        }
      } catch {}
    });
    if (containerRef.current) ro.observe(containerRef.current);

    return () => {
      termRef.current = null;
      ro.disconnect();
      ws.close();
      try {
        term.dispose();
      } catch {}
    };
  }, []);

  const focus = useCallback(() => termRef.current?.focus(), []);

  return { focus };
}

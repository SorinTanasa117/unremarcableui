import { WebSocketServer, WebSocket } from 'ws';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WORKSPACE_DIR = path.resolve(__dirname, '../../agent-workspace');

// Dynamically load node-pty so failure doesn't crash the REST server
let pty: any = null;
try {
  // Top-level await is fully supported in ESM
  pty = await import('node-pty');
} catch (err: any) {
  console.error('\n⚠️  WARNING: node-pty failed to load. Terminal feature will be disabled.');
  console.error('Error details:', err.message);
  console.error('To fix, run: npm rebuild node-pty\n');
}

export function setupTerminalWS(wss: WebSocketServer) {
  wss.on('connection', (ws: WebSocket, req) => {
    const url = req.url ?? '';

    // User terminal: /ws/terminal
    if (url.includes('/terminal')) {
      if (!pty) {
        ws.send(JSON.stringify({
          type: 'output',
          data: `\r\n\x1b[31m[ERROR] PowerShell terminal is unavailable.\x1b[0m\r\n` +
                `\x1b[33mReason:\x1b[0m node-pty native module could not be loaded.\r\n` +
                `Please check your backend server console logs for details.\r\n` +
                `You can still use the Chat, File Explorer, and Web Search features!\r\n\r\n`
        }));
        ws.close();
        return;
      }

      let ptyProcess: any = null;

      try {
        ptyProcess = pty.spawn('powershell.exe', [], {
          name: 'xterm-color',
          cols: 120,
          rows: 36,
          cwd: WORKSPACE_DIR,
          env: process.env as Record<string, string>,
        });

        ptyProcess.onData((data: string) => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'output', data }));
          }
        });

        ptyProcess.onExit(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'exit' }));
          }
        });
      } catch (err: any) {
        console.error('❌ Failed to spawn PTY terminal process:', err);
        ws.send(JSON.stringify({
          type: 'output',
          data: `\r\n\x1b[31m[ERROR] Failed to start PowerShell terminal.\x1b[0m\r\n` +
                `\x1b[33mReason:\x1b[0m ${err.message}\r\n\r\n`
        }));
        ws.close();
        return;
      }

      ws.on('message', (raw) => {
        try {
          const msg = JSON.parse(raw.toString());
          if (msg.type === 'input') {
            ptyProcess?.write(msg.data);
          } else if (msg.type === 'resize') {
            ptyProcess?.resize(msg.cols, msg.rows);
          }
        } catch {}
      });

      ws.on('close', () => {
        try { ptyProcess?.kill(); } catch {}
      });
    }
  });
}

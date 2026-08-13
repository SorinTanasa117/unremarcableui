import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import ollamaRouter from './routes/ollama.js';
import filesRouter from './routes/files.js';
import toolsRouter from './routes/tools.js';
import novelsRouter from './routes/novels.js';
import { setupTerminalWS } from './routes/terminal.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = 5050;

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Global request logger
app.use((req, _res, next) => {
  console.log(`[Express API] ${req.method} ${req.url}`);
  next();
});

// REST routes
app.use('/api/ollama', ollamaRouter);
app.use('/api/files', filesRouter);
app.use('/api/tools', toolsRouter);
app.use('/api/novels', novelsRouter);

// Health check
app.get('/api/health', (_req, res) => res.json({ ok: true }));

const server = http.createServer(app);

// Agent responses can stream through multiple model and tool turns. Node defaults
// request timeouts to five minutes, which prematurely ends longer runs.
const AGENT_RUN_TIMEOUT_MS = 30 * 60 * 1000;
server.requestTimeout = AGENT_RUN_TIMEOUT_MS;
server.timeout = AGENT_RUN_TIMEOUT_MS;

// WebSocket server — route manually on upgrade
const wss = new WebSocketServer({ noServer: true });
setupTerminalWS(wss);

server.on('upgrade', (request, socket, head) => {
  const url = new URL(request.url || '', `http://${request.headers.host || 'localhost'}`);
  if (url.pathname.startsWith('/ws')) {
    wss.handleUpgrade(request, socket, head, (ws) => {
      wss.emit('connection', ws, request);
    });
  } else {
    socket.destroy();
  }
});

server.on('error', (err: any) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\n❌ ERROR: Port ${PORT} is already in use by another process.`);
    console.error(`Please kill the process using port ${PORT} or change the port in server/index.ts.\n`);
  } else {
    console.error('❌ Server error:', err);
  }
  process.exit(1);
});

server.listen(PORT, () => {
  console.log(`\n🚀 Ollama Agent UI server running on http://localhost:${PORT}\n`);
});

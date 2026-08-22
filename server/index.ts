import 'dotenv/config';
import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import ollamaRouter from './routes/ollama.js';
import cloudRouter from './routes/cloud.js';
import filesRouter from './routes/files.js';
import toolsRouter from './routes/tools.js';
import novelsRouter from './routes/novels.js';
import { setupTerminalWS } from './routes/terminal.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number.parseInt(process.env.PORT ?? '5050', 10);

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
app.use('/api/cloud', cloudRouter);
app.use('/api/files', filesRouter);
app.use('/api/tools', toolsRouter);
app.use('/api/novels', novelsRouter);

// Health check
app.get('/api/health', (_req, res) => res.json({ ok: true }));

const server = http.createServer(app);

// Agent responses can stream through multiple model and tool turns without arbitrary time limits.
// Setting requestTimeout, timeout, and headersTimeout to 0 disables Node's default timeouts.
server.requestTimeout = 0;
server.timeout = 0;
server.headersTimeout = 0;
server.keepAliveTimeout = 0;

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

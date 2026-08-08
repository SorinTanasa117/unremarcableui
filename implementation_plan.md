# Ollama Agent UI — Implementation Plan

## Overview

A standalone, full-featured local AI agent interface built with **React + TypeScript + Express**, running entirely on Windows (no WSL/Docker). It connects to a local Ollama server, gives the agent web browsing capabilities (DuckDuckGo + Playwright), file system write access to a configurable working directory, and presents a premium dark-mode UI to the user.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   React Frontend (Vite)                  │
│  Chat Panel │ File Explorer │ Terminal │ Progress Panel  │
└──────────────────────┬──────────────────────────────────┘
                       │  WebSocket + REST API
┌──────────────────────▼──────────────────────────────────┐
│               Express Backend (Node.js / TSX)            │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Ollama    │  │  Tool Runner │  │  File System API │ │
│  │  API Proxy │  │  (Playwright)│  │  (agent-workspace│ │
│  │  (stream)  │  │  (DDG search)│  │   read/write)    │ │
│  └────────────┘  └──────────────┘  └──────────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │       PTY Terminal (node-pty → WebSocket)        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  Ollama Server  │
              │  localhost:11434│
              └─────────────────┘
```

---

## Features

### ✅ Phase 1 — Core (this build)

| Feature | Details |
|---|---|
| **Prompt intake** | Multi-line textarea with keyboard shortcuts |
| **Ollama streaming chat** | SSE/WebSocket stream from Ollama's `/api/chat` |
| **Tool-use loop** | Agent calls tools: `web_search`, `browse_url`, `write_file`, `read_file`, `run_terminal` |
| **DuckDuckGo search** | Free `ddg-search` API — no key needed |
| **Playwright browsing** | Headless Chromium scrapes pages the agent requests |
| **File write to disk** | Agent writes files to `<project>/agent-workspace/` |
| **File preview + edit** | Monaco-style editor pane with syntax highlighting (CodeMirror) |
| **File explorer** | Tree view of agent-workspace directory |
| **User terminal** | xterm.js + node-pty PTY shell in the browser, cwd = agent-workspace |
| **Agent backend terminal** | Separate xterm.js pane showing the agent's own command executions |
| **Progress reports** | Lightweight status bar: `🔍 Searching...`, `📝 Writing code...`, `🌐 Browsing example.com — proceed?` |
| **Stop / Resume** | Abort controller stops the agent mid-stream; resume queues a continuation |
| **Token counter** | Character-based estimate shown as a progress bar (X / 262,144 tokens) |
| **Model switcher** | Dropdown showing all `ollama list` models; locked while agent is running |
| **Memory + auto-compaction** | Keep rolling context ≤ 262k tokens; summarize oldest messages when limit approached |
| **Dark premium UI** | Glassmorphism panels, animated status indicators, Inter font |

---

## Project Structure

```
ollama-agent-ui/
├── client/
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css
│       ├── components/
│       │   ├── ChatPanel.tsx          # Main chat + streaming
│       │   ├── MessageBubble.tsx      # Individual message
│       │   ├── ProgressBar.tsx        # Token counter
│       │   ├── StatusIndicator.tsx    # Progress reports
│       │   ├── FileExplorer.tsx       # File tree
│       │   ├── FileEditor.tsx         # CodeMirror editor pane
│       │   ├── TerminalPane.tsx       # xterm.js pane
│       │   ├── ModelSelector.tsx      # Ollama model dropdown
│       │   └── ConfirmDialog.tsx      # "Can I proceed?" modal
│       ├── hooks/
│       │   ├── useOllamaStream.ts     # WebSocket chat hook
│       │   ├── useTerminal.ts         # PTY terminal hook
│       │   └── useFileSystem.ts       # File API hook
│       └── lib/
│           ├── tokenCounter.ts        # Token estimation
│           └── queryClient.ts
├── server/
│   ├── index.ts                       # Express + WS server
│   ├── routes/
│   │   ├── ollama.ts                  # Ollama proxy + tool loop
│   │   ├── files.ts                   # File system CRUD
│   │   ├── terminal.ts                # PTY WebSocket
│   │   └── tools.ts                   # DuckDuckGo + Playwright
│   └── lib/
│       ├── toolRunner.ts              # Tool dispatch logic
│       ├── contextManager.ts          # 262k rolling context
│       └── playwright.ts              # Browser automation
├── agent-workspace/                   # Agent file output directory
├── package.json
├── tsconfig.json
└── vite.config.ts
```

---

## Key Technical Decisions

### Tool-Use Protocol
The agent uses Ollama's native `tools` API (available in models like `qwen2.5`, `llama3.1`, `mistral-nemo`). The backend runs a **tool-call loop**: stream response → if tool call detected → execute tool → feed result back → continue streaming. A `stop` flag is checked between each loop iteration.

### DuckDuckGo Search
Uses the free `duck-duck-scrape` npm package (no API key). Returns organic results with titles, URLs, and snippets.

### Playwright
Runs as a Node subprocess using `@playwright/test` or `playwright-core` with Chromium. The backend launches a browser instance on server start and keeps it alive for reuse, closing pages after each tool call.

### PTY Terminal (User)
Uses `node-pty` to spawn a real `cmd.exe` or `powershell.exe` shell, bridged to the browser via WebSocket + xterm.js. CWD starts in `agent-workspace/`.

### Token Counter
Uses `tiktoken` (cl100k_base) for accurate token counting. The 262k limit is tracked across all messages in the context. When >90% full, the oldest non-system messages are summarized by the model itself.

### Stop / Resume
- **Stop**: Sets a server-side flag, aborts the Ollama fetch stream. The current partial response is saved.
- **Resume**: Sends a special `[RESUME]` message that continues from where the agent left off using the saved context.

---

## Open Questions

> [!IMPORTANT]
> **Ollama URL**: Assuming `http://localhost:11434`. Is your Ollama running on a different host or port?

> [!IMPORTANT]
> **Playwright prompt**: When the agent wants to browse a URL, a confirmation dialog will appear ("Agent wants to visit `example.com` — Proceed?"). This matches your requirement of "can I proceed" style reporting.

> [!NOTE]
> **Token model**: The 262k limit will be tracked using character estimation (≈4 chars/token) for speed, with an option to switch to exact `tiktoken` counting (slower). Which do you prefer?

> [!NOTE]
> **Shell**: The user terminal will default to `cmd.exe`. PowerShell is also supported — which would you prefer?

---

## Dependencies to Install

```
npm packages:
- playwright (Playwright for browsing)
- duck-duck-scrape (DuckDuckGo search)
- node-pty (PTY terminal)
- xterm + xterm-addon-fit (browser terminal)
- @codemirror/... (code editor)
- ws (WebSocket server)
- tiktoken (token counting)
```

---

## Verification Plan

### Automated
- `npx tsc --noEmit` — type check passes
- `npm run dev` — server starts on port 5050

### Manual
1. UI loads in browser at `http://localhost:5050`
2. Model dropdown populates from Ollama
3. Prompt submitted → streaming response appears
4. Agent uses web_search → DuckDuckGo results appear in progress panel
5. Agent writes a file → file appears in file explorer
6. Stop button halts the agent
7. User terminal opens a real shell in agent-workspace

import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const WORKSPACE_DIR = path.resolve(__dirname, '../../agent-workspace');

// Tool definitions sent to Ollama
export const TOOL_DEFINITIONS = [
  {
    type: 'function',
    function: {
      name: 'web_search',
      description: 'Search the web using Brave Search. If Brave explicitly reports exhausted credits, fall back to DuckDuckGo. Return titles, URLs, and snippets.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'The search query' },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'browse_url',
      description: 'Browse a URL with a headless browser and return the visible text content of the page.',
      parameters: {
        type: 'object',
        properties: {
          url: { type: 'string', description: 'The URL to visit' },
        },
        required: ['url'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'write_file',
      description: 'Write content to a file inside the agent-workspace directory.',
      parameters: {
        type: 'object',
        properties: {
          filepath: { type: 'string', description: 'Relative path within agent-workspace (e.g. src/app.py)' },
          content: { type: 'string', description: 'The file content to write' },
        },
        required: ['filepath', 'content'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'edit_file',
      description: 'Fix or change part of an existing file WITHOUT rewriting the whole file. Replaces one exact, unique occurrence of old_str with new_str. Prefer this over write_file for edits and bug fixes.',
      parameters: {
        type: 'object',
        properties: {
          filepath: { type: 'string', description: 'Relative path within agent-workspace' },
          old_str: { type: 'string', description: 'Exact text to replace. Must match once and be unique; include surrounding context if needed.' },
          new_str: { type: 'string', description: 'Replacement text.' },
        },
        required: ['filepath', 'old_str', 'new_str'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: 'Read the content of a file inside the agent-workspace directory.',
      parameters: {
        type: 'object',
        properties: {
          filepath: { type: 'string', description: 'Relative path within agent-workspace' },
        },
        required: ['filepath'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'run_terminal',
      description: 'Run a shell command inside the agent-workspace directory (Windows/cmd.exe) and return stdout+stderr. Do NOT start a long-running server here as a normal command (e.g. "node server.js", "npm start", "vite") and do NOT chain it with "&" — it will be auto-detached. To test a server: start it in one call, then probe it (curl http://localhost:PORT/...) in a SEPARATE call.',
      parameters: {
        type: 'object',
        properties: {
          command: { type: 'string', description: 'The command to run (e.g. "npm install"). Windows cmd.exe syntax.' },
        },
        required: ['command'],
      },
    },
  },
];
